import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest

from mandahands.embodiments import registry
from mandahands.embodiments.procedural.generator import build, sample_genome, topology_key
from mandahands.sim.builder import SceneCfg, add_self_collision, build_scene, simple_hand
from mandahands.sim.groups import OFF_SOLIMP, Group, signature, union_pairs


def _class_variants(coupling="mimic", n=3):
  """n GENHAND variants of one topology class (with self-collision pairs), from the committed manifest."""
  man = registry.manifest()
  by = {}
  for h in man["hands"]:
    t = man["topologies"][h["t"]]
    if h["valid"] and t["coupling"] == coupling:
      by.setdefault(topology_key(t), []).append(h)
  hs = next(v for v in by.values() if len(v) >= n)[:n]
  out = []
  for h in hs:
    topo = man["topologies"][h["t"]]
    spec, _ = build(topo, sample_genome(topo, man["seed"], h["t"], h["v"]))
    add_self_collision(spec)
    out.append(spec)
  return out


def _scenes(hands, **kw):
  ms = [build_scene(SceneCfg(), h, wrist_pos=[0, 0, 0.4], obj_pos=[0.5, 0.5, 0.03], **kw).compile() for h in hands]
  return ms, [mujoco.MjData(m) for m in ms]


def test_group_worlds_match_their_variants_on_cpu():
  hands = _class_variants()
  union_pairs(hands)
  ms, ds = _scenes(hands)
  assert len({signature(m) for m in ms}) == 1
  g = Group(ms, ds, nworld=6, njmax=200)
  assert {"body_pos", "geom_size", "body_mass"} <= set(g.batched)  # geometry and dynamics really vary per world
  close = [m.actuator_ctrlrange[:, 1] * 0.8 for m in ms]
  g.d.ctrl.assign(np.stack([close[w % 3] for w in range(6)]).astype(np.float32))
  for m, d, c in zip(ms, ds, close):
    d.ctrl[:] = c
  err = 0.0
  for _ in range(300):  # fingers close in free space (object out of reach); self-collision pairs may engage
    mjw.step(g.m, g.d)
    for m, d in zip(ms, ds):
      mujoco.mj_step(m, d)
    q = g.d.qpos.numpy()
    err = max(err, max(np.abs(q[w] - ds[w % 3].qpos).max() for w in range(6)))
  assert err < 1e-3, err


def test_soft_pair_is_disabled():
  # simple_hand's fingertips meet when fully closed; variant 1 has all its pairs disabled
  a, b = simple_hand(SceneCfg()), simple_hand(SceneCfg())
  for p in b.pairs:
    p.solimp = OFF_SOLIMP
  ms, ds = _scenes([a, b])
  g = Group(ms, ds, nworld=2, njmax=200)
  assert "pair_solimp" in g.batched
  g.d.ctrl.assign(np.tile(ms[0].actuator_ctrlrange[:, 1], (2, 1)).astype(np.float32))
  for _ in range(400):
    mjw.step(g.m, g.d)
  # a disabled pair still reports contacts, but exerts no force: world 1's fingertips pass through each other
  q = g.d.qpos.numpy()
  hand = slice(7, 7 + ms[0].nu)
  assert np.abs(q[0, hand] - q[1, hand]).max() > 0.05  # world 1's fingertips pass through each other


def test_structural_mismatch_rejected():
  ms, ds = _scenes([simple_hand(SceneCfg()), simple_hand(SceneCfg(n_fingers=4))])
  with pytest.raises(ValueError):
    Group(ms, ds, nworld=2)
