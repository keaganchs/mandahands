import mujoco
import numpy as np
import pytest

from mandahands.embodiments.procedural import validity
from mandahands.embodiments.procedural.generator import build, sample_genome, sample_topology
from mandahands.sim.builder import SceneCfg, simple_hand

STRUCT = ("nbody", "njnt", "nu", "neq", "ntendon", "ngeom", "nhistory")


def _hand(t, v=0, seed=0):
  topo = sample_topology(seed, t)
  return topo, *build(topo, sample_genome(topo, seed, t, v))


def test_topologies_compile_and_metadata_matches():
  for t in range(80):
    topo, spec, meta = _hand(t)
    m = spec.compile()
    assert m.nu == len(meta["actuators"]) and m.njnt == len(meta["joints"]), topo
    assert m.neq == len(meta["coupling"]) and m.ntendon == len(meta["tendons"]), topo
    assert (m.actuator_delay > 0).all() or (m.actuator_delay == 0).all()
    if topo["coupling"] == "synergy":
      assert m.nu == 1
    if topo["layout"] == "jaw":
      assert (m.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE).all()


def test_variants_share_structure():
  # a topology class differs only in continuous parameters (what per-world batching can vary, M5)
  for t in range(20):
    ms = [_hand(t, v)[1].compile() for v in range(3)]
    for f in STRUCT:
      assert len({getattr(m, f) for m in ms}) == 1, (t, f)
    assert all((m.geom_type == ms[0].geom_type).all() for m in ms)


def _tip_motion(topo, g, finger):
  """Fingertip displacement in the hand frame when every flexion joint of `finger` moves +0.5 rad."""
  spec, meta = build(topo, g)
  m = spec.compile()
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  tip = max((b for b in range(m.nbody) if m.body(b).name.startswith(f"f{finger}_")), key=lambda b: int(m.body(b).name[-1]))
  p0 = d.xpos[tip].copy()
  for j in meta["joints"]:
    if j["finger"] == finger and j["kind"] == "flex":
      d.qpos[m.jnt_qposadr[m.joint(j["name"]).id]] = 0.5
  mujoco.mj_kinematics(m, d)
  return d.xpos[tip] - p0, p0


def test_curl_directions():
  topo = dict(layout="anthro", n_fingers=3, n_phalanges=2, coupling="full", tip="capsule", thumb="opposed",
              thumb_phalanges=2, abduction="none")
  g = sample_genome(topo, 0, 0, 0)
  assert _tip_motion(topo, g, 0)[0][1] > 0.005  # row finger curls toward the palm front (+y)
  assert _tip_motion(topo, g, 2)[0][1] < -0.005  # opposed thumb curls back toward the fingers (-y)
  topo = dict(layout="radial", n_fingers=3, n_phalanges=2, coupling="full", tip="capsule", thumb="none", abduction="none")
  g = sample_genome(topo, 0, 0, 0)
  for f in range(3):
    dp, p0 = _tip_motion(topo, g, f)
    assert dp[:2] @ p0[:2] < 0  # inward


def test_self_penetration_detected():
  # fingers on a 3 mm circle overlap each other at rest
  m = simple_hand(SceneCfg(obj_half=-0.012)).compile()
  assert validity.self_penetration(m)
  assert not validity.self_penetration(simple_hand(SceneCfg()).compile())


@pytest.mark.parametrize("closes", [True, False])
def test_filter_needs_a_closing_hand(closes):
  topo = dict(layout="radial", n_fingers=3, n_phalanges=1, coupling="full", tip="capsule", thumb="none", abduction="none")
  g = sample_genome(topo, 0, 0, 0)
  if not closes:
    for f in g["fingers"]:
      f["flex_hi"] = [0.05]  # fingers can barely move: nothing gets enclosed
  spec, _ = build(topo, g)
  r = validity.check(spec)
  assert r["valid"] is closes, r.get("ref")


def test_registry_splits():
  from mandahands.embodiments import registry

  man = registry.manifest()
  split_ids = {s: set(registry.ids(s)) for s in ("train", "test_id", "test_ood_topology", "test_ood_scale")}
  assert all(not (a & b) for (s, a) in split_ids.items() for (u, b) in split_ids.items() if s < u)
  assert all(h["valid"] == ("split" in h) for h in man["hands"])
  subsets = [set(registry.ids(f"train_{n}")) for n in man["scaling_subsets"]]
  assert all(a <= b for a, b in zip(subsets, subsets[1:])) and subsets[-1] == split_ids["train"]
  spec, meta = registry.genhand(sorted(split_ids["train"])[0])  # regenerates and checks the hash
  assert spec.compile().npair > 0 and meta["split"] == "train"
