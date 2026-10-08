import dataclasses

import mujoco
import numpy as np
import pytest
import torch

from mandahands.embodiments import registry
from mandahands.embodiments.descriptors import ACT_FIELDS, JOINT_FIELDS, Normalizer, SupportBank, describe, noisy
from mandahands.policy.tokenizer import TokenTable
from mandahands.sim.scripted import probe

J, A = {f: i for i, f in enumerate(JOINT_FIELDS)}, {f: i for i, f in enumerate(ACT_FIELDS)}


def toy(flip=False, quat=(1, 0, 0, 0)):
  """One two-link finger curling toward +y against a fixed post; distal joint mimics the proximal at 0.5.

  flip: the source authors both joints (and the actuator) with the opposite sign convention.
  quat: the whole hand rotated under the wrist (a different source frame).
  """
  sg = -1 if flip else 1
  s = mujoco.MjSpec()
  s.compiler.degree = False
  mount = s.worldbody.add_body(name="hand_mount")
  palm = mount.add_body(name="palm", quat=list(quat))
  palm.add_geom(name="palm_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.01, 0.01], pos=[0, 0, -0.01])
  palm.add_geom(name="post", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0, 0], fromto=[0, 0.05, -0.02, 0, 0.05, -0.08])
  rng = lambda lo, hi: [lo, hi] if not flip else [-hi, -lo]
  b0 = palm.add_body(name="l0", pos=[0, 0, -0.02])
  j0 = b0.add_joint(name="j0", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[sg, 0, 0], range=rng(0, 1.5), damping=0.02)
  b0.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0, 0], fromto=[0, 0, 0, 0, 0, -0.04], mass=0.02)
  b1 = b0.add_body(name="l1", pos=[0, 0, -0.04])
  j1 = b1.add_joint(name="j1", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[sg, 0, 0], range=rng(0, 1.2))
  b1.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0, 0], fromto=[0, 0, 0, 0, 0, -0.03], mass=0.01)
  s.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="j1", name2="j0", data=[0, 0.5, 0, 0, 0] + [0] * 6)
  a = s.add_actuator(name="a0", target="j0", trntype=mujoco.mjtTrn.mjTRN_JOINT)
  a.set_to_position(kp=2.0, kv=0.1)
  a.ctrlrange, a.ctrllimited = rng(0, 1.5), True
  a.forcerange, a.forcelimited = [-0.5, 0.5], True
  a.delay, a.nsample = 0.01, 8
  return s


def test_toy_known_values():
  pr = probe(toy())
  d = describe(toy(), [dict(name="a0", velocity=4.0, backlash=0.01, source="measured")], pr)
  # canonical z: palm centre of mass -> open fingertip (tilted toward the post here); y: closing direction (+y)
  assert (d.frame @ np.array([0, 0, -1.0]))[2] > 0.9 and (d.frame @ np.array([0, 1.0, 0]))[1] > 0.9
  j0, j1 = d.joint
  np.testing.assert_allclose(j0[J["pos_x"] : J["pos_x"] + 3], d.frame @ ([0, 0, -0.02] - pr["palm"]), atol=1e-6)  # palm -> joint
  np.testing.assert_allclose(j0[J["axis_x"] : J["axis_x"] + 3], d.frame @ [1.0, 0, 0], atol=1e-6)
  np.testing.assert_allclose(j0[[J["lo"], J["hi"], J["range"]]], [0, 1.5, 1.5])
  assert j0[J["depth"]] == 0 and j0[J["n_below"]] == 1 and j1[J["depth"]] == 1 and j1[J["n_below"]] == 0
  assert j0[J["link_mass"]] == pytest.approx(0.02) and j0[J["damping"]] == pytest.approx(0.02)
  np.testing.assert_allclose(d.coupling, [[1.0, 0.5]])  # mimic follower in the actuator's coupling row
  a = d.act[0]
  np.testing.assert_allclose(a[[A["kp"], A["kv"], A["force_close"], A["force_open"], A["ctrl_span"], A["latency"]]],
                             [2.0, 0.1, 0.5, 0.5, 1.5, 0.01])
  assert a[A["velocity"]] == 4.0 and a[A["backlash"]] == 0.01 and a[A["n_driven"]] == 2 and a[A["src_measured"]] == 1


@pytest.mark.parametrize("variant", [dict(flip=True), dict(quat=(0.7071068, 0, 0.7071068, 0)), dict(flip=True, quat=(0, 1, 0, 0))])
def test_invariant_to_source_frame_and_sign_convention(variant):
  ref = describe(toy())
  d = describe(toy(**variant))
  np.testing.assert_allclose(d.joint, ref.joint, atol=1e-6)
  np.testing.assert_allclose(d.act, ref.act, atol=1e-6)
  np.testing.assert_allclose(d.coupling, ref.coupling, atol=1e-9)
  if variant.get("flip"):
    assert (d.joint_sign == -1).all() and (d.act_sign == -1).all()


def _mixed():
  """Toy plus GENHAND hands with different DoF and coupling: full, tendon_pull, synergy."""
  man = registry.manifest()
  topo = lambda h: man["topologies"][h["t"]]
  pick = []
  for want in ("full", "tendon_pull", "synergy"):
    pick.append(next(h["id"] for h in man["hands"] if h.get("split") == "train" and topo(h)["coupling"] == want))
  return [describe(toy())] + [registry.descriptor(i) for i in pick]


def test_tokens_batch_across_dof_and_coupling():
  descs = _mixed()
  assert len({(len(d.joint), len(d.act)) for d in descs}) >= 3  # really different sizes
  norm = Normalizer.fit(descs)
  tab = TokenTable(descs, norm)
  emb = torch.tensor([0, 1, 2, 3, 2, 0])
  st = tab.static(emb)
  B = len(emb)
  assert st["joint"].shape == (B, tab.J, len(JOINT_FIELDS)) and st["act"].shape == (B, tab.A, len(ACT_FIELDS))
  assert st["coupling"].shape == (B, tab.A, tab.J)
  for b, e in enumerate(emb.tolist()):
    d = descs[e]
    nj, na = len(d.joint), len(d.act)
    assert st["joint_mask"][b].sum() == nj and st["act_mask"][b].sum() == na
    assert (st["joint"][b, nj:] == 0).all() and (st["act"][b, na:] == 0).all()
    np.testing.assert_allclose(st["coupling"][b, :na, :nj].numpy(), d.coupling, atol=1e-6)
    assert (st["coupling"][b, na:] == 0).all() and (st["coupling"][b, :, nj:] == 0).all()
    np.testing.assert_allclose(st["joint"][b, :nj].numpy(), norm.joint(d.joint), atol=1e-5)

  # actions: +1 = closed command, -1 = open command; padded actuators stay 0; round trip
  ones = torch.ones(B, tab.A)
  for b, e in enumerate(emb.tolist()):
    na = len(descs[e].act)
    np.testing.assert_allclose(tab.to_ctrl(emb, ones)[b, :na].numpy(), descs[e].ctrl_close, atol=1e-5)
    np.testing.assert_allclose(tab.to_ctrl(emb, -ones)[b, :na].numpy(), descs[e].ctrl_open, atol=1e-5)
    assert (tab.to_ctrl(emb, ones)[b, na:] == 0).all()
  a = torch.rand(B, tab.A) * 2 - 1
  np.testing.assert_allclose(tab.to_action(emb, tab.to_ctrl(emb, a)).numpy(), (a * tab.act_mask[emb]).numpy(), atol=1e-5)

  # joint state: canonical q at the closing limit -> +1, padded slots 0
  q = torch.zeros(B, tab.J)
  for b, e in enumerate(emb.tolist()):
    d = descs[e]
    q[b, : len(d.joint)] = torch.as_tensor(d.joint_sign * d.joint[:, J["hi"]])
  js = tab.joint_state(emb, q, torch.zeros_like(q))
  assert js.shape == (B, tab.J, 4)
  for b, e in enumerate(emb.tolist()):
    nj = len(descs[e].joint)
    np.testing.assert_allclose(js[b, :nj, 0].numpy(), 1.0, atol=1e-5)
    assert (js[b, nj:] == 0).all()
  assert tab.act_state(emb, a, torch.zeros(B, tab.A), torch.zeros(B, tab.A), torch.zeros(B, tab.A)).shape == (B, tab.A, 3)


def test_normalizer_noise_and_support(tmp_path):
  descs = _mixed()
  norm = Normalizer.fit(descs)
  z = np.concatenate([norm.joint(d.joint) for d in descs])
  varying = [i for i, f in enumerate(JOINT_FIELDS) if norm.stats["joint"]["std"][i] != 1.0]
  np.testing.assert_allclose(z[:, varying].mean(0), 0, atol=1e-6)
  np.testing.assert_allclose(z[:, varying].std(0), 1, atol=1e-6)
  norm.save(tmp_path / "s.json")
  np.testing.assert_allclose(Normalizer.load(tmp_path / "s.json").act(descs[1].act), norm.act(descs[1].act))

  d = descs[1]
  n = noisy(d, np.random.default_rng(0), 0.1)
  i = J["link_mass"]
  assert (np.abs(n.joint[:, i] / d.joint[:, i] - 1) <= 0.1 + 1e-9).all() and not np.allclose(n.joint[:, i], d.joint[:, i])
  np.testing.assert_array_equal(n.coupling, d.coupling)
  np.testing.assert_array_equal(n.joint[:, J["axis_x"]], d.joint[:, J["axis_x"]])

  bank = SupportBank(descs[1:], norm)
  inside = bank.score(descs[1])
  far = dataclasses.replace(descs[1], act=descs[1].act * np.where(np.isin(np.arange(len(ACT_FIELDS)), [A["kp"], A["kv"]]), 1000.0, 1.0))
  assert bank.score(far)["out_of_support"] and bank.score(far)["act_nn"] > 5 * inside["act_nn"]
