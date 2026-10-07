import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest

from mandahands.sim.batch import ContactOverflow, check_overflow, contact_stats, make_batch
from mandahands.sim.builder import SceneCfg, build_scene, close_ctrl


def _script(mjm, i, mz):
  """Close over 0.5 s, then lift the wrist 10 cm over 1 s starting at 0.7 s."""
  t = i * mjm.opt.timestep
  ctrl = close_ctrl(mjm, min(t / 0.5, 1) * 0.7)
  mocap_z = mz + (min((t - 0.7) / 1.0, 1) * 0.1 if t > 0.7 else 0.0)
  return ctrl, mocap_z


def test_warp_matches_cpu_close_and_lift():
  mjm = build_scene(SceneCfg()).compile()
  d = mujoco.MjData(mjm)
  m, dw = make_batch(mjm, 1, njmax=160)
  mz, oid = d.mocap_pos[0, 2], mjm.body("object").id
  # observed max error over this rollout: ~1e-4 rad hand joints, ~1e-5 m object; tolerances are 10x that
  for i in range(1000):
    d.ctrl[:], d.mocap_pos[0, 2] = _script(mjm, i, mz)
    dw.ctrl.assign(d.ctrl[None].astype(np.float32))
    dw.mocap_pos.assign(d.mocap_pos[None].astype(np.float32))
    mujoco.mj_step(mjm, d)
    mjw.step(m, dw)
    if i % 50 == 49:
      q = dw.qpos.numpy()[0]
      np.testing.assert_allclose(q[7:13], d.qpos[7:13], atol=1e-3, err_msg=f"hand joints, step {i}")
      np.testing.assert_allclose(q[:3], d.qpos[:3], atol=1e-4, err_msg=f"wrist pos, step {i}")
      np.testing.assert_allclose(q[13:16], d.qpos[13:16], atol=1e-4, err_msg=f"object pos, step {i}")
  check_overflow(contact_stats(dw))
  assert d.xpos[oid, 2] > 0.12 and dw.xpos.numpy()[0, oid, 2] > 0.12, "object not lifted"


def test_overflow_is_detected():
  mjm = build_scene(SceneCfg()).compile()
  m, d = make_batch(mjm, 4, njmax=8)
  d.ctrl.assign(np.tile(close_ctrl(mjm, 0.7), (4, 1)).astype(np.float32))
  for _ in range(300):
    mjw.step(m, d)
  with pytest.raises(ContactOverflow):
    check_overflow(contact_stats(d))
