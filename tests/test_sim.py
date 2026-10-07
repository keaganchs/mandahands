import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest

from mandahands.sim.batch import ContactOverflow, check_overflow, contact_stats, make_batch
from mandahands.sim.builder import SceneCfg, build_scene, close_ctrl, simple_hand


def _script(mjm, i, mz):
  """Close over 0.5 s, then lift the wrist 10 cm over 1 s starting at 0.7 s."""
  t = i * mjm.opt.timestep
  ctrl = close_ctrl(mjm, min(t / 0.5, 1) * 0.7)
  mocap_z = mz + (min((t - 0.7) / 1.0, 1) * 0.1 if t > 0.7 else 0.0)
  return ctrl, mocap_z


def _rollout_both(**scene_kw):
  mjm = build_scene(SceneCfg(), **scene_kw).compile()
  d = mujoco.MjData(mjm)
  m, dw = make_batch(mjm, 1, njmax=160)
  mz = d.mocap_pos[0, 2]
  err = 0.0
  for i in range(1000):
    d.ctrl[:], d.mocap_pos[0, 2] = _script(mjm, i, mz)
    dw.ctrl.assign(d.ctrl[None].astype(np.float32))
    dw.mocap_pos.assign(d.mocap_pos[None].astype(np.float32))
    mujoco.mj_step(mjm, d)
    mjw.step(m, dw)
    err = max(err, np.abs(dw.qpos.numpy()[0, :13] - d.qpos[:13]).max())  # wrist + hand
  check_overflow(contact_stats(dw))
  oid = mjm.body("object").id
  return err, d.xpos[oid, 2], dw.xpos.numpy()[0, oid, 2]


def test_warp_matches_cpu_free_space():
  # no contact at all (object out of reach, fingertip self-collision pairs removed):
  # trajectories must match tightly (observed ~7e-7 rad over 2 s)
  hand = simple_hand(SceneCfg())
  for p in list(hand.pairs):
    hand.delete(p)
  err, _, _ = _rollout_both(hand=hand, wrist_pos=[0, 0, 0.105], obj_pos=[0.5, 0.5, 0.03])
  assert err < 1e-5, err


def test_warp_matches_cpu_close_and_lift_outcome():
  # contact onset is sensitive to float32 vs float64, so require the same outcome rather than the same
  # trajectory. Observed: both 0.1129 m (object starts at 0.03, wrist lifts 0.10, ~2 cm slip at impratio 3)
  _, z_cpu, z_warp = _rollout_both()
  assert z_cpu > 0.10 and z_warp > 0.10, (z_cpu, z_warp)
  assert abs(z_cpu - z_warp) < 0.015, (z_cpu, z_warp)


def test_overflow_is_detected():
  mjm = build_scene(SceneCfg()).compile()
  m, d = make_batch(mjm, 4, njmax=8)
  d.ctrl.assign(np.tile(close_ctrl(mjm, 0.7), (4, 1)).astype(np.float32))
  for _ in range(300):
    mjw.step(m, d)
  with pytest.raises(ContactOverflow):
    check_overflow(contact_stats(d))
