"""Validity filter for generated hands, and a hand-level summary vector used for coverage plots.

A hand is valid when its rest and open poses are free of self-penetration and the scripted close-and-lift
(sim/scripted.py) lifts at least K_MIN of the reference objects (sphere, cylinder, box at two absolute sizes).
"""

import itertools

import mujoco
import numpy as np

from mandahands.sim.builder import add_self_collision
from mandahands.sim.scripted import SQUEEZE, close_and_lift, probe

REF_SIZES = (0.012, 0.025)  # m, object half-extent: 2.4 cm and 5 cm objects
REF_OBJECTS = [(o, s) for s in REF_SIZES for o in ("sphere", "cylinder", "box")]
K_MIN = 2  # calibrated on the real hands: every one lifts at least 2 (see docs/m3_report.md)
TRIES = 4  # candidate centres per reference object
PEN_TOL = 1e-3  # m


def self_penetration(m: mujoco.MjModel, qpos: np.ndarray | None = None) -> list[tuple[str, str, float]]:
  """Hand collision geom pairs that overlap by more than PEN_TOL, excluding same-body and parent-child pairs."""
  d = mujoco.MjData(m)
  if qpos is not None:
    d.qpos[:] = qpos
  mujoco.mj_kinematics(m, d)
  gs = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
  fromto, out = np.zeros(6), []
  for a, b in itertools.combinations(gs, 2):
    ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
    if ba == bb or m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:
      continue
    dist = mujoco.mj_geomDistance(m, d, a, b, 0.0, fromto)
    if dist < -PEN_TOL:
      out.append((m.geom(a).name, m.geom(b).name, float(dist)))
  return out


FEATURES = ["n_fingers", "n_joints", "n_actuators", "act_per_joint", "frac_slide", "log_span", "log_reach",
            "log_closure", "log_mass", "hinge_range", "log_force", "log_aperture"]


def summary(m: mujoco.MjModel, pr: dict) -> dict:
  """Hand-level morphology/actuation summary from the compiled hand and its probe (same for real and generated).

  ponytail: hand-level only; M4's per-joint descriptors replace this for token-level coverage.
  """
  to, tc = pr["tips_open"], pr["tips_closed"]
  span = max((np.linalg.norm(a - b) for a, b in itertools.combinations(to, 2)), default=0.0)
  hinge = m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
  ctrl_span = m.actuator_ctrlrange[:, 1] - m.actuator_ctrlrange[:, 0]
  force = m.actuator_gainprm[:, 0] * ctrl_span
  lim = m.actuator_forcelimited.astype(bool)
  force[lim] = np.minimum(force[lim], m.actuator_forcerange[lim, 1])
  return dict(
    n_fingers=int(pr["n_tips"]),
    n_joints=int(m.njnt),
    n_actuators=int(m.nu),
    act_per_joint=m.nu / max(m.njnt, 1),
    frac_slide=float((m.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE).mean()),
    log_span=float(np.log(span + 1e-3)),
    log_reach=float(np.log(np.linalg.norm(to - pr["palm"], axis=1).mean())),
    log_closure=float(np.log(np.linalg.norm(to - tc, axis=1).mean() + 1e-3)),
    log_mass=float(np.log(m.body_mass.sum())),
    hinge_range=float((m.jnt_range[hinge, 1] - m.jnt_range[hinge, 0]).mean()) if hinge.any() else 0.0,
    log_force=float(np.log(np.median(force))),
    log_aperture=float(np.log(max((cl for _, cl in pr["candidates"]), default=0.0) + 1e-3)),
  )


def ref_grasps(spec: mujoco.MjSpec, pr: dict) -> list[dict]:
  """Scripted close-and-lift of each reference object; up to TRIES candidate centres x two closing strategies."""
  ref = []
  for obj, size in REF_OBJECTS:
    # pinch candidates (probe order) first, then the tightest-fitting others
    pinch = [(cl, tuple(c)) for c, cl in pr["candidates"][: pr["n_pinch"]] if cl >= size]
    rest = sorted((cl, tuple(c)) for c, cl in pr["candidates"][pr["n_pinch"] :] if cl >= size)
    cands = (pinch + rest)[:TRIES]
    best = None
    # two scripted closings: stop-on-contact + squeeze, then full closure (neither dominates on real hands)
    for (cl, c), squeeze in itertools.product(cands, (SQUEEZE, None)):
      r = close_and_lift(spec, pr, np.array(c), cl, obj, size=size, squeeze=squeeze)
      if r["ok"]:
        best = r
        break
    ref.append(dict(obj=obj, size=size, ok=best is not None, tried=len(cands),
                    trial=None if best is None else dict(centre=best["centre"], drift=best["drift"], dz=best["dz"], squeeze=squeeze)))
  return ref


def check(spec: mujoco.MjSpec) -> dict:
  """Run the filter on a hand spec (modified in place: self-collision pairs are added)."""
  m = spec.copy().compile()
  pen = self_penetration(m)
  if pen:
    return dict(valid=False, reason="self_penetration_rest", detail=pen[:3])
  sc = add_self_collision(spec)
  pr = probe(spec)
  m = spec.copy().compile()
  qo = np.array([pr["qpos_open"][m.joint(j).name] for j in range(m.njnt)])
  pen = self_penetration(m, qo)
  if pen:
    return dict(valid=False, reason="self_penetration_open", detail=pen[:3], self_collision=sc)

  ref = ref_grasps(spec, pr)
  n_ok = sum(r["ok"] for r in ref)
  return dict(valid=n_ok >= K_MIN, reason=None if n_ok >= K_MIN else "grasp", n_ref=n_ok, ref=ref,
              self_collision=sc, close_rule=pr["close_rule"], summary=summary(m, pr))
