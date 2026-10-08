"""Scripted close-and-lift test for any hand MjSpec rooted at body 'hand_mount'.

probe():  per-actuator closing direction (the ctrl limit that brings fingertips together, found by simulation
          so coupled joints behave), open/closed fingertip sets, and candidate object centres with their
          clearance to the open hand (largest sphere that fits there).
close_and_lift(): palm above the object; object held by a support weld; close, release support, lift 10 cm, hold.
grasp_search(): candidate centres x primitive types; pass if any lifts the object.
"""

import itertools

import mujoco
import numpy as np

from mandahands.sim.builder import SceneCfg, build_scene

LIFT = 0.10
T_SETTLE, T_CLOSE, T_RELEASE, T_LIFT, T_HOLD = 0.2, 0.6, 0.2, 1.0, 0.5
MIN_CLEARANCE = 0.005
SQUEEZE = 0.15  # fraction of an actuator's range it keeps closing after first touching the object


def _subtree(m, root):
  out = [root]
  for b in range(root + 1, m.nbody):
    if m.body_parentid[b] in out:
      out.append(b)
  return out


def _tip_points(m, d, leaves, geoms):
  """Per leaf body, the collision geom centre farthest from its parent body (approximate fingertip)."""
  pts = []
  for b in leaves:
    par = d.xpos[m.body_parentid[b]]
    pts.append(max((d.geom_xpos[g] for g in geoms if m.geom_bodyid[g] == b), key=lambda p: np.linalg.norm(p - par)))
  return np.array(pts)


def _settle(m, d, ctrl, t=0.5):
  d.ctrl[:] = ctrl
  for _ in range(int(t / m.opt.timestep)):
    mujoco.mj_step(m, d)


def probe(hand: mujoco.MjSpec, close_override: dict[str, str] | None = None) -> dict:
  """close_override: actuator name -> "lo" | "hi" for hands whose convention the heuristics get wrong."""
  mh = hand.copy().compile()
  hand_geoms = [g for g in range(mh.ngeom) if mh.geom_contype[g] or mh.geom_conaffinity[g]]

  spec = hand.copy()
  ball = spec.worldbody.add_body(name="_probe", mocap=True)  # added last: hand geom ids are unchanged
  ball.add_geom(name="_probe_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-4, 0, 0])
  m = spec.compile()
  m.opt.gravity[:] = 0
  m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  m.geom_contype[:], m.geom_conaffinity[:] = 0, 0  # no self-collision, as in the scene
  d = mujoco.MjData(m)

  root = m.body("hand_mount").id
  sub = _subtree(m, root)
  # extracted hands hang their palm link under an empty hand_mount; the palm is that link
  kids = [b for b in sub if m.body_parentid[b] == root]
  palm_body = root if m.body_geomnum[root] or len(kids) != 1 else kids[0]
  has_child = {m.body_parentid[b] for b in sub if b != root}
  geom_bodies = {m.geom_bodyid[g] for g in hand_geoms}
  leaves = [b for b in sub if b not in has_child and b in geom_bodies and b != root]

  lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
  base = np.clip(np.zeros(m.nu), lo, hi)

  # closing = fingertips approaching the rest of the hand (palm, other fingers, opposing jaw):
  # per tip, distance to the nearest hand geom not on its own finger chain
  fromto = np.zeros(6)
  targets = []
  for b in leaves:
    chain, x = set(), b
    while x != palm_body and x != root:
      chain.add(x)
      x = m.body_parentid[x]
    tip_gs = [g for g in hand_geoms if m.geom_bodyid[g] == b]
    targets.append((tip_gs, [g for g in hand_geoms if m.geom_bodyid[g] not in chain]))

  def spread(ctrl):
    mujoco.mj_resetData(m, d)
    _settle(m, d, ctrl)
    return sum(min(mujoco.mj_geomDistance(m, d, a, o, 1.0, fromto) for a in tg for o in og) for tg, og in targets)

  # closing direction, in order: manual override; asymmetric range -> the end farther from zero (rest is
  # straight/open for flexion joints and grippers); symmetric range (abduction, thumb rotation) -> geometry
  close, ambiguous, rule = hi.copy(), [], {}
  for i in range(m.nu):
    name = m.actuator(i).name
    if close_override and name in close_override:
      close[i] = lo[i] if close_override[name] == "lo" else hi[i]
      rule[name] = "override"
      continue
    a, b = abs(lo[i]), abs(hi[i])
    if max(a, b) > 1.5 * min(a, b):
      close[i] = lo[i] if a > b else hi[i]
      rule[name] = "range"
      continue
    rule[name] = "geometry"
    c_lo, c_hi = base.copy(), base.copy()
    c_lo[i], c_hi[i] = lo[i], hi[i]
    s_lo, s_hi = spread(c_lo), spread(c_hi)
    if abs(s_lo - s_hi) < 1e-4:
      ambiguous.append(m.actuator(i).name)
    close[i] = lo[i] if s_lo < s_hi else hi[i]
  open_ = np.where(close == hi, lo, hi)

  pg = m.geom("_probe_g").id

  def clearance(c):
    d.mocap_pos[0] = c
    mujoco.mj_kinematics(m, d)
    return min(mujoco.mj_geomDistance(m, d, pg, g, 1.0, fromto) for g in hand_geoms)

  mujoco.mj_resetData(m, d)
  _settle(m, d, close)
  tips_closed = _tip_points(m, d, leaves, hand_geoms)
  qpos_closed = d.qpos.copy()
  mujoco.mj_resetData(m, d)
  _settle(m, d, open_)
  qpos_open_vec = d.qpos.copy()
  tips_open = _tip_points(m, d, leaves, hand_geoms)
  palm = d.xipos[palm_body].copy()
  qpos_open = {m.joint(j).name: float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)}
  qpos_closed_named = {m.joint(j).name: float(qpos_closed[m.jnt_qposadr[j]]) for j in range(m.njnt)}

  # candidate centres: the plane spanned by palm->open tips and palm->closed tips, plus a grid over the hand's
  # extent (covers symmetric grippers, where that plane collapses to a line). Keep enclosed points: clear of the
  # open hand, but the closed hand reaches well into that clearance. Rank by intrusion depth (open - closed
  # clearance: how firmly the hand closes on an object there), keeping well-separated points.
  go, gc = tips_open.mean(0) - palm, tips_closed.mean(0) - palm
  plane = [palm + a * go + b * gc for a, b in itertools.product(np.linspace(0, 1.2, 7), repeat=2) if a + b > 0]
  ext = np.vstack([tips_open, tips_closed, palm[None]])
  lo_b, hi_b = ext.min(0) - 0.02, ext.max(0) + 0.02
  grid = np.stack(np.meshgrid(*[np.linspace(a, b, 9) for a, b in zip(lo_b, hi_b)], indexing="ij"), -1).reshape(-1, 3)
  pts = np.vstack([plane, grid])
  cl_open = np.array([clearance(c) for c in pts])
  d.qpos[:] = qpos_closed
  cl_closed = np.array([clearance(c) for c in pts])
  keep = (cl_open > MIN_CLEARANCE) & (cl_closed < 0.5 * cl_open)

  def pick(order, n=6):
    out = []
    for i in order:
      if i < 0:
        continue
      if keep[i] and all(np.linalg.norm(pts[i] - c) > 0.5 * cl for c, cl in out):
        out.append((pts[i], float(cl_open[i])))
      if len(out) == n:
        break
    return out

  # three candidate lists (none wins on every hand): intrusion depth, largest enclosed object, and largest
  # enclosed object restricted to the palm/fingertip plane
  n_plane = len(plane)
  # pinch candidates first: where the closed fingertips converge, and on the way back to the palm (power grasp)
  conv = tips_closed.mean(0)
  pinch = [conv + f * (palm - conv) for f in (0.0, 0.25, 0.5, 0.75)]
  d.qpos[:] = qpos_open_vec
  cands = [(p, cl) for p in pinch if (cl := clearance(p)) > MIN_CLEARANCE]
  n_pinch = len(cands)
  for order in (np.argsort(-(cl_open - cl_closed)), np.argsort(-cl_open), np.argsort(-cl_open[:n_plane])):
    cands += [c for c in pick(order) if not any(np.allclose(c[0], x[0]) for x in cands)]
  return dict(n_pinch=n_pinch,
    ctrl_open=open_, ctrl_close=close, close_rule=rule, qpos_open=qpos_open, qpos_closed=qpos_closed_named, ambiguous=ambiguous, n_tips=len(leaves), palm=palm,
    tips_open=tips_open, tips_closed=tips_closed, candidates=cands,
  )


def _quat_to(a, b):
  """Unit quaternion rotating unit vector a onto b."""
  v, c = np.cross(a, b), float(np.dot(a, b))
  if c < -0.999999:
    perp = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
    return np.r_[0, perp / np.linalg.norm(perp)]
  q = np.r_[1 + c, v]
  return q / np.linalg.norm(q)


# object half-extent as a fraction of clearance so the primitive fits inside the clearance sphere
_FIT = {"sphere": 0.9, "cylinder": 0.9 / np.sqrt(2), "box": 0.9 / np.sqrt(3)}


def _driven_bodies(m: mujoco.MjModel) -> np.ndarray:
  """(nu, nbody) mask: bodies moved by each actuator (subtrees of its joints, or of its tendon's joints)."""
  out = np.zeros((m.nu, m.nbody), bool)
  for i in range(m.nu):
    tid = m.actuator_trnid[i, 0]
    if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
      joints = [tid]
    elif m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
      adr, num = m.tendon_adr[tid], m.tendon_num[tid]
      joints = [m.wrap_objid[w] for w in range(adr, adr + num) if m.wrap_type[w] == mujoco.mjtWrap.mjWRAP_JOINT]
    else:
      continue
    for j in joints:
      for b in _subtree(m, m.jnt_bodyid[j]):
        out[i, b] = True
  return out


def place(hand: mujoco.MjSpec, pr: dict, centre, obj: str, size: float, cfg: SceneCfg | None = None, spec_fn=build_scene):
  """Scene with an object of half-extent `size` at `centre` (hand-mount frame), the palm straight above it.
  Returns (scene spec, wrist_pos, wrist_quat, obj_pos)."""
  cfg = SceneCfg(**{**(cfg or SceneCfg()).__dict__, "obj": obj, "obj_half": size})
  ax = (centre - pr["palm"]) / np.linalg.norm(centre - pr["palm"])
  quat = _quat_to(ax, np.array([0.0, 0.0, -1.0]))
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, quat)
  obj_pos = np.array([0, 0, 0.5])
  wrist_pos = obj_pos - rot.reshape(3, 3) @ centre
  return spec_fn(cfg, hand, wrist_pos=wrist_pos, wrist_quat=quat, obj_pos=obj_pos), wrist_pos, quat, obj_pos


def start_state(m: mujoco.MjModel, pr: dict) -> mujoco.MjData:
  """Hand at its open pose (stepping the command from rest to open flings light fingers: LEAP ~200 rad/s), object
  held by its support weld."""
  d = mujoco.MjData(m)
  for name, q in pr["qpos_open"].items():
    d.qpos[m.jnt_qposadr[m.joint(name).id]] = q
  d.eq_active[m.equality("object_support").id] = 1
  mujoco.mj_forward(m, d)
  return d


def close_and_lift(hand: mujoco.MjSpec, pr: dict, centre, clearance: float, obj: str, cfg: SceneCfg | None = None,
                   on_phase=None, size: float | None = None, squeeze: float | None = SQUEEZE) -> dict:
  """Object of type obj fitted at `centre` (mount frame); the palm is placed straight above it.

  size: object half-extent; default fits the object inside the clearance sphere.
  squeeze: see SQUEEZE; None closes every actuator fully (the M2 behaviour).

  on_phase(name, m, d) is called after each phase (open/closed/released/lifted/held), e.g. to render.
  """
  on_phase = on_phase or (lambda name, m, d: None)
  size = size or _FIT[obj] * clearance
  spec, wrist_pos, quat, obj_pos = place(hand, pr, centre, obj, size, cfg)
  m = spec.compile()
  d = start_state(m, pr)
  oid, wid = m.body("object").id, m.body("wrist").id
  dt = m.opt.timestep
  z0 = d.mocap_pos[0, 2]

  def run(t, ctrl_fn=None, lift_fn=None):
    for k in range(int(t / dt)):
      if ctrl_fn is not None:
        d.ctrl[:] = ctrl_fn(k * dt / t)
      if lift_fn is not None:
        d.mocap_pos[0, 2] = z0 + lift_fn(k * dt / t)
      mujoco.mj_step(m, d)

  run(T_SETTLE, lambda s: pr["ctrl_open"])
  on_phase("open", m, d)
  pen_open = min([c.dist for c in d.contact[: d.ncon]], default=0.0)
  # close: each actuator ramps open -> closed until a link it drives touches the object, then squeezes
  # SQUEEZE more of its range and stops (full closure at full torque squirts small objects out)
  drives = _driven_bodies(m)
  frac = np.full(m.nu, np.inf)  # per-actuator stop fraction, set at first touch
  n_close = int(T_CLOSE / dt)
  for k in range(n_close):
    ramp = min(k / (0.7 * n_close), 1.0)
    if squeeze is not None:
      touching = np.zeros(m.nbody, bool)
      for c in d.contact[: d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if oid in (b1, b2):
          touching[b1 + b2 - oid] = True
      hit = (drives & touching).any(1) & np.isinf(frac)
      frac[hit] = ramp + squeeze
    d.ctrl[:] = pr["ctrl_open"] + np.minimum(np.minimum(ramp, frac), 1.0) * (pr["ctrl_close"] - pr["ctrl_open"])
    mujoco.mj_step(m, d)
  on_phase("closed", m, d)
  n_contact = sum(1 for c in d.contact[: d.ncon] if oid in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]))
  d.eq_active[m.equality("object_support").id] = 0
  run(T_RELEASE)
  on_phase("released", m, d)
  rel0 = d.xpos[oid] - d.xpos[wid]
  run(T_LIFT, lift_fn=lambda s: LIFT * s)
  on_phase("lifted", m, d)
  run(T_HOLD, lift_fn=lambda s: LIFT)
  on_phase("held", m, d)
  drift = float(np.linalg.norm((d.xpos[oid] - d.xpos[wid]) - rel0))
  dz = float(d.xpos[oid, 2] - obj_pos[2])
  # pen_open: the open hand must not start inside the object, or "holding" is just interpenetration
  # BADQACC makes MuJoCo silently reset mjData (object teleports back): an unstable trial never passes
  unstable = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
  ok = bool(abs(dz - LIFT) < 0.03 and drift < 0.02 and pen_open > -0.005 and not unstable)
  return dict(obj=obj, size=size, centre=np.asarray(centre).tolist(), ok=ok, dz=dz, drift=drift,
              pen_open=float(pen_open), n_contact=n_contact, unstable=bool(unstable), wrist_pos=wrist_pos.tolist(), wrist_quat=quat.tolist())


def grasp_search(hand: mujoco.MjSpec, pr: dict | None = None, n_centres: int = 18, close_override=None,
                 cfg: SceneCfg | None = None) -> tuple[dict, list[dict]]:
  pr = pr or probe(hand, close_override)
  results = []
  for (c, cl), obj in itertools.product(pr["candidates"][:n_centres], ("sphere", "cylinder", "box")):
    results.append(close_and_lift(hand, pr, c, cl, obj, cfg))
  return pr, results
