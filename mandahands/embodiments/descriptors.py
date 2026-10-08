"""Static joint / actuator / coupling descriptors, computed from the model (never learned per embodiment ID).

Conventions, so descriptors are comparable across hands whose source frames and sign conventions differ:
- Canonical hand frame: origin at the palm centre of mass; z from there toward the open fingertips;
  y = the direction the fingertips close toward (orthogonal to z); x = y cross z. Hands that close symmetrically
  (radial grippers) have no preferred y: it is then taken toward the first fingertip.
- Closing sign: every joint and actuator coordinate is flipped where needed so that positive = closing
  (from the probe's open and closed poses). Ranges, coupling ratios and forces follow the same convention.
  Dynamic state and actions must use the same signs (`sign` arrays), see mandahands/policy/tokenizer.py.

Coupling: C[a, j] = change of joint j per unit change of actuator a's coordinate (joint gear, tendon
coefficients, and linear joint equalities propagated through chains). Closed-loop constraints (connect / weld
equalities, e.g. the Robotiq 4-bar) are not representable as ratios: such passive joints get no entry and the
hand is flagged.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

JOINT_FIELDS = (
  [f"pos_{a}" for a in "xyz"] + [f"axis_{a}" for a in "xyz"] + ["is_slide", "lo", "hi", "range", "link_mass"]
  + [f"link_extent_{i}" for i in range(3)] + ["depth", "n_below"] + [f"tip_vec_{a}" for a in "xyz"] + ["tip_dist"]
  + [f"tip_pos_{a}" for a in "xyz"] + [f"base_pos_{a}" for a in "xyz"] + ["damping", "armature", "frictionloss", "stiffness"]
)
ACT_FIELDS = (
  ["kp", "kv", "force_close", "force_open", "ctrl_span", "gear", "velocity", "latency", "backlash", "n_driven",
   "coupling_sum", "drives_slide"] + [f"driven_pos_{a}" for a in "xyz"] + ["src_measured", "src_datasheet", "src_estimated"]
)
# normalization transform per field: "log" (positive scale, log(x + eps)), "z" (z-score), "none" (unit vectors, flags)
LOG_J = {"link_mass", "tip_dist", "damping", "armature", "frictionloss", "stiffness"} | {f"link_extent_{i}" for i in range(3)}
LOG_A = {"kp", "kv", "force_close", "force_open", "ctrl_span", "gear", "velocity", "latency", "backlash"}
NONE_J = {f"axis_{a}" for a in "xyz"} | {"is_slide"}
NONE_A = {"src_measured", "src_datasheet", "src_estimated", "drives_slide"}
LOG_EPS = 1e-4
SOURCES = ("measured", "datasheet", "estimated")
DEFAULT_VELOCITY = 5.0  # rad/s when a source gives none (flagged estimated)
MAX_VELOCITY = 50.0  # rad/s; above this a source velocity is a placeholder (e.g. URDF velocity="1000")
# unknown in every source model; until sysid (M11) real hands get these priors (generator medians)
PRIOR_LATENCY, PRIOR_BACKLASH = 0.01, 0.015
PROVENANCE_NOISE = dict(measured=0.03, datasheet=0.1, estimated=0.3)  # relative descriptor noise per provenance


@dataclass
class Descriptor:
  joint: np.ndarray  # (J, len(JOINT_FIELDS))
  act: np.ndarray  # (A, len(ACT_FIELDS))
  coupling: np.ndarray  # (A, J)
  joint_sign: np.ndarray  # (J,) +1 / -1: canonical coordinate = sign * MuJoCo coordinate
  act_sign: np.ndarray  # (A,)
  ctrl_open: np.ndarray  # (A,) MuJoCo ctrl of the open / closed hand (action -1 / +1)
  ctrl_close: np.ndarray
  frame: np.ndarray  # (3, 3) rows = canonical axes in the hand_mount frame
  joint_names: list
  act_names: list
  flags: list


def _subtree(m, root):
  out = [root]
  for b in range(root + 1, m.nbody):
    if m.body_parentid[b] in out:
      out.append(b)
  return out


def coupling_matrix(m: mujoco.MjModel) -> tuple[np.ndarray, list]:
  """C[a, j] in MuJoCo coordinates, and flags for constraints that are not representable."""
  C = np.zeros((m.nu, m.njnt))
  for a in range(m.nu):
    tid, g = m.actuator_trnid[a, 0], m.actuator_gear[a, 0]
    if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
      C[a, tid] = g
    elif int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON):
      for w in range(m.tendon_adr[tid], m.tendon_adr[tid] + m.tendon_num[tid]):
        if int(m.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_JOINT):
          C[a, m.wrap_objid[w]] += g * m.wrap_prm[w]
  flags = []
  typ = [int(t) for t in m.eq_type]  # numpy ints vs pybind enums: compare as ints
  JNT, CONNECT, WELD = int(mujoco.mjtEq.mjEQ_JOINT), int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)
  eqs = [(m.eq_obj1id[i], m.eq_obj2id[i], m.eq_data[i, 1]) for i in range(m.neq) if typ[i] == JNT]
  if any(abs(m.eq_data[i, 2:5]).max() > 1e-9 for i in range(m.neq) if typ[i] == JNT):
    flags.append("coupling_nonlinear_linearized")
  if any(t in (CONNECT, WELD) for t in typ):
    flags.append("closed_loop_not_in_coupling")
  for _ in range(m.njnt):  # follower = ratio * leader, propagated through chains
    before = C.copy()
    for f, lead, r in eqs:
      C[:, f] = r * C[:, lead]
    if np.allclose(C, before):
      break
  return C, flags


def _frame(palm, tips_open, tips_closed):
  z = tips_open.mean(0) - palm
  z /= np.linalg.norm(z)
  y = (tips_closed - tips_open).mean(0)
  y -= z * (y @ z)
  if np.linalg.norm(y) < 1e-3:  # symmetric closing (radial grippers): toward the first fingertip
    y = tips_open[0] - palm
    y -= z * (y @ z)
  y /= np.linalg.norm(y)
  return np.stack([np.cross(y, z), y, z])


def describe(spec: mujoco.MjSpec, act_meta: list[dict] | None = None, pr: dict | None = None) -> Descriptor:
  """Descriptors for a hand MjSpec rooted at 'hand_mount'.

  act_meta: per-actuator dicts (by name) with optional velocity, backlash, latency and source
  ("measured" / "datasheet" / "estimated"); missing values are filled with defaults and marked estimated.
  pr: probe() result (computed if not given).
  """
  from mandahands.sim.scripted import probe

  pr = pr or probe(spec)
  m = spec.copy().compile()
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  R = _frame(pr["palm"], pr["tips_open"], pr["tips_closed"])
  origin = np.asarray(pr["palm"])  # palm centre of mass: a geometric origin (source frames put the root anywhere)
  can = lambda v: R @ (v - origin)
  vec = lambda v: R @ v
  meta = {a["name"]: a for a in (act_meta or [])}
  flags = []

  jsign = np.array([1.0 if pr["qpos_closed"][m.joint(j).name] >= pr["qpos_open"][m.joint(j).name] else -1.0 for j in range(m.njnt)])
  asign = np.where(pr["ctrl_close"] >= pr["ctrl_open"], 1.0, -1.0)
  C0, cflags = coupling_matrix(m)
  flags += cflags
  C = asign[:, None] * C0 * jsign[None, :]

  coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
  root = m.body("hand_mount").id
  jbody = m.jnt_bodyid
  # palm: the mount plus its chain of single, jointless children (extracted hands hang their palm link below it)
  palm_end = root
  while True:
    kids = [c for c in range(m.nbody) if m.body_parentid[c] == palm_end and c != palm_end]
    if len(kids) != 1 or m.body_jntnum[kids[0]]:
      break
    palm_end = kids[0]

  def tip_of(b):  # farthest collision-geom centre in the subtree, from the body origin
    pts = [d.geom_xpos[g] for g in coll if m.geom_bodyid[g] in _subtree(m, b)]
    return max(pts, key=lambda p: np.linalg.norm(p - d.xpos[b])) if pts else d.xpos[b]

  rows = []
  for j in range(m.njnt):
    b = jbody[j]
    anchor = d.xanchor[j]
    if m.jnt_limited[j]:
      lo, hi = m.jnt_range[j]
    elif (drv := [a for a in range(m.nu) if abs(C0[a, j]) > 1e-9 and int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)]):
      lo, hi = sorted(C0[drv[0], j] * m.actuator_ctrlrange[drv[0]])  # unlimited follower: range implied by its leader
    else:
      lo, hi = -np.pi, np.pi  # unlimited: full turn
    lo, hi = (lo, hi) if jsign[j] > 0 else (-hi, -lo)
    ext = np.zeros(3)
    gs = [g for g in coll if m.geom_bodyid[g] == b]
    if gs:  # body-frame AABB of the link's collision geoms (from each geom's own AABB), half extents, sorted
      pts = []
      for g in gs:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        corners = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        pts.append(corners @ rot.reshape(3, 3).T + m.geom_pos[g])
      pts = np.concatenate(pts)
      ext = np.sort((pts.max(0) - pts.min(0)) / 2)[::-1]
    anc = []  # joints between the palm and this joint
    x = m.body_parentid[b]
    while x != root and x > 0:
      anc += [k for k in range(m.njnt) if jbody[k] == x]
      x = m.body_parentid[x]
    anc += [k for k in range(j) if jbody[k] == b]  # earlier joints on the same body
    below = [k for k in range(m.njnt) if k != j and jbody[k] in _subtree(m, b) and k not in anc]
    base = b  # finger base: the ancestor hanging off the palm
    while m.body_parentid[base] != palm_end and m.body_parentid[base] > 0:
      base = m.body_parentid[base]
    tip = tip_of(b)
    rows.append(np.concatenate([
      can(anchor), vec(d.xaxis[j]) * jsign[j], [float(int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE)), lo, hi, hi - lo],
      [m.body_mass[b]], ext, [len(anc), len(below)], vec(tip - anchor), [np.linalg.norm(tip - anchor)], can(tip),
      can(d.xpos[base]), [m.dof_damping[m.jnt_dofadr[j]], m.dof_armature[m.jnt_dofadr[j]], m.dof_frictionloss[m.jnt_dofadr[j]],
                          m.jnt_stiffness[j]],
    ]))
  joint = np.array(rows, dtype=np.float64).reshape(m.njnt, len(JOINT_FIELDS))

  rows = []
  for a in range(m.nu):
    name = m.actuator(a).name
    am = meta.get(name, {})
    kp = m.actuator_gainprm[a, 0]
    kv = -m.actuator_biasprm[a, 2]
    span = m.actuator_ctrlrange[a, 1] - m.actuator_ctrlrange[a, 0]
    flo, fhi = (m.actuator_forcerange[a] if m.actuator_forcelimited[a] else (-kp * span, kp * span))
    f_close, f_open = (fhi, -flo) if asign[a] > 0 else (-flo, fhi)
    src = am.get("source", "estimated")
    vel = am.get("velocity") or 0.0
    if not 0 < vel <= MAX_VELOCITY:
      vel = DEFAULT_VELOCITY
      src = "estimated" if src == "measured" else src
    latency = m.actuator_delay[a] if am.get("latency") is not None or m.actuator_delay[a] > 0 else PRIOR_LATENCY
    backlash = am["backlash"] if am.get("backlash") is not None else PRIOR_BACKLASH
    driven = np.abs(C[a]) > 1e-9
    rows.append(np.concatenate([
      [kp, kv, max(f_close, 0.0), max(f_open, 0.0), span, abs(m.actuator_gear[a, 0]), vel, latency, backlash,
       driven.sum(), np.abs(C[a]).sum(),
       float(driven.any() and (m.jnt_type[driven] == int(mujoco.mjtJoint.mjJNT_SLIDE)).mean() > 0.5)],
      joint[driven, :3].mean(0) if driven.any() else np.zeros(3),
      [float(src == s) for s in SOURCES],
    ]))
  act = np.array(rows, dtype=np.float64).reshape(m.nu, len(ACT_FIELDS))
  return Descriptor(joint, act, C, jsign, asign, np.asarray(pr["ctrl_open"], float), np.asarray(pr["ctrl_close"], float), R, [m.joint(j).name for j in range(m.njnt)],
                    [m.actuator(a).name for a in range(m.nu)], flags)


def _transform(x: np.ndarray, fields, log: set) -> np.ndarray:
  y = x.copy()
  for i, f in enumerate(fields):
    if f in log:
      y[..., i] = np.log(np.maximum(y[..., i], 0.0) + LOG_EPS)
  return y


class Normalizer:
  """Per-field transform (log for positive scales) then z-score with statistics of the training embodiments."""

  def __init__(self, stats: dict):
    self.stats = stats
    self.j_mu, self.j_sd = (np.array(stats["joint"][k]) for k in ("mean", "std"))
    self.a_mu, self.a_sd = (np.array(stats["act"][k]) for k in ("mean", "std"))

  @classmethod
  def fit(cls, descs: list[Descriptor]) -> "Normalizer":
    def st(x, fields, log, none):
      t = _transform(x, fields, log)
      mu, sd = t.mean(0), t.std(0)
      for i, f in enumerate(fields):
        if f in none or sd[i] < 1e-6:  # untouched: unit vectors, flags and constant fields
          mu[i], sd[i] = 0.0, 1.0
      return dict(mean=mu.tolist(), std=sd.tolist(), fields=list(fields))

    J = np.concatenate([d.joint for d in descs])
    A = np.concatenate([d.act for d in descs])
    return cls(dict(joint=st(J, JOINT_FIELDS, LOG_J, NONE_J), act=st(A, ACT_FIELDS, LOG_A, NONE_A), n_embodiments=len(descs)))

  def joint(self, x):
    return (_transform(x, JOINT_FIELDS, LOG_J) - self.j_mu) / self.j_sd

  def act(self, x):
    return (_transform(x, ACT_FIELDS, LOG_A) - self.a_mu) / self.a_sd

  def save(self, path: Path):
    Path(path).write_text(json.dumps(self.stats, indent=1))

  @classmethod
  def load(cls, path: Path) -> "Normalizer":
    return cls(json.loads(Path(path).read_text()))


NOISY_J = {"link_mass", "damping", "armature", "frictionloss", "stiffness", "tip_dist", "lo", "hi", "range"} | {f"link_extent_{i}" for i in range(3)}
POS_J = ("pos", "tip_vec", "tip_pos", "base_pos")
NOISY_A = {"kp", "kv", "force_close", "force_open", "velocity", "latency", "backlash"}


def noisy(desc: Descriptor, rng: np.random.Generator, scale: float = 0.1) -> Descriptor:
  """Descriptor noise for training (spec: +-5-10% on geometric and actuator fields): multiplicative U(-s, s) on
  joint scale fields and +-s x |v| on position-like 3-vectors; actuator fields get PROVENANCE_NOISE by their
  provenance flag. Coupling, axes, flags and counts are left exact."""
  J, A = desc.joint.copy(), desc.act.copy()
  for i, f in enumerate(JOINT_FIELDS):
    if f in NOISY_J:
      J[:, i] *= 1 + rng.uniform(-scale, scale, len(J))
  for p in POS_J:
    i = JOINT_FIELDS.index(f"{p}_x")
    v = J[:, i : i + 3]
    J[:, i : i + 3] = v + rng.uniform(-scale, scale, v.shape) * np.linalg.norm(v, axis=1, keepdims=True)
  # actuator noise width by provenance: measured values are tight, estimated ones wide
  width = A[:, [ACT_FIELDS.index(f"src_{s}") for s in SOURCES]] @ np.array([PROVENANCE_NOISE[s] for s in SOURCES])
  for i, f in enumerate(ACT_FIELDS):
    if f in NOISY_A:
      A[:, i] *= 1 + rng.uniform(-1, 1, len(A)) * width
  return Descriptor(J, A, desc.coupling, desc.joint_sign, desc.act_sign, desc.ctrl_open, desc.ctrl_close, desc.frame,
                    desc.joint_names, desc.act_names, desc.flags)


class SupportBank:
  """Out-of-support check: nearest-neighbour distance of a hand's normalized tokens to the training tokens.

  A hand's score is the mean over its joint (actuator) tokens of the distance to the nearest training token of
  another hand. The reference distribution is the same score for every training hand, leave-one-hand-out.
  """

  def __init__(self, descs: list[Descriptor], norm: Normalizer):
    self.norm = norm
    self.J = np.concatenate([norm.joint(d.joint) for d in descs])
    self.A = np.concatenate([norm.act(d.act) for d in descs])
    self.jh = np.concatenate([np.full(len(d.joint), i) for i, d in enumerate(descs)])
    self.ah = np.concatenate([np.full(len(d.act), i) for i, d in enumerate(descs)])
    self.base = np.array([self._score(i) for i in range(len(descs))])  # (n_train, 2): joint, act

  @staticmethod
  def _nn(x, bank):
    d2 = (x * x).sum(1)[:, None] + (bank * bank).sum(1)[None] - 2 * x @ bank.T
    return np.sqrt(np.maximum(d2.min(1), 0))

  def _score(self, i):
    return (self._nn(self.J[self.jh == i], self.J[self.jh != i]).mean(), self._nn(self.A[self.ah == i], self.A[self.ah != i]).mean())

  def score(self, d: Descriptor) -> dict:
    j = self._nn(self.norm.joint(d.joint), self.J).mean()
    a = self._nn(self.norm.act(d.act), self.A).mean()
    pj, pa = (self.base[:, 0] < j).mean() * 100, (self.base[:, 1] < a).mean() * 100
    p99 = np.percentile(self.base, 99, 0)
    return dict(joint_nn=float(j), act_nn=float(a), joint_pct=float(pj), act_pct=float(pa),
                out_of_support=bool(j > p99[0] or a > p99[1]))
