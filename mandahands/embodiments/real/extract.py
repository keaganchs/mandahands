"""Cut a hand subtree out of a robot model into a standalone, simulatable hand MjSpec plus metadata."""

import subprocess
from pathlib import Path

import mujoco
import numpy as np

from mandahands.embodiments.real.load import find_packages, load
from mandahands.embodiments.real.sources import EXTRA_PACKAGES, REPOS, HandSource
from mandahands.sim.builder import add_self_collision

SOURCES = Path("data/sources")
COACD_CACHE = Path("data/cache/coacd")
OUT = Path("data/real_hands")

DEFAULT_EFFORT = {mujoco.mjtJoint.mjJNT_HINGE: 1.0, mujoco.mjtJoint.mjJNT_SLIDE: 20.0}  # Nm / N, when the source has none
MAX_EFFORT = {mujoco.mjtJoint.mjJNT_HINGE: 20.0, mujoco.mjtJoint.mjJNT_SLIDE: 200.0}  # above this: a placeholder
MAX_GYRATION = 0.1  # m; a hand link with a larger radius of gyration has placeholder inertia
MIN_SPEED = 3.0  # rad/s; a finger joint that can't reach this at full effort has placeholder damping
MIN_ARMATURE = 1e-4  # URDFs carry no rotor inertia; tiny finger links are unstable at 2 ms without some


def _packages() -> dict[str, Path]:
  pk = find_packages(SOURCES)
  pk.update({k: SOURCES / v for k, v in EXTRA_PACKAGES.items()})
  return pk


def _subtree(body) -> list:
  out, stack = [], [body]
  while stack:
    b = stack.pop()
    out.append(b)
    stack.extend(b.bodies)
  return out


def _decompose(spec: mujoco.MjSpec, bodies) -> int:
  """Replace each collision mesh geom with its CoACD convex parts (cached). Returns the number of geoms split."""
  import hashlib

  import coacd
  import trimesh

  coacd.set_log_level("error")
  n = 0
  for b in bodies:
    for g in [g for g in b.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH and (g.contype or g.conaffinity)]:
      mesh = spec.mesh(g.meshname)
      raw = Path(mesh.file).read_bytes()
      key = hashlib.sha1(raw + str(list(mesh.scale)).encode()).hexdigest()[:16]
      cache = COACD_CACHE / f"{key}.npz"
      if cache.exists():
        z = np.load(cache)
        parts = [(z[f"v{i}"], z[f"f{i}"]) for i in range(int(z["n"]))]
      else:
        t = trimesh.load(mesh.file, force="mesh")
        t.apply_scale(np.asarray(mesh.scale))
        parts = coacd.run_coacd(coacd.Mesh(t.vertices, t.faces), threshold=0.05, max_convex_hull=8)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, n=len(parts), **{f"v{i}": v for i, (v, _) in enumerate(parts)}, **{f"f{i}": f for i, (_, f) in enumerate(parts)})
      if len(parts) < 2:
        continue
      for i, (v, f) in enumerate(parts):
        pm = spec.add_mesh(name=f"{g.meshname}_cd{i}", uservert=np.asarray(v).ravel().tolist(), userface=np.asarray(f).ravel().tolist())
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=pm.name, pos=g.pos, quat=g.quat, friction=g.friction,
                   contype=g.contype, conaffinity=g.conaffinity, mass=0)
      g.contype = g.conaffinity = 0  # keep the original for mass/inertia and visuals, drop it from collision
      n += 1
  return n


def _servo(a, j, effort: float):
  """Position servo over the joint's range with full effort at a quarter-range error. Gains are guesses; flagged."""
  rng = list(j.range)
  kp = effort / max(0.25 * (rng[1] - rng[0]), 1e-3)
  a.set_to_position(kp=kp, kv=0.05 * kp)
  a.gear = [1, 0, 0, 0, 0, 0]
  a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
  a.ctrlrange = rng
  a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
  a.forcerange = [-effort, effort]


ROM_STEP = 0.05  # rad
ROM_TOL = 0.002  # m of new penetration that counts as the hand clearly intersecting itself


def _resolve_ranges(spec: mujoco.MjSpec, joints) -> dict[str, list[float]]:
  """Hinge joints with no usable range (unlimited, or wider than a full turn, e.g. R2's +-4 rad) get their full
  range of motion: +-pi, cut only where sweeping the joint (others at 0) makes the links it moves penetrate another
  part of the hand by more than ROM_TOL beyond what they already do at 0. Returns joint -> new range."""
  todo = [j for j in joints if j.type == mujoco.mjtJoint.mjJNT_HINGE
          and not (j.range[0] < j.range[1] and j.range[1] - j.range[0] <= 2 * np.pi)]
  if not todo:
    return {}
  probe = spec.copy()
  ball = probe.worldbody.add_body(name="_rom_probe", mocap=True)  # added last: hand ids unchanged
  ball.add_geom(name="_rom_probe_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-4, 0, 0], contype=0, conaffinity=0)
  m = probe.compile()
  d = mujoco.MjData(m)
  pg = m.geom("_rom_probe_g").id
  coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
  fromto = np.zeros(6)
  out = {}
  for j in todo:
    jid = m.joint(j.name).id
    moving = {jid_b for jid_b in range(m.nbody) if _is_desc(m, jid_b, m.jnt_bodyid[jid])}
    pairs = [(a, b) for a in coll if m.geom_bodyid[a] in moving for b in coll if m.geom_bodyid[b] not in moving]
    still = [b for b in coll if m.geom_bodyid[b] not in moving]
    mov = [a for a in coll if m.geom_bodyid[a] in moving]

    def dist():
      # surface penetration between moving and still geoms (saturates at a shared joint, where neighbouring
      # links already overlap), plus points of the moving links (geom centres, capsule/cylinder ends) entering
      # still geoms (catches a link folding back through its parent)
      dd = [mujoco.mj_geomDistance(m, d, a, b, 0.05, fromto) for a, b in pairs]
      for a in mov:
        pts = [d.geom_xpos[a]]
        if m.geom_type[a] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
          ax = d.geom_xmat[a].reshape(3, 3)[:, 2] * m.geom_size[a, 1]
          pts += [d.geom_xpos[a] + ax, d.geom_xpos[a] - ax]
        for p in pts:
          d.mocap_pos[0] = p
          mujoco.mj_kinematics(m, d)
          dd += [mujoco.mj_geomDistance(m, d, pg, b, 0.05, fromto) for b in still]
      return np.array(dd)

    mujoco.mj_resetData(m, d)
    mujoco.mj_kinematics(m, d)
    d0 = np.minimum(dist(), 0.0)
    lim = []
    for sign in (-1, 1):
      q = 0.0
      while q + ROM_STEP <= np.pi:
        d.qpos[m.jnt_qposadr[jid]] = sign * (q + ROM_STEP)
        mujoco.mj_kinematics(m, d)
        if (dist() < d0 - ROM_TOL).any():
          break
        q += ROM_STEP
      lim.append(sign * q if q + ROM_STEP <= np.pi else sign * np.pi)
    out[j.name] = lim
    j.range = lim
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
  return out


def _is_desc(m, b, root) -> bool:
  while b > 0:
    if b == root:
      return True
    b = m.body_parentid[b]
  return b == root


def _box_inertial(m: mujoco.MjModel, i: int, b) -> None:
  """Solid-box estimate from the body's collision-geom AABB, density 1000 kg/m^3 at 50% fill.

  ponytail: crude, but right order of magnitude; good enough to replace placeholder inertials. Upgrade to
  per-geom volume integration if these links matter for sysid.
  """
  gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == i and (m.geom_contype[g] or m.geom_conaffinity[g])]
  if not gs:  # virtual link: nominal 1 g point-ish mass
    b.mass, b.ipos, b.iquat, b.inertia, b.fullinertia = 1e-3, [0, 0, 0], [1, 0, 0, 0], [1e-7] * 3, [np.nan] * 6
    return
  lo = np.min([m.geom_pos[g] + m.geom_aabb[g, :3] - m.geom_aabb[g, 3:] for g in gs], axis=0)
  hi = np.max([m.geom_pos[g] + m.geom_aabb[g, :3] + m.geom_aabb[g, 3:] for g in gs], axis=0)
  ext = np.maximum(hi - lo, 1e-3)
  mass = 1000 * 0.5 * np.prod(ext)
  b.mass, b.ipos, b.iquat = mass, (lo + hi) / 2, [1, 0, 0, 0]
  b.inertia = mass / 12 * np.array([ext[1] ** 2 + ext[2] ** 2, ext[0] ** 2 + ext[2] ** 2, ext[0] ** 2 + ext[1] ** 2])
  b.fullinertia = [np.nan] * 6


def extract(hs: HandSource) -> tuple[mujoco.MjSpec, dict]:
  src, notes, urdf_vel = load(SOURCES / hs.file, _packages())
  notes = list(hs.notes) + notes
  flags: list[str] = []
  if src.body(hs.hand_root) is None:
    raise KeyError(f"{hs.id}: hand_root {hs.hand_root!r} not found")

  spec = mujoco.MjSpec()
  spec.modelname = hs.id
  spec.compiler.autolimits = True
  spec.compiler.degree = False  # MjSpec defaults to degrees
  mount = spec.worldbody.add_body(name="hand_mount")
  root = mount.add_frame().attach_body(src.body(hs.hand_root), "", "")

  # the hand root's own joints connect it to the arm: remove them (and anything referring to them)
  arm_joints = {j.name for j in root.joints}
  for j in list(root.joints):
    spec.delete(j)
  for a in list(spec.actuators):
    if a.target in arm_joints:
      spec.delete(a)
  for e in list(spec.equalities):
    if e.name1 in arm_joints or e.name2 in arm_joints:
      spec.delete(e)

  bodies = _subtree(root)
  joints = [j for b in bodies for j in b.joints]
  if not joints:
    raise ValueError(f"{hs.id}: no joints below {hs.hand_root}")

  # coupling: joint equalities (URDF mimic) make obj1 follow obj2
  follower = {e.name1: (e.name2, list(e.data[:5])) for e in spec.equalities
              if e.type == mujoco.mjtEq.mjEQ_JOINT and e.name2}
  driven = {a.target for a in spec.actuators if a.trntype == mujoco.mjtTrn.mjTRN_JOINT}
  tendon_driven = [a.name for a in spec.actuators if a.trntype == mujoco.mjtTrn.mjTRN_TENDON]

  # collision: bodies with no collidable geom get a capsule along their link segment
  approx = []
  for b in bodies:
    if any(g.contype or g.conaffinity for g in b.geoms):
      continue
    kids = list(b.bodies)
    if b is root:  # palm: a fan of capsules to each finger base
      ends = [np.asarray(k.pos) for k in kids]
    else:
      ends = [np.mean([k.pos for k in kids], axis=0) if kids else 0.6 * np.asarray(b.pos)]
    for i, tip in enumerate(e for e in ends if np.linalg.norm(e) > 1e-3):
      b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0, 0], fromto=[0, 0, 0, *tip], name=f"{b.name}_approx{i}", mass=0)
      approx.append(b.name)
  if approx:
    flags.append(f"collision_approximated:{len(approx)}")
    notes.append(f"capsule collision geoms added on {len(approx)} links with no collision geometry")

  if hs.decompose:
    n = _decompose(spec, bodies)
    flags.append(f"convex_decomposed:{n}")

  rom = _resolve_ranges(spec, [j for j in joints if j.name not in follower])  # followers move with their leader
  if rom:
    cut = sum(1 for r in rom.values() if min(-r[0], r[1]) < np.pi - 1e-9)
    flags.append(f"range_full_rom:{len(rom)}")
    notes.append(f"{len(rom)} hinge joints had no usable range (unlimited or wider than a full turn): full range of motion "
                 f"+-pi, cut by a self-intersection sweep on {cut} of them")

  # missing effort: the default per joint, times the joints the actuator drives through coupling
  default_effort = lambda j: DEFAULT_EFFORT[j.type] * (1 + sum(l == j.name for l, _ in follower.values()))

  # actuators: keep the source model's (MJCF); otherwise a position servo per independent joint
  act_meta = []
  if driven or tendon_driven:
    for a in spec.actuators:
      origin = "source_model"
      if a.trntype == mujoco.mjtTrn.mjTRN_JOINT and a.biastype == mujoco.mjtBias.mjBIAS_NONE:
        # torque motor: the policy commands positions, so turn it into a servo with the motor's peak torque
        j = spec.joint(a.target)
        effort = float(np.abs(a.ctrlrange).max() * a.gear[0]) or max(j.actfrcrange[1], 0.0) or default_effort(j)
        _servo(a, j, effort)
        origin = "converted_from_motor"
        flags.append("actuator_converted_from_motor")
        act_meta.append(dict(name=a.name, target=a.target, origin=origin, effort=effort))
        continue
      act_meta.append(dict(name=a.name, target=a.target, origin=origin))
  else:
    flags.append("actuators_estimated")
    placeholder_effort = []
    for j in joints:
      if j.name in follower:
        continue
      lim = max(j.actfrcrange[1], 0.0)  # URDF effort lands here
      if lim > MAX_EFFORT[j.type]:
        lim = 0.0
        placeholder_effort.append(j.name)
      effort = lim or default_effort(j)
      a = spec.add_actuator(name=f"{j.name}_pos", target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
      _servo(a, j, effort)
      act_meta.append(dict(name=a.name, target=j.name, origin="estimated", effort=effort,
                           effort_source="urdf" if lim else "default", velocity=urdf_vel.get(j.name, 0.0)))
    if placeholder_effort:
      flags.append(f"effort_placeholder:{len(placeholder_effort)}")
      notes.append(f"{len(placeholder_effort)} URDF effort limits above {MAX_EFFORT[mujoco.mjtJoint.mjJNT_HINGE]} Nm treated as placeholders")

  # placeholder inertials (e.g. 1 kg / 1 kg m^2 per phalanx): recompute mass and inertia from collision geometry
  # (attached bodies keep their source compiler, so inertiafromgeom can't be switched on here)
  mc = spec.copy().compile()
  bad = []
  for b in bodies[1:]:
    i = mc.body(b.name).id
    if mc.body_mass[i] > 0 and np.sqrt(mc.body_inertia[i].max() / mc.body_mass[i]) > MAX_GYRATION:
      _box_inertial(mc, i, b)
      bad.append(b.name)
  if bad:
    flags.append(f"inertia_from_geometry:{len(bad)}")
    notes.append(f"{len(bad)} links had implausible inertia (radius of gyration > {MAX_GYRATION} m); mass/inertia recomputed from collision geoms")

  for j in joints:
    if j.armature < MIN_ARMATURE:
      j.armature = MIN_ARMATURE

  # placeholder damping (e.g. iCub's 1 Nm s/rad on every finger joint): the joint could not reach MIN_SPEED at its
  # actuator's effort. Replace with damping that costs 10% of the effort at MIN_SPEED.
  effort_of = {a["target"]: a["effort"] for a in act_meta if "effort" in a}
  for f, (l, _) in follower.items():
    if l in effort_of:
      effort_of[f] = effort_of[l]
  damped = []
  for j in joints:
    e = effort_of.get(j.name)
    if e and j.damping[0] * MIN_SPEED > e:
      j.damping = [0.1 * e / MIN_SPEED, 0, 0]
      damped.append(j.name)
  if damped:
    flags.append(f"damping_placeholder:{len(damped)}")
    notes.append(f"{len(damped)} joints had damping too high to move at {MIN_SPEED} rad/s with their actuator's effort; "
                 f"set to 10% of the effort at that speed")

  self_collision = add_self_collision(spec)
  # (after actuators: it simulates the hand closing)

  url, commit, _ = REPOS[hs.repo]
  meta = dict(
    id=hs.id, robot=hs.robot, source=hs.source, license=hs.license,
    provenance=dict(url=url, commit=commit, file=hs.file, hand_root=hs.hand_root),
    joints=[j.name for j in joints],
    actuators=act_meta,
    coupling=[dict(follower=f, leader=l, polycoef=c) for f, (l, c) in follower.items()],
    tendon_actuators=tendon_driven,
    removed_arm_joints=sorted(arm_joints),
    self_collision=self_collision,
    flags=flags, notes=notes,
  )
  return spec, meta


def git_head(repo: str) -> str:
  return subprocess.run(["git", "-C", str(SOURCES / repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
