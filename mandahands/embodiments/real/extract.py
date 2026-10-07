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
  """Position servo with full effort at a quarter-range error. Gains are guesses; flagged in metadata."""
  # unlimited, or wider than a full turn (placeholder, e.g. R2's +-4 rad): command within +-pi/2
  placeholder = not (j.range[0] < j.range[1] and j.range[1] - j.range[0] <= 2 * np.pi)
  rng = [-np.pi / 2, np.pi / 2] if placeholder else list(j.range)
  kp = effort / max(0.25 * (rng[1] - rng[0]), 1e-3)
  a.set_to_position(kp=kp, kv=0.05 * kp)
  a.gear = [1, 0, 0, 0, 0, 0]
  a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
  a.ctrlrange = rng
  return placeholder
  a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
  a.forcerange = [-effort, effort]


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

  # actuators: keep the source model's (MJCF); otherwise a position servo per independent joint
  act_meta = []
  if driven or tendon_driven:
    for a in spec.actuators:
      origin = "source_model"
      if a.trntype == mujoco.mjtTrn.mjTRN_JOINT and a.biastype == mujoco.mjtBias.mjBIAS_NONE:
        # torque motor: the policy commands positions, so turn it into a servo with the motor's peak torque
        j = spec.joint(a.target)
        effort = float(np.abs(a.ctrlrange).max() * a.gear[0]) or max(j.actfrcrange[1], 0.0) or DEFAULT_EFFORT[j.type]
        _servo(a, j, effort)
        origin = "converted_from_motor"
        flags.append("actuator_converted_from_motor")
      act_meta.append(dict(name=a.name, target=a.target, origin=origin))
  else:
    flags.append("actuators_estimated")
    placeholder_effort, placeholder_range = [], []
    for j in joints:
      if j.name in follower:
        continue
      lim = max(j.actfrcrange[1], 0.0)  # URDF effort lands here
      if lim > MAX_EFFORT[j.type]:
        lim = 0.0
        placeholder_effort.append(j.name)
      effort = lim or DEFAULT_EFFORT[j.type]
      a = spec.add_actuator(name=f"{j.name}_pos", target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
      if _servo(a, j, effort):
        placeholder_range.append(j.name)
      act_meta.append(dict(name=a.name, target=j.name, origin="estimated", effort=effort,
                           effort_source="urdf" if lim else "default", velocity=urdf_vel.get(j.name, 0.0)))
    if placeholder_range:
      flags.append(f"range_placeholder:{len(placeholder_range)}")
      notes.append(f"{len(placeholder_range)} joints unlimited or wider than 2*pi; actuator command clamped to +-pi/2")
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

  self_collision = add_self_collision(spec) if hs.self_collision else dict(n_pairs=0, n_fingers=0, disabled=True)
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
