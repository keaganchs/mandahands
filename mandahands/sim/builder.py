"""MjSpec scene assembly: floating wrist (mocap + soft weld), hand, object, support."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class SceneCfg:
  timestep: float = 0.002
  iterations: int = 10
  ls_iterations: int = 10
  impratio: float = 3.0  # 10 blew up light real-hand fingers on contact (M2 report); 1 loses R2/JVRC grasps
  integrator: int = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # Euler blows up stiff servos on small real-hand links
  condim: int = 4  # finger-object pairs; MuJoCo takes the max of the two geoms
  weld_solref: tuple[float, float] = (0.02, 1.0)  # soft wrist, mimics arm compliance
  # test hand: radially placed fingers, fully actuated hinge chains
  n_fingers: int = 3
  n_phalanges: int = 2
  phalanx_len: float = 0.04
  phalanx_radius: float = 0.008
  kp: float = 2.0
  kv: float = 0.05
  # object: "box" / "sphere" / "cylinder" primitive (obj_half = half extent / radius), or "mesh" made of n_parts convex hulls (stands in for a CoACD decomposition)
  obj: str = "box"
  obj_half: float = 0.03
  n_parts: int = 1
  seed: int = 0


# collision bits: hand geoms collide with object and support, never with each other
HAND_BITS = dict(contype=2, conaffinity=1)
OBJ_BITS = dict(contype=1, conaffinity=3)


def simple_hand(cfg: SceneCfg) -> mujoco.MjSpec:
  """Hand-written radial test hand rooted at body 'hand_mount'; fingers hang along -z and curl inward."""
  s = mujoco.MjSpec()
  s.compiler.autolimits = True
  s.compiler.degree = False  # MjSpec defaults to degrees
  r_base = cfg.obj_half + 0.015
  half = r_base + cfg.phalanx_radius
  mount = s.worldbody.add_body(name="hand_mount")
  mount.add_geom(name="palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, 0.01], mass=0.3)
  for f in range(cfg.n_fingers):
    th = 2 * np.pi * f / cfg.n_fingers
    radial = np.array([np.cos(th), np.sin(th), 0.0])
    axis = np.cross([0, 0, 1], radial)  # positive angle curls the finger inward
    parent, pos = mount, np.r_[r_base * radial[:2], -0.01]
    for p in range(cfg.n_phalanges):
      b = parent.add_body(name=f"f{f}_p{p}", pos=pos)
      j = b.add_joint(
        name=f"f{f}_j{p}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis,
        range=[-0.4, 1.2] if p == 0 else [0.0, 1.5], damping=0.01, armature=0.001,
      )
      b.add_geom(
        name=f"f{f}_g{p}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[cfg.phalanx_radius, 0, 0],
        fromto=[0, 0, 0, 0, 0, -cfg.phalanx_len], mass=0.02,
      )
      a = s.add_actuator(name=f"f{f}_a{p}", target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
      a.set_to_position(kp=cfg.kp, kv=cfg.kv)
      a.ctrlrange = j.range
      a.ctrllimited = True
      parent, pos = b, [0, 0, -cfg.phalanx_len]
  return s


def build_scene(
  cfg: SceneCfg,
  hand: mujoco.MjSpec | None = None,
  wrist_pos=None,
  wrist_quat=(1.0, 0.0, 0.0, 0.0),
  obj_pos=None,
) -> mujoco.MjSpec:
  """Scene with `hand` (an MjSpec rooted at body 'hand_mount') mounted on the floating wrist.

  Defaults (hand=None) give the M1 test hand hovering over the object.
  """
  if hand is None:
    hand = simple_hand(cfg)
    if wrist_pos is None:
      wrist_pos = [0, 0, 0.015 + cfg.n_phalanges * cfg.phalanx_len + 0.01]
  s = mujoco.MjSpec()
  s.modelname = "mandahands_scene"
  s.option.timestep = cfg.timestep
  s.option.iterations = cfg.iterations
  s.option.ls_iterations = cfg.ls_iterations
  s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  s.option.impratio = cfg.impratio
  s.option.integrator = cfg.integrator
  s.compiler.autolimits = True
  s.compiler.degree = False  # MjSpec defaults to degrees

  wb = s.worldbody
  wb.add_geom(name="support", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.5, 0.5, 0.01])

  # wrist: free body welded to a mocap target, so it follows commanded poses with compliance
  wb.add_body(name="wrist_target", mocap=True, pos=wrist_pos, quat=wrist_quat)
  wrist = wb.add_body(name="wrist", pos=wrist_pos, quat=wrist_quat)
  wrist.add_freejoint(name="wrist_free")
  wrist.add_geom(name="wrist_mass", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], mass=0.05, contype=0, conaffinity=0)
  s.add_equality(
    name="wrist_weld", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
    name1="wrist", name2="wrist_target", solref=list(cfg.weld_solref),
    data=[0.0] * 10 + [1.0],  # zero relpose = keep the qpos0 offset; MjSpec's default data is not zero
  )
  # attach a copy: attaching an already-compiled spec corrupts equality ids
  mount = wrist.add_frame().attach_body(hand.copy().body("hand_mount"), "", "")
  stack = [mount]
  while stack:
    b = stack.pop()
    stack.extend(b.bodies)
    for g in b.geoms:
      if g.contype or g.conaffinity:
        g.contype, g.conaffinity = HAND_BITS["contype"], HAND_BITS["conaffinity"]
        # one contact model for every hand (randomized later); sources disagree wildly, e.g. LEAP's
        # solref=1e-4 makes light fingers blow up on contact. Friction stays per hand.
        g.solref, g.solimp = [0.02, 1.0], [0.9, 0.95, 0.001, 0.5, 2.0]
        g.margin = g.gap = 0.0
        g.priority, g.solmix = 0, 1.0

  h = cfg.obj_half
  obj = wb.add_body(name="object", pos=obj_pos if obj_pos is not None else [0, 0, h])
  obj.add_freejoint(name="object_free")
  # support: an object-to-world weld, released (d.eq_active) once the hand has closed
  s.add_equality(
    name="object_support", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
    name1="object", data=[0.0] * 10 + [1.0], active=False,
  )
  common = dict(condim=cfg.condim, friction=[1.0, 0.01, 0.0001], density=500, **OBJ_BITS)
  if cfg.obj == "box":
    obj.add_geom(name="obj_g0", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h, h, h], **common)
  elif cfg.obj == "sphere":
    obj.add_geom(name="obj_g0", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[h, 0, 0], **common)
  elif cfg.obj == "cylinder":
    obj.add_geom(name="obj_g0", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[h, h, 0], **common)
  elif cfg.obj == "mesh":
    # n_parts jittered slabs along x; MuJoCo collides each mesh as its convex hull
    rng = np.random.default_rng(cfg.seed)
    edges = np.linspace(-h, h, cfg.n_parts + 1)
    for i in range(cfg.n_parts):
      corners = np.array([[x, y, z] for x in edges[i : i + 2] for y in (-h, h) for z in (-h, h)])
      corners[:, 1:] *= rng.uniform(0.8, 1.0, size=(8, 2))
      s.add_mesh(name=f"obj_m{i}", uservert=corners.ravel().tolist())
      obj.add_geom(name=f"obj_g{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"obj_m{i}", **common)
  else:
    raise ValueError(f"unknown obj {cfg.obj!r}")
  return s


def close_ctrl(m: mujoco.MjModel, frac: float) -> np.ndarray:
  """Scripted grasp command: interpolate each actuator from 0 toward frac of its upper limit."""
  return frac * m.actuator_ctrlrange[:, 1]
