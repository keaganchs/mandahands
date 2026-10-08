"""Procedural hand generator: seeded topology + per-variant genome -> hand MjSpec rooted at body 'hand_mount'.

Hand frame: hand_mount is the wrist; the palm box hangs below it along -z and fingers attach to its bottom face.
Each finger has a direction e (where it points when straight) and a curl direction d; flexion axis = e x d, so
a positive flexion angle curls the finger toward d.

Layouts (topology level):
  anthro: fingers in a row along x curling toward +y (the palm front); thumb none / opposed (on the palm front,
          curling back toward the fingers) / coplanar (at the row end, splayed outward, curling toward +y)
  radial: fingers evenly around the palm, curling inward (2 = opposed gripper, 3 = three-finger gripper)
  jaw:    2 or 3 prismatic jaws sliding inward (parallel-jaw and three-jaw grippers)

Everything is sampled in two stages so a topology class (same bodies, joints, actuators, equalities, tendons,
geom types) has many geometric/dynamic variants: sample_topology(seed, t), then sample_genome(topology, seed, t, v).
Bump GENERATOR_VERSION whenever sampling or building changes what a (seed, t, v) produces.
"""

import json

import mujoco
import numpy as np

GENERATOR_VERSION = 4  # v2: one-directional and bidirectional tendons; v3: hands without kv / joint damping / frictionloss;
# v4: abduction up to +-1 rad
NSAMPLE = 12  # actuator history length: covers delays up to (NSAMPLE - 2) * 2 ms
MAX_LATENCY = 0.02
SCALE = (0.6, 1.8)

TIP_GEOM = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "box": mujoco.mjtGeom.mjGEOM_BOX, "pad": mujoco.mjtGeom.mjGEOM_ELLIPSOID}


def _rng(*key):
  return np.random.default_rng(list(key))


def _logu(r, lo, hi):
  return float(np.exp(r.uniform(np.log(lo), np.log(hi))))


def sample_topology(seed: int, t: int) -> dict:
  r = _rng(seed, t)
  # anthro hands fail the validity filter most often (~55% vs ~10% radial, ~0% jaw); this prior gives a valid set
  # that is about half anthropomorphic, like the real hands
  layout = str(r.choice(["anthro", "radial", "jaw"], p=[0.65, 0.25, 0.10]))
  if layout == "jaw":
    return dict(layout="jaw", n_fingers=int(r.choice([2, 3])), coupling=str(r.choice(["mimic", "full"])))
  # tendon_pull: one cable per finger that only pulls (closes), joints opened by return springs;
  # tendon_bidir: one cable per finger driven both ways (push-pull / antagonistic), no springs
  coupling = str(r.choice(["full", "mimic", "tendon_pull", "tendon_bidir", "synergy"], p=[0.4, 0.25, 0.1, 0.1, 0.15]))
  abduction = "none" if coupling == "synergy" else str(r.choice(["none", "all", "thumb"], p=[0.45, 0.35, 0.2]))
  topo = dict(
    layout=layout,
    n_fingers=int(r.integers(2, 6)),
    n_phalanges=int(r.choice([1, 2, 3, 4], p=[0.15, 0.3, 0.35, 0.2])),  # real fingers mostly have 2-3 flexion joints
    coupling=coupling,
    tip=str(r.choice(list(TIP_GEOM))),
    abd_order=str(r.choice(["abd_first", "flex_first"])),
    thumb="none",
  )
  if layout == "anthro":
    topo["thumb"] = str(r.choice(["none", "opposed", "coplanar"], p=[0.2, 0.5, 0.3]))
    if topo["thumb"] != "none":
      topo["thumb_phalanges"] = int(np.clip(topo["n_phalanges"] + r.integers(-1, 1), 1, 4))
  if abduction == "thumb" and topo["thumb"] == "none":
    abduction = "none"
  topo["abduction"] = abduction
  if abduction == "none":
    del topo["abd_order"]
  if coupling in ("mimic", "tendon_pull", "tendon_bidir") and topo["n_phalanges"] == 1 and topo.get("thumb_phalanges", 1) == 1:
    topo["coupling"] = "full"  # nothing to couple on single-joint fingers
  return topo


def topology_key(topo: dict) -> str:
  return json.dumps(topo, sort_keys=True, separators=(",", ":"))


def _fingers(topo: dict) -> list[dict]:
  """Per-finger topology: role, number of flexion joints, abduction joint."""
  out = []
  n_row = topo["n_fingers"] - (topo["thumb"] != "none")
  abd = topo["abduction"]
  for i in range(n_row):
    out.append(dict(role="finger", n=topo.get("n_phalanges", 1), abd=abd == "all"))
  if topo["thumb"] != "none":
    out.append(dict(role="thumb", n=topo["thumb_phalanges"], abd=abd in ("all", "thumb")))
  return out


def sample_genome(topo: dict, seed: int, t: int, v: int) -> dict:
  """All continuous parameters of one hand. Lengths in m, angles in rad."""
  r = _rng(seed, t, v, 1)
  s = _logu(r, *SCALE)  # overall scale: small gripper (0.6) to large robot hand (1.8, LEAP/Allegro-size); 1 ~ adult human
  # palm_density: motors and electronics usually sit in the palm (real hands are much heavier than their links)
  g = dict(scale=s, density=_logu(r, 400, 1500), palm_density=r.uniform(1, 5), friction=r.uniform(0.6, 1.2))
  # actuator class: hand-level means, jittered per actuator below
  g["act"] = dict(
    torque=_logu(r, 0.15, 3.0) * s**2,  # Nm (hinge); slide jaws use force_jaw
    force_jaw=_logu(r, 5, 60) * s**2,  # N
    stiff_frac=r.uniform(0.15, 0.6),  # position error (fraction of range) at which the servo saturates
    # kv / kp (s). 20% of hands have no servo derivative term and damp in the joint instead (as Menagerie models do)
    damp_ratio=0.0 if r.uniform() < 0.2 else _logu(r, 0.02, 0.1),
    armature=_logu(r, 1e-4, 5e-3) * s**2,
    damping=_logu(r, 5e-3, 0.1) * s**2 * (0.0 if r.uniform() < 0.3 else 1.0),  # many real models have none
    frictionloss=0.0 if r.uniform() < 0.5 else r.uniform(0, 0.05),  # fraction of the torque limit; often unmodeled
    velocity=_logu(r, 2, 10),  # rad/s (x 0.05 m per rad for slides): descriptor only, not simulated until M7
    backlash=r.uniform(0, 0.03),  # rad: descriptor only, not simulated until M7
    latency=r.uniform(0, MAX_LATENCY),  # s: simulated (actuator delay)
  )
  if g["act"]["damp_ratio"] == 0.0:  # damping lives in the joints: make sure there is some
    g["act"]["damping"] = _logu(r, 0.02, 0.3) * s**2
  jit = lambda: r.uniform(0.85, 1.15)

  if topo["layout"] == "jaw":
    n = topo["n_fingers"]
    open_r = s * r.uniform(0.03, 0.06)
    g["palm"] = [open_r + s * 0.015, open_r * (1 if n == 3 else r.uniform(0.3, 0.6)) + s * 0.01, s * r.uniform(0.01, 0.025)]
    phase = r.uniform(-0.15, 0.15, n)
    g["fingers"] = [
      dict(angle=2 * np.pi * i / n + float(phase[i]), radius=open_r, length=s * r.uniform(0.03, 0.07),
           width=s * r.uniform(0.008, 0.02), thick=s * r.uniform(0.004, 0.01), travel=open_r * r.uniform(0.7, 0.95),
           ratio=1.0)
      for i in range(n)
    ]
    g["acts"] = [dict(gain_jit=jit(), force_jit=jit()) for _ in range(n)]
    return g

  fingers = _fingers(topo)
  rad = s * r.uniform(0.006, 0.012)
  length = s * r.uniform(0.05, 0.11)
  n_row = sum(f["role"] == "finger" for f in fingers)
  g["palm"] = [s * r.uniform(0.025, 0.05), s * r.uniform(0.008, 0.02), s * r.uniform(0.015, 0.045)]
  # the palm must fit the fingers side by side (row) or around its rim (radial), else every wide-finger
  # sample overlaps at rest and is rejected
  if topo["layout"] == "radial":
    base_r = max(g["palm"][0] * 0.75, len(fingers) * rad * 1.3 / np.pi)
    g["palm"][0] = g["palm"][1] = max(g["palm"][0], base_r + rad)
    g["palm"][2] = s * r.uniform(0.008, 0.02)  # gripper palms are flat
  else:
    g["palm"][0] = max(g["palm"][0], n_row * rad * r.uniform(1.15, 1.4))
  ratio = r.uniform(0.65, 1.0)  # distal / proximal phalanx length
  fs = []
  for i, f in enumerate(fingers):
    thumb = f["role"] == "thumb"
    L = length * (r.uniform(0.6, 1.0) if thumb else r.uniform(0.85, 1.15))
    w = (ratio ** np.arange(f["n"])) * (1 + r.uniform(-0.1, 0.1, f["n"]))
    lengths = L * w / w.sum()
    fd = dict(
      lengths=lengths.tolist(),
      # phalanges at least 2.5 radii long, else non-adjacent links of the same finger overlap
      radius=min(rad * (r.uniform(1.0, 1.3) if thumb else r.uniform(0.9, 1.1)), lengths.min() / 2.5),
      taper=r.uniform(0.75, 1.0),
      flex_lo=(-r.uniform(0, 0.35, f["n"]) - np.r_[r.uniform(0, 0.15), np.zeros(f["n"] - 1)]).tolist(),  # base joint opens wider
      flex_hi=r.uniform(1.0, 2.2, f["n"]).tolist(),
      abd=r.uniform(0.15, 1.0) if f["abd"] else 0.0,  # real abduction joints reach +-0.5-1.05 rad
      axis_noise=r.normal(0, 0.06, (f["n"] + 1, 3)).tolist(),  # small rotation vector per joint axis (~3.5 deg sd)
      ratio=r.uniform(0.6, 1.2, f["n"]).tolist(),  # mimic / tendon coupling coefficients
      # tendon_pull hands: joint return spring, fraction of torque per rad; must beat frictionloss (<= 5% of torque)
      # over most of the range or the finger stays half closed
      spring=r.uniform(0.15, 0.4),
    )
    if topo["layout"] == "radial":
      fd["angle"] = 2 * np.pi * i / len(fingers) + r.uniform(-0.3, 0.3) / len(fingers)
      fd["base_r"] = base_r * r.uniform(1.0, 1.15)
      fd["splay"] = r.uniform(0, 0.4)  # tilt outward from -z
    elif thumb:
      fd["side"] = float(r.choice([-1, 1]))
      # opposed: pivot near the palm's bottom edge (relative to the thumb length), so the thumb curls under the palm
      # instead of into it; coplanar: fraction of palm length from the bottom edge
      fd["z"] = r.uniform(-0.3, 0.1) if topo["thumb"] == "opposed" else r.uniform(0.2, 1.0)
      fd["oblique"] = r.uniform(0.6, 1.2)  # coplanar: curl direction tilted toward the palm centre
      fd["spread"] = r.uniform(0.5, 1.2)  # opposed: rest direction tilted away from the fingers (open aperture)
      fd["x"] = r.uniform(0.2, 0.9)  # fraction of palm half-width
      fd["offset"] = s * r.uniform(0.015, 0.05)  # opposed: thenar, distance in front of the palm face
      fd["splay"] = r.uniform(0.2, 0.7) if topo["thumb"] == "coplanar" else r.uniform(0, 0.4)
    else:
      span = g["palm"][0] - rad
      fd["x"] = (-span + 2 * span * i / max(n_row - 1, 1)) if n_row > 1 else 0.0
      fd["x"] += r.uniform(-0.1, 0.1) * rad
      fd["splay"] = (fd["x"] / max(span, 1e-9)) * r.uniform(0, 0.2)  # outer fingers fan out
    fs.append(fd)
  g["fingers"] = fs
  n_act = len(_actuated(topo, fingers))
  g["acts"] = [dict(gain_jit=jit(), force_jit=jit()) for _ in range(n_act)]
  return g


def _actuated(topo, fingers) -> list[tuple]:
  """Actuator list as (finger, kind, joint index): kind is "flex" / "abd" / "tendon"."""
  out = []
  for i, f in enumerate(fingers):
    if f["abd"]:
      out.append((i, "abd", 0))
    if topo["coupling"] == "full":
      out += [(i, "flex", k) for k in range(f["n"])]
    elif topo["coupling"] == "mimic":
      out.append((i, "flex", 0))
    elif topo["coupling"].startswith("tendon"):
      out.append((i, "tendon", 0))
    elif topo["coupling"] == "synergy" and i == 0:
      out.append((0, "flex", 0))
  return out


def _rot(axis_angle: np.ndarray) -> np.ndarray:
  q = np.zeros(4)
  a = np.linalg.norm(axis_angle)
  mujoco.mju_axisAngle2Quat(q, axis_angle / a if a > 0 else np.array([1.0, 0, 0]), a)
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, q)
  return mat.reshape(3, 3)


def build(topo: dict, g: dict) -> tuple[mujoco.MjSpec, dict]:
  """Hand MjSpec rooted at 'hand_mount' plus metadata (joints, actuators with descriptor values, coupling)."""
  s = mujoco.MjSpec()
  s.compiler.autolimits = True
  s.compiler.degree = False
  s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  mount = s.worldbody.add_body(name="hand_mount")
  pw, pt, pl = g["palm"]
  common = dict(density=g["density"], friction=[g["friction"], 0.01, 0.0001])
  mount.add_geom(name="palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[pw, pt, pl], pos=[0, 0, -pl],
                 **{**common, "density": g["density"] * g["palm_density"]})
  a = g["act"]
  meta = dict(topology=topo, joints=[], actuators=[], coupling=[], tendons=[])

  def servo(name, target, trn, span, force, acts_i, slide=False):
    j = g["acts"][acts_i]
    kp = force * j["force_jit"] / (a["stiff_frac"] * span) * j["gain_jit"]
    act = s.add_actuator(name=name, target=target, trntype=trn)
    act.set_to_position(kp=kp, kv=a["damp_ratio"] * kp)
    act.forcelimited, act.forcerange = True, [-force * j["force_jit"], force * j["force_jit"]]
    act.delay, act.nsample = a["latency"], NSAMPLE
    meta["actuators"].append(dict(
      name=name, target=target, kp=kp, kv=a["damp_ratio"] * kp, force=force * j["force_jit"], force_min=-force * j["force_jit"],
      velocity=a["velocity"] * (0.05 if slide else 1), backlash=a["backlash"],
      latency=a["latency"], source="procedural",
    ))
    return act

  if topo["layout"] == "jaw":
    for i, f in enumerate(g["fingers"]):
      out = np.array([np.cos(f["angle"]), np.sin(f["angle"]), 0.0])
      b = mount.add_body(name=f"j{i}", pos=[*(f["radius"] * out[:2]), -2 * pl])
      jn = b.add_joint(name=f"j{i}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=-out, range=[0, f["travel"]],
                       armature=a["armature"] * 10, damping=a["damping"] * 10,
                       frictionloss=a["frictionloss"] * a["force_jaw"])
      # pad: thin along the slide direction, wide tangentially
      tang = np.cross([0, 0, 1], out)
      quat = np.zeros(4)
      mujoco.mju_mat2Quat(quat, np.column_stack([out, tang, [0, 0, 1]]).ravel())
      b.add_geom(name=f"j{i}_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=[f["thick"], f["width"], f["length"] / 2],
                 pos=[0, 0, -f["length"] / 2], quat=quat, **common)
      meta["joints"].append(dict(name=jn.name, finger=i, kind="slide"))
      if topo["coupling"] == "full" or i == 0:
        servo(f"j{i}_act", jn.name, mujoco.mjtTrn.mjTRN_JOINT, f["travel"], a["force_jaw"], len(meta["actuators"]), slide=True)
        s.actuators[-1].ctrlrange = [0, f["travel"]]
      else:
        lead = g["fingers"][0]
        r = f["travel"] / lead["travel"]
        s.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=jn.name, name2="j0_slide", data=[0, r, 0, 0, 0] + [0] * 6)
        meta["coupling"].append(dict(follower=jn.name, leader="j0_slide", polycoef=[0, r, 0, 0, 0]))
    return s, meta

  fingers = _fingers(topo)
  acts = _actuated(topo, fingers)
  leader = None
  for i, (f, fg) in enumerate(zip(fingers, g["fingers"])):
    thumb = f["role"] == "thumb"
    if topo["layout"] == "radial":
      out = np.array([np.cos(fg["angle"]), np.sin(fg["angle"]), 0.0])
      base = np.r_[fg["base_r"] * out[:2], -2 * pl]
      d = -out
      e = _rot(np.cross([0, 0, 1], out) * -fg["splay"]) @ np.array([0, 0, -1.0])  # tilt outward
    elif thumb and topo["thumb"] == "opposed":
      base = np.array([fg["side"] * fg["x"] * pw, pt + fg["offset"], -2 * pl + fg["z"] * sum(fg["lengths"])])
      mount.add_geom(name=f"thenar{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[fg["radius"], 0, 0],
                     fromto=[base[0], 0, max(base[2], -2 * pl + fg["radius"]), *base], **common)
      d = np.array([0, -1.0, 0])
      e = _rot(np.array([0, 1.0, 0]) * -fg["side"] * fg["splay"]) @ _rot(np.array([fg["spread"], 0, 0])) @ np.array([0, 0, -1.0])
    elif thumb:  # coplanar: at the row end, splayed outward in the palm plane
      base = np.array([fg["side"] * (pw + fg["radius"]), 0, -2 * pl * (1 - fg["z"])])
      d = np.array([-fg["side"] * np.sin(fg["oblique"]), np.cos(fg["oblique"]), 0])
      e = _rot(np.array([0, 1.0, 0]) * -fg["side"] * fg["splay"]) @ np.array([0, 0, -1.0])
    else:
      base = np.array([fg["x"], 0, -2 * pl])
      d = np.array([0, 1.0, 0])
      e = _rot(np.array([0, 1.0, 0]) * -fg["splay"]) @ np.array([0, 0, -1.0])
    d = d - e * (d @ e)
    d /= np.linalg.norm(d)
    noise = [_rot(np.asarray(x)) for x in fg["axis_noise"]]
    flex_axis = np.cross(e, d)

    parent, pos, rad = mount, base, fg["radius"]
    for k, L in enumerate(fg["lengths"]):
      b = parent.add_body(name=f"f{i}_l{k}", pos=pos)
      jnts = []
      if k == 0 and f["abd"]:
        jnts.append(("abd", noise[-1] @ d, [-fg["abd"], fg["abd"]]))
      jnts.append(("flex", noise[k] @ flex_axis, [fg["flex_lo"][k], fg["flex_hi"][k]]))
      if k == 0 and f["abd"] and topo["abd_order"] == "flex_first":
        jnts.reverse()
      for kind, ax, rng in jnts:
        jn = b.add_joint(name=f"f{i}_{kind}{k}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=ax, range=rng,
                         armature=a["armature"], damping=a["damping"], frictionloss=a["frictionloss"] * a["torque"])
        if topo["coupling"] == "tendon_pull" and kind == "flex":
          jn.stiffness = [fg["spring"] * a["torque"], 0, 0]  # return spring: the tendon only pulls
        meta["joints"].append(dict(name=jn.name, finger=i, kind=kind, depth=k, role=f["role"]))
      tip = k == len(fg["lengths"]) - 1
      rk = rad * fg["taper"] ** k
      gtype = TIP_GEOM[topo["tip"]] if tip else mujoco.mjtGeom.mjGEOM_CAPSULE
      geom = dict(name=f"f{i}_g{k}", type=gtype, **common)
      if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        geom.update(size=[rk, 0, 0], fromto=[*([0, 0, 0]), *(e * (L - rk))] if tip else [0, 0, 0, *(e * L)])
      else:  # box / pad: oriented with z along e and x along d; pad is flatter toward the curl side
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.column_stack([d, np.cross(e, d), e]).ravel())
        geom.update(size=[rk * (0.75 if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID else 1), rk, L / 2], pos=e * L / 2, quat=quat)
      b.add_geom(**geom)
      parent, pos = b, e * L

    flex = [j for j in meta["joints"] if j["finger"] == i and j["kind"] == "flex"]
    for ai, (fi, kind, k) in enumerate(acts):
      if fi != i:
        continue
      if kind == "abd":
        servo(f"f{i}_abd_act", f"f{i}_abd0", mujoco.mjtTrn.mjTRN_JOINT, 2 * fg["abd"], a["torque"], ai)
        s.actuators[-1].ctrlrange = [-fg["abd"], fg["abd"]]
      elif kind == "flex":
        servo(f"f{i}_flex{k}_act", f"f{i}_flex{k}", mujoco.mjtTrn.mjTRN_JOINT, fg["flex_hi"][k] - fg["flex_lo"][k], a["torque"], ai)
        s.actuators[-1].ctrlrange = [fg["flex_lo"][k], fg["flex_hi"][k]]
      else:  # tendon: one cable through every flexion joint of the finger
        coefs = fg["ratio"]
        ten = s.add_tendon(name=f"f{i}_tendon")
        for j, c in zip(flex, coefs):
          ten.wrap_joint(j["name"], c)
        hi = float(np.dot(coefs, fg["flex_hi"]))
        servo(f"f{i}_tendon_act", ten.name, mujoco.mjtTrn.mjTRN_TENDON, hi, a["torque"], ai)
        s.actuators[-1].ctrlrange = [0, hi]
        if topo["coupling"] == "tendon_pull":  # a cable can't push: closing force only, springs open the finger
          s.actuators[-1].forcerange = [0, s.actuators[-1].forcerange[1]]
          meta["actuators"][-1]["force_min"] = 0.0
        meta["tendons"].append(dict(name=ten.name, actuator=f"f{i}_tendon_act", joints=[j["name"] for j in flex], coefs=list(coefs),
                                    directional=topo["coupling"] == "tendon_pull"))

    # equality coupling: mimic (distal joints follow the finger's proximal joint) or synergy (every flexion
    # joint follows finger 0's proximal joint). Ratios are scaled so leader and follower reach their limits together.
    if topo["coupling"] in ("mimic", "synergy"):
      if leader is None:
        leader = (flex[0]["name"], fg["flex_hi"][0])
      lead_name, lead_hi = (flex[0]["name"], fg["flex_hi"][0]) if topo["coupling"] == "mimic" else leader
      for k, j in enumerate(flex):
        if j["name"] == lead_name:
          continue
        rr = fg["flex_hi"][k] / lead_hi * fg["ratio"][k] ** 0.25  # jitter, within a ~5% band of the limit match
        s.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=j["name"], name2=lead_name, data=[0, rr, 0, 0, 0] + [0] * 6)
        meta["coupling"].append(dict(follower=j["name"], leader=lead_name, polycoef=[0, rr, 0, 0, 0]))
  return s, meta
