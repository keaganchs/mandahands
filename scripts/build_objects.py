"""Build the object library: procedural + YCB objects, decomposition errors, splits, per-object contact counts.

  python scripts/fetch_ycb.py && python scripts/build_objects.py     # registry/objects-v1.json, docs/objects_report.md

Contact counts: every included object rests on the support; a reference hand descends over it from above, closes
and lifts (MJWarp, 16 worlds with random yaw and offset per object x hand). Peak contacts and constraint rows per
world are recorded against capacity.
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

VERSION = "objects-v1"
SEED = 0
N_PER_FAMILY = 15
MAX_ERR_ABS, MAX_ERR_REL = 0.005, 0.08  # exclude if decomposition p99 Hausdorff > max(5 mm, 8% of the bbox diagonal)
MAX_MIN_EXTENT = 0.11  # m: no dimension fits a single hand's grasp -> bimanual (backlog B3)
DENSITY = (300.0, 1500.0)  # kg/m^3, randomized per episode (M7); nominal 600 for the contact runs
FRICTION = (0.5, 1.2)
HANDS = ("leap_right", "g1_dex3_1_right")
NCONMAX, NJMAX, NWORLD = 200, 1500, 16


def ycb_one(obj):
  from mandahands.objects import ycb
  from mandahands.objects.parts import volume

  parts, info = ycb.decompose(obj)
  err = ycb.hausdorff(info.pop("mesh"), parts)
  return obj, dict(parts=len(parts), hull_verts=sum(len(p["verts"]) for p in parts), volume=volume(parts), **info, **err)


def proc_one(args):
  fam, i = args
  from mandahands.objects.parts import bbox, volume
  from mandahands.objects.procedural import generate

  ps = generate(SEED, fam, i)
  lo, hi = bbox(ps)
  return (fam, i), dict(parts=len(ps), volume=volume(ps), extents=(hi - lo).tolist())


def splits(objs, rng):
  """Per tier: hold out ~2 whole categories (test_category) and ~15% of the instances of the rest (test_instance)."""
  for tier in ("procedural", "ycb"):
    inc = [o for o in objs if o["tier"] == tier and not o["excluded"]]
    cats = sorted({o["category"] for o in inc})
    held = set(rng.choice(cats, 2, replace=False).tolist())
    for c in cats:
      group = [o for o in inc if o["category"] == c]
      if c in held:
        for o in group:
          o["split"] = "test_category"
        continue
      order = rng.permutation(len(group))
      n_test = max(1, round(0.15 * len(group))) if len(group) >= 3 else 0
      for k, j in enumerate(order):
        group[j]["split"] = "test_instance" if k < n_test else "train"


def contact_run(args):
  """(object entry, hand id) -> peak contacts / rows per world over descend-close-lift, and lift success."""
  o, hand_id = args
  import mujoco
  import mujoco_warp as mjw
  import warp as wp

  from mandahands.embodiments import registry
  from mandahands.embodiments.descriptors import _frame
  from mandahands.objects import library
  from mandahands.objects.parts import bbox
  from mandahands.sim.batch import contact_stats
  from mandahands.sim.builder import SceneCfg, build_scene
  from mandahands.sim.scripted import _quat_to, probe

  wp.config.quiet = True
  parts = library.parts(o["id"], VERSION)
  hand, _ = registry.real(hand_id)
  pr = probe(hand)
  R = _frame(pr["palm"], pr["tips_open"], pr["tips_closed"])
  q = _quat_to(R[2], np.array([0, 0, -1.0]))  # fingers point down
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, q)
  rot = rot.reshape(3, 3)
  lo, hi = bbox(parts)
  top = hi[2] - lo[2]
  tip_z = (pr["tips_open"] @ rot.T)[:, 2].min()
  pc = rot @ pr["palm"]
  gc = rot @ pr["tips_closed"].mean(0)  # the wrist origin can sit far from the hand (LEAP: 6 cm): centre where the fingers close
  palm_z = pc[2]
  z0 = top + 0.02 - tip_z
  z1 = max(top + 0.005 - palm_z, 0.005 - tip_z)
  m = build_scene(SceneCfg(), hand, wrist_pos=[-gc[0], -gc[1], z0], wrist_quat=q, obj_pos=[0, 0, -lo[2]], obj_parts=parts,
                  obj_density=600.0).compile()
  d = mujoco.MjData(m)
  for name, v in pr["qpos_open"].items():
    d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
  mujoco.mj_forward(m, d)
  rng = np.random.default_rng(0)
  mw = mjw.put_model(m)
  dw = mjw.make_data(m, nworld=NWORLD, nconmax=NCONMAX, njmax=NJMAX)
  # per world: yaw about the vertical and a +-1 cm offset of the wrist (and its mocap target)
  qpos, mpos, mquat = np.tile(d.qpos, (NWORLD, 1)), np.tile(d.mocap_pos, (NWORLD, 1, 1)), np.tile(d.mocap_quat, (NWORLD, 1, 1))
  wq = m.jnt_qposadr[m.joint("wrist_free").id]
  for w in range(NWORLD):
    yaw = np.r_[np.cos(rng.uniform(0, np.pi)), 0, 0, np.sin(rng.uniform(0, np.pi))]
    yaw /= np.linalg.norm(yaw)
    qq = np.zeros(4)
    mujoco.mju_mulQuat(qq, yaw, q)
    off = np.r_[rng.uniform(-0.01, 0.01, 2), 0]
    qpos[w, wq : wq + 3] += off
    qpos[w, wq + 3 : wq + 7] = qq
    mpos[w, 0] += off
    mquat[w, 0] = qq
  dw.qpos.assign(qpos.astype(np.float32))
  dw.mocap_pos.assign(mpos.astype(np.float32))
  dw.mocap_quat.assign(mquat.astype(np.float32))
  dt, peak = m.opt.timestep, {}
  phases = [(0.5, "descend"), (0.6, "close"), (0.5, "lift"), (0.3, "hold")]
  base_z = mpos[:, 0, 2].copy()
  for dur, ph in phases:
    n = int(dur / dt) // 10
    for k in range(n):
      s = (k + 1) / n
      z = base_z + {"descend": -(z0 - z1) * s, "close": -(z0 - z1), "lift": -(z0 - z1) + 0.1 * s, "hold": -(z0 - z1) + 0.1}[ph]
      mpos[:, 0, 2] = z
      dw.mocap_pos.assign(mpos.astype(np.float32))
      frac = {"descend": 0.0, "close": s}.get(ph, 1.0)
      dw.ctrl.assign(np.tile(pr["ctrl_open"] + frac * (pr["ctrl_close"] - pr["ctrl_open"]), (NWORLD, 1)).astype(np.float32))
      for _ in range(10):
        mjw.step(mw, dw)
      st = contact_stats(dw)
      peak = {k2: (v | peak.get(k2, 0)) if k2 == "overflow_mask" else max(v, peak.get(k2, v)) for k2, v in st.items()}
  oz = dw.xpos.numpy()[:, m.body("object").id, 2]
  return o["id"], hand_id, dict(ncon_world_max=peak["ncon_world_max"], nefc_world_max=peak["nefc_world_max"],
                                capacity_overflow_worlds=peak["capacity_overflow_worlds"], lifted=float((oz > -lo[2] + 0.05).mean()))


def sheet(objs, out_png, n=48):
  import mujoco
  from PIL import Image, ImageDraw

  from mandahands.objects import library
  from mandahands.objects.parts import add_to_body, bbox

  rng = np.random.default_rng(1)
  pick = [objs[i] for i in sorted(rng.choice(len(objs), min(n, len(objs)), replace=False))]
  tiles = []
  for o in pick:
    ps = library.parts(o["id"], VERSION)
    s = mujoco.MjSpec()
    s.worldbody.add_light(pos=[0, 0, 1], dir=[0, 0, -1])
    b = s.worldbody.add_body(name="o")
    add_to_body(s, b, ps, 500.0, rgba=[0.75, 0.7, 0.6, 1])
    m = s.compile()
    m.vis.headlight.ambient[:] = 0.4
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo, hi = bbox(ps)
    with mujoco.Renderer(m, 160, 160) as r:
      cam = mujoco.MjvCamera()
      cam.distance, cam.elevation, cam.azimuth = 2.2 * np.linalg.norm(hi - lo) + 0.02, -30, 45
      r.update_scene(d, cam)
      im = Image.fromarray(r.render())
    ImageDraw.Draw(im).text((3, 3), o["id"][:26], fill=(255, 255, 0))
    tiles.append(np.asarray(im))
  while len(tiles) % 8:
    tiles.append(np.zeros_like(tiles[0]))
  Image.fromarray(np.concatenate([np.concatenate(tiles[i : i + 8], 1) for i in range(0, len(tiles), 8)], 0)).save(out_png)


def geometry(workers, path):
  from mandahands.objects import ycb
  from mandahands.objects.procedural import FAMILIES

  t0 = time.time()
  with ProcessPoolExecutor(workers) as ex:
    proc = dict(ex.map(proc_one, [(f, i) for f in FAMILIES for i in range(N_PER_FAMILY)]))
    yres = dict(ex.map(ycb_one, ycb.objects()))
  print(f"geometry in {time.time() - t0:.0f} s", flush=True)

  objs = []
  for (fam, i), r in proc.items():
    objs.append(dict(id=f"proc-{fam}-{i:02d}", tier="procedural", category=fam, index=i, license="generated", **r,
                     decomposition=None, excluded=None))
  for name, r in yres.items():
    objs.append(dict(id=f"ycb-{name}", tier="ycb", category=ycb.category(name), source_name=name, license="CC-BY-4.0",
                     url=f"https://ycb-benchmarks.s3.amazonaws.com/data/google/{name}_google_16k.tgz", **r, decomposition=dict(ycb.COACD)))
  for o in objs:
    ext = np.asarray(o["extents"])
    reasons = []
    if o["tier"] == "ycb" and o["hausdorff_p99"] > max(MAX_ERR_ABS, MAX_ERR_REL * np.linalg.norm(ext)):
      reasons.append("decomposition_error")
    if ext.min() > MAX_MIN_EXTENT:
      reasons.append("too_large_single_hand")
    o["excluded"] = ", ".join(reasons) or None
  splits(objs, np.random.default_rng(SEED))
  git = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=40"], capture_output=True, text=True).stdout.strip()
  man = dict(version=VERSION, seed=SEED, git=git, n_per_family=N_PER_FAMILY, max_err=dict(abs=MAX_ERR_ABS, rel=MAX_ERR_REL),
             max_min_extent=MAX_MIN_EXTENT, density=DENSITY, friction=FRICTION, objects=objs)
  path.write_text(json.dumps(man, indent=0))
  print(f"wrote {path}", flush=True)
  sheet([o for o in objs if not o["excluded"]], "docs/objects_sheet.png")
  return man, objs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--skip-contacts", action="store_true")
  ap.add_argument("--contacts-only", action="store_true", help="rerun the contact stage on the existing manifest")
  ap.add_argument("--workers", type=int, default=os.cpu_count(), help="CoACD is memory-hungry: fewer workers on small machines")
  a = ap.parse_args()

  path = Path("registry") / f"{VERSION}.json"
  if a.contacts_only:
    man = json.loads(path.read_text())
    objs = man["objects"]
  else:
    man, objs = geometry(a.workers, path)
  contacts = {}
  if not a.skip_contacts:
    inc = [o for o in objs if not o["excluded"]]
    t0 = time.time()
    with ProcessPoolExecutor(4) as ex:  # GPU-bound; a few processes overlap compile and stepping
      for oid, hid, r in ex.map(contact_run, [(o, h) for o in inc for h in HANDS]):
        contacts.setdefault(oid, {})[hid] = r
    print(f"contacts in {time.time() - t0:.0f} s", flush=True)
    for o in objs:
      o["contacts"] = contacts.get(o["id"])
    man["contact_protocol"] = dict(hands=HANDS, nworld=NWORLD, nconmax=NCONMAX, njmax=NJMAX, density=600.0)
    path.write_text(json.dumps(man, indent=0))
  report(man)


def report(man):
  objs = man["objects"]
  inc = [o for o in objs if not o["excluded"]]
  L = [f"# Object library {man['version']}", "", f"Generated by `scripts/build_objects.py` (git `{man['git'][:10]}`).", "",
       "## Summary", "", "| tier | objects | excluded | train | test_instance | test_category | parts median / max |", "|---|---|---|---|---|---|---|"]
  for tier in ("procedural", "ycb"):
    t = [o for o in objs if o["tier"] == tier]
    cnt = lambda s: sum(o.get("split") == s for o in t)
    L.append(f"| {tier} | {len(t)} | {sum(bool(o['excluded']) for o in t)} | {cnt('train')} | {cnt('test_instance')} | {cnt('test_category')} | "
             f"{int(np.median([o['parts'] for o in t]))} / {max(o['parts'] for o in t)} |")
  held = sorted({(o["tier"], o["category"]) for o in objs if o.get("split") == "test_category"})
  L += ["", "Held-out categories: " + ", ".join(f"{c} ({t})" for t, c in held), ""]
  have = [o for o in inc if o.get("contacts")]
  if have:
    allr = [r for o in have for r in o["contacts"].values()]
    L += ["## Contact counts", "",
          f"Protocol: {man['contact_protocol']['nworld']} worlds per object x hand ({', '.join(man['contact_protocol']['hands'])}), "
          f"descend over the object, close, lift 10 cm, hold. Capacity nconmax {man['contact_protocol']['nconmax']} / njmax "
          f"{man['contact_protocol']['njmax']} per world.", "",
          f"- Peak contacts per world: median {np.median([r['ncon_world_max'] for r in allr]):.0f}, max {max(r['ncon_world_max'] for r in allr)}",
          f"- Peak constraint rows per world: median {np.median([r['nefc_world_max'] for r in allr]):.0f}, max {max(r['nefc_world_max'] for r in allr)}",
          f"- **Capacity overflow: {sum(r['capacity_overflow_worlds'] for r in allr)} worlds** across {len(allr)} runs",
          f"- Lifted (object rose > 5 cm, open-loop top grasp): {100 * np.mean([r['lifted'] for r in allr]):.0f}% of worlds", ""]
  L += ["## Objects", "", "| id | category | split | parts | extents (cm) | decomp. error p99 / mean (mm) | peak contacts / rows | lifted |",
        "|---|---|---|---|---|---|---|---|"]
  for o in sorted(objs, key=lambda o: (o["tier"], o["category"], o["id"])):
    ext = " x ".join(f"{100 * e:.1f}" for e in o["extents"])
    err = f"{1000 * o['hausdorff_p99']:.1f} / {1000 * o['mean']:.1f}" if o["tier"] == "ycb" else "exact"
    c = o.get("contacts") or {}
    cc = ", ".join(f"{r['ncon_world_max']}/{r['nefc_world_max']}" for r in c.values()) or "-"
    li = ", ".join(f"{100 * r['lifted']:.0f}%" for r in c.values()) or "-"
    L.append(f"| {o['id']} | {o['category']} | {o.get('split') or 'excluded: ' + str(o['excluded'])} | {o['parts']} | {ext} | {err} | {cc} | {li} |")
  L += ["", "![objects](objects_sheet.png)", ""]
  Path("docs/objects_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
  main()
