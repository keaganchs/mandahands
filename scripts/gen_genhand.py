"""Generate a GENHAND dataset: sample topologies x variants, run the validity filter, assign splits, measure
descriptor-space coverage against the real hands, and write the manifest + report.

  python scripts/gen_genhand.py                       # GENHAND-1K v2: registry/genhand-1k-v2.json, docs/genhand_report.md
  python scripts/gen_genhand.py --topologies 20 --variants 4 --name genhand-dev --out /tmp/x   # quick run

Needs data/real_hands/ (scripts/extract_real.py) for the coverage comparison.
"""

import argparse
import collections
import hashlib
import itertools
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

SPLIT_SEED = 1
OOD_TOPOLOGY_FRAC = 0.1
OOD_SCALE_FRAC = 0.075  # of the log-scale sampling range, at each end
TEST_ID_FRAC = 0.1
SCALING_N = (8, 32, 128, 512)


def run_one(args):
  seed, t, v = args
  from mandahands.embodiments.procedural.generator import build, sample_genome, sample_topology
  from mandahands.embodiments.procedural.validity import check
  from mandahands.embodiments.registry import spec_hash

  topo = sample_topology(seed, t)
  g = sample_genome(topo, seed, t, v)
  spec, meta = build(topo, g)
  h = spec_hash(spec)
  t0 = time.time()
  try:
    r = check(spec)
  except Exception as e:  # report, don't hide: counted as its own rejection reason
    r = dict(valid=False, reason=f"error: {type(e).__name__}: {e}")
  m = spec.compile()
  return dict(
    id=f"gh-{t:04d}-{v:02d}", t=t, v=v, hash=h, valid=bool(r["valid"]), reason=r["reason"], n_ref=r.get("n_ref"),
    ref="".join("x" if x["ok"] else "." for x in r.get("ref", [])), summary=r.get("summary"),
    n_pairs=r.get("self_collision", {}).get("n_pairs"), scale=g["scale"], njnt=m.njnt, nu=m.nu,
    sec=time.time() - t0,
  )


def real_one(hid):
  import mujoco

  from mandahands.embodiments.procedural.validity import summary
  from mandahands.sim.scripted import probe

  spec = mujoco.MjSpec.from_file(f"data/real_hands/{hid}/hand.xml")
  return hid, summary(spec.copy().compile(), probe(spec))


def splits(hands, topos, scale_range):
  """Valid hands -> train / test_id / test_ood_topology / test_ood_scale; scaling subsets are nested train prefixes."""
  from mandahands.embodiments.procedural.generator import topology_key

  rng = np.random.default_rng(SPLIT_SEED)
  valid = [h for h in hands if h["valid"]]
  keys = sorted({topology_key(topos[h["t"]]) for h in valid})
  ood_keys = set(rng.choice(keys, max(1, round(OOD_TOPOLOGY_FRAC * len(keys))), replace=False))
  lo, hi = np.log(scale_range)
  cut = OOD_SCALE_FRAC * (hi - lo)
  rest = []
  for h in valid:
    if topology_key(topos[h["t"]]) in ood_keys:
      h["split"] = "test_ood_topology"
    elif not lo + cut <= np.log(h["scale"]) <= hi - cut:
      h["split"] = "test_ood_scale"
    else:
      rest.append(h)
  rng.shuffle(rest)
  n_test = round(TEST_ID_FRAC * len(rest))
  for i, h in enumerate(rest):
    h["split"] = "test_id" if i < n_test else "train"
  train = sorted(h["id"] for h in rest[n_test:])
  order = list(rng.permutation(train))
  return {str(n): sorted(order[:n]) for n in SCALING_N if n < len(order)} | {"all": train}


def coverage(hands, real, out_png):
  """Standardize summaries by train statistics; nearest-neighbour distances real -> train vs leave-one-out train -> train."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  from mandahands.embodiments.procedural.validity import FEATURES

  X = lambda hs: np.array([[h["summary"][f] for f in FEATURES] for h in hs])
  train = [h for h in hands if h.get("split") == "train"]
  Xt = X(train)
  mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
  Z = lambda x: (x - mu) / sd
  Zt = Z(Xt)
  nn = lambda z: np.sqrt(((Zt[None] - z[:, None]) ** 2).sum(-1)).min(1)
  # baseline: leave-one-out nearest neighbour within train (stabler than the ~80 test_id hands)
  dt = np.sqrt(((Zt[None] - Zt[:, None]) ** 2).sum(-1))
  np.fill_diagonal(dt, np.inf)
  base = dt.min(1)
  Xr = np.array([[s[f] for f in FEATURES] for s in real.values()])
  d_real = nn(Z(Xr))
  p1, p99 = np.percentile(Xt, 1, 0), np.percentile(Xt, 99, 0)
  rows = []
  for (hid, s), dist, x in zip(real.items(), d_real, Xr):
    outside = [f"{f} {'low' if v < a else 'high'}" for f, v, a, b in zip(FEATURES, x, p1, p99) if not a <= v <= b]
    rows.append(dict(id=hid, nn=float(dist), pct=float((base < dist).mean() * 100), outside=outside,
                     nearest=[train[i]["id"] for i in np.argsort(np.sqrt(((Zt - Z(x)) ** 2).sum(-1)))[:3]]))

  # PCA on train, everything projected into it
  _, _, vt = np.linalg.svd(Zt, full_matrices=False)
  P = lambda z: z @ vt[:2].T
  fig, ax = plt.subplots(1, 2, figsize=(13, 6))
  groups = [("train", "#bbbbbb", 8), ("test_id", "#4c72b0", 10), ("test_ood_topology", "#55a868", 12), ("test_ood_scale", "#c44e52", 12)]
  for split, col, sz in groups:
    hs = [h for h in hands if h.get("split") == split]
    if hs:
      ax[0].scatter(*P(Z(X(hs))).T, s=sz, c=col, label=f"{split} ({len(hs)})", alpha=0.7, lw=0)
  pr = P(Z(Xr))
  ax[0].scatter(*pr.T, s=40, c="black", marker="*", label=f"real ({len(real)})")
  for (x, y), hid in zip(pr, real):
    ax[0].annotate(hid.replace("_right", "").replace("_left", ""), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
  ax[0].set(title="Hand summary features, PCA of standardized train set", xlabel="PC1", ylabel="PC2")
  ax[0].legend(fontsize=8)
  fi, fj = FEATURES.index("log_reach"), FEATURES.index("n_actuators")
  ax[1].scatter(Xt[:, fi], Xt[:, fj] + np.random.default_rng(0).uniform(-0.3, 0.3, len(Xt)), s=8, c="#bbbbbb", lw=0, label="train (jittered)")
  ax[1].scatter(Xr[:, fi], Xr[:, fj], s=40, c="black", marker="*", label="real")
  for x, hid in zip(Xr, real):
    ax[1].annotate(hid.replace("_right", "").replace("_left", ""), (x[fi], x[fj]), fontsize=7, xytext=(3, 3), textcoords="offset points")
  ax[1].set(title="Reach vs actuator count", xlabel="log mean palm-to-fingertip distance (m)", ylabel="actuators")
  ax[1].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(out_png, dpi=110)
  expl = (np.linalg.svd(Zt, compute_uv=False) ** 2)
  return dict(rows=rows, baseline=dict(p50=float(np.median(base)), p95=float(np.percentile(base, 95)), p99=float(np.percentile(base, 99)),
                                       max=float(base.max())), pca_explained=(expl[:2] / expl.sum()).tolist(),
              train_range={f: [float(a), float(b)] for f, a, b in zip(FEATURES, p1, p99)})


def sheet(hands, topos, seed, out_png, n=36):
  """Contact sheet of random valid hands in their open pose."""
  import mujoco
  from PIL import Image, ImageDraw

  from mandahands.embodiments.procedural.generator import build, sample_genome
  from mandahands.sim.scripted import probe

  rng = np.random.default_rng(0)
  pick = rng.choice([h for h in hands if h["valid"]], n, replace=False)
  tiles = []
  for h in pick:
    topo = topos[h["t"]]
    g = sample_genome(topo, seed, h["t"], h["v"])
    spec, _ = build(topo, g)
    pr = probe(spec)
    m = spec.compile()
    m.vis.headlight.ambient[:] = 0.5
    d = mujoco.MjData(m)
    for name, q in pr["qpos_open"].items():
      d.qpos[m.jnt_qposadr[m.joint(name).id]] = q
    mujoco.mj_kinematics(m, d)
    with mujoco.Renderer(m, 200, 200) as r:
      cam = mujoco.MjvCamera()
      cam.distance, cam.elevation, cam.azimuth = 0.32 * g["scale"], -25, 120
      cam.lookat[:] = [0, 0.01 * g["scale"], -0.07 * g["scale"]]
      r.update_scene(d, cam)
      im = Image.fromarray(r.render())
    lab = f"{h['id']} {topo['layout']} {topo['n_fingers']}f {topo['coupling']}"
    ImageDraw.Draw(im).text((3, 3), lab, fill=(255, 255, 0))
    tiles.append(np.asarray(im))
  rows = [np.concatenate(tiles[i : i + 6], 1) for i in range(0, n, 6)]
  Image.fromarray(np.concatenate(rows, 0)).save(out_png)


def tally(hands, topos, field):
  c = collections.defaultdict(lambda: [0, 0])
  for h in hands:
    c[str(topos[h["t"]].get(field, "-"))][h["valid"]] += 1
  return {k: dict(valid=v[1], rejected=v[0], rate=v[0] / (v[0] + v[1])) for k, v in sorted(c.items())}


def report(man, cov, path, sheet_png, cov_png):
  from mandahands.embodiments.procedural.validity import FEATURES

  hands = man["hands"]
  n, nv = len(hands), sum(h["valid"] for h in hands)
  L = [f"# {man['name']} report", "",
       f"Generated by `scripts/gen_genhand.py` (generator v{man['generator_version']}, seed {man['seed']}, "
       f"{man['n_topologies']} topology draws x {man['n_variants']} variants, git `{man['git'][:10]}`).", "",
       "## Validity filter", "",
       f"- **{nv} / {n} valid ({100 * nv / n:.1f}%), rejection rate {100 * (n - nv) / n:.1f}%**",
       f"- Filter: no self-penetration (> {man['filter']['pen_tol'] * 1000:.0f} mm, non-adjacent links) at rest and at the open pose; "
       f"scripted close-and-lift lifts >= {man['filter']['k_min']} of {len(man['filter']['ref_objects'])} reference objects "
       f"({', '.join(sorted({o for o, _ in man['filter']['ref_objects']}))} at half-extents "
       f"{', '.join(f'{1000 * s:.0f}' for s in sorted({s for _, s in man['filter']['ref_objects']}))} mm).", "",
       "| reason | hands |", "|---|---|"]
  for k, v in collections.Counter(h["reason"] or "valid" for h in hands).most_common():
    L.append(f"| {k} | {v} |")
  L += ["", "Reference objects lifted, per hand that reached the grasp test:", "", "| lifted | hands |", "|---|---|"]
  for k, v in sorted(collections.Counter(h["n_ref"] for h in hands if h["n_ref"] is not None).items()):
    L.append(f"| {k} | {v} |")
  for field in ("layout", "thumb", "coupling", "n_fingers", "n_phalanges", "tip", "abduction"):
    L += ["", f"Rejection by `{field}`:", "", "| value | valid | rejected | rate |", "|---|---|---|---|"]
    for k, v in man["tally"][field].items():
      L.append(f"| {k} | {v['valid']} | {v['rejected']} | {100 * v['rate']:.0f}% |")
  L += ["", "## Splits", "", "| split | hands |", "|---|---|"]
  for k, v in collections.Counter(h.get("split", "rejected") for h in hands).most_common():
    L.append(f"| {k} | {v} |")
  L += ["", f"Scaling subsets (nested prefixes of a seeded permutation of train): "
        + ", ".join(f"{k}: {len(v)}" for k, v in man["scaling_subsets"].items()), "",
        f"Topology classes with a valid hand: {man['classes']['n']}; valid hands per class: "
        f"median {man['classes']['median']}, max {man['classes']['max']}; "
        f"classes whose variants differ in self-collision pair count: {man['classes']['npair_varies']}.", "",
        "## Coverage: real hands vs the procedural cloud", "",
        f"Hand-level summary vector ({', '.join(f'`{f}`' for f in FEATURES)}), standardized by train statistics. "
        f"Nearest-neighbour distance from each real hand to the train set, compared with leave-one-out train -> train "
        f"(median {cov['baseline']['p50']:.2f}, p95 {cov['baseline']['p95']:.2f}, p99 {cov['baseline']['p99']:.2f}, max {cov['baseline']['max']:.2f}). "
        f"PCA: first two components explain {100 * sum(cov['pca_explained']):.0f}% of the train variance.", "",
        f"![coverage]({cov_png})", "",
        "| real hand | NN distance | percentile vs train | features outside train p1-p99 | nearest train hands |", "|---|---|---|---|---|"]
  for r in sorted(cov["rows"], key=lambda r: -r["nn"]):
    flag = " **out of support**" if r["nn"] > cov["baseline"]["p99"] else ""
    L.append(f"| {r['id']} | {r['nn']:.2f}{flag} | {r['pct']:.0f} | {', '.join(r['outside']) or '-'} | {', '.join(r['nearest'])} |")
  L += ["", "## Sample of valid hands (open pose)", "", f"![sheet]({sheet_png})", ""]
  path.write_text("\n".join(L) + "\n")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--name", default="genhand-1k-v2")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--topologies", type=int, default=200)
  ap.add_argument("--variants", type=int, default=8)
  ap.add_argument("--workers", type=int, default=os.cpu_count())
  ap.add_argument("--out", default=".", help="root for registry/ and docs/")
  a = ap.parse_args()

  from mandahands.embodiments.procedural import validity as V
  from mandahands.embodiments.procedural.generator import GENERATOR_VERSION, SCALE, sample_topology, topology_key
  from mandahands.sim.scripted import SQUEEZE

  out = Path(a.out)
  (out / "registry").mkdir(parents=True, exist_ok=True)
  (out / "docs").mkdir(parents=True, exist_ok=True)
  topos = [sample_topology(a.seed, t) for t in range(a.topologies)]
  t0 = time.time()
  jobs = list(itertools.product([a.seed], range(a.topologies), range(a.variants)))
  with ProcessPoolExecutor(a.workers) as ex:
    hands = list(ex.map(run_one, jobs, chunksize=4))
  print(f"filtered {len(hands)} hands in {time.time() - t0:.0f} s; valid {sum(h['valid'] for h in hands)}")

  subsets = splits(hands, topos, SCALE)
  real_ids = sorted(p.parent.name for p in Path("data/real_hands").glob("*/hand.xml"))
  with ProcessPoolExecutor(a.workers) as ex:
    real = dict(ex.map(real_one, real_ids))

  cls = collections.defaultdict(list)
  for h in hands:
    if h["valid"]:
      cls[topology_key(topos[h["t"]])].append(h)
  sizes = sorted(len(v) for v in cls.values())
  git = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=40"], capture_output=True, text=True).stdout.strip()
  man = dict(
    name=a.name, generator_version=GENERATOR_VERSION, seed=a.seed, n_topologies=a.topologies, n_variants=a.variants,
    git=git, filter=dict(ref_objects=V.REF_OBJECTS, k_min=V.K_MIN, tries=V.TRIES, pen_tol=V.PEN_TOL, squeeze=SQUEEZE),
    split_seed=SPLIT_SEED, topologies=topos, scaling_subsets=subsets,
    classes=dict(n=len(cls), median=int(np.median(sizes)), max=sizes[-1],
                 npair_varies=sum(len({h["n_pairs"] for h in v}) > 1 for v in cls.values())),
    tally={f: tally(hands, topos, f) for f in ("layout", "thumb", "coupling", "n_fingers", "n_phalanges", "tip", "abduction")},
    hands=hands,
  )
  cov_png, sheet_png = out / "docs" / "genhand_coverage.png", out / "docs" / "genhand_sheet.png"
  cov = coverage(hands, real, cov_png)
  man["coverage"] = cov
  man["real_summary"] = real
  (out / "registry" / f"{a.name}.json").write_text(json.dumps(man, separators=(",", ":")))
  sheet(hands, topos, a.seed, sheet_png)
  report(man, cov, out / "docs" / "genhand_report.md", sheet_png.name, cov_png.name)
  digest = hashlib.sha1((out / "registry" / f"{a.name}.json").read_bytes()).hexdigest()[:12]
  print(f"wrote {out / 'registry' / a.name}.json ({digest}), {out / 'docs' / 'genhand_report.md'}")


if __name__ == "__main__":
  main()
