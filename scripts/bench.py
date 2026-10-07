"""Throughput benchmark: physics steps/s and GPU memory vs nworld, nconmax, solver iterations,
object convex-part count and hand DoF. One-axis-at-a-time sweeps around a base point.

Each point runs in a fresh subprocess so the mempool high-water mark is per point.

  python scripts/bench.py                       # default sweeps, writes results/bench_<gpu>.{json,md}
  python scripts/bench.py --nworld 256 1024 4096 32768
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = dict(nworld=4096, iterations=10, nconmax=None, njmax=None, obj="box", n_parts=1, n_fingers=3, n_phalanges=2)


def run_point(p: dict, settle: int, nstep: int) -> dict:
  import numpy as np
  import warp as wp

  from mujoco_warp._src.types import OverflowType

  from mandahands.sim.batch import capture_step, contact_stats, make_batch
  from mandahands.sim.builder import SceneCfg, build_scene, close_ctrl

  cfg = SceneCfg(**{k: v for k, v in p.items() if k in {f.name for f in dataclasses.fields(SceneCfg)}})
  mjm = build_scene(cfg).compile()
  m, d = make_batch(mjm, p["nworld"], p["nconmax"], p["njmax"])
  # hold a closed grasp: the contact-heavy steady state the env spends most time in
  d.ctrl.assign(np.tile(close_ctrl(mjm, 0.7), (p["nworld"], 1)).astype(np.float32))
  graph = capture_step(m, d)
  for _ in range(settle):
    wp.capture_launch(graph)
  wp.synchronize()

  elapsed, peak = 0.0, {}
  for _ in range(nstep // 100):  # contact stats sampled between timed chunks
    t = time.perf_counter()
    for _ in range(100):
      wp.capture_launch(graph)
    wp.synchronize()
    elapsed += time.perf_counter() - t
    s = contact_stats(d)
    peak = {k: (v | peak.get(k, 0)) if k == "overflow_mask" else max(v, peak.get(k, v)) for k, v in s.items()}
  return {
    **p, **peak, "nq": mjm.nq, "nu": mjm.nu,
    "sps": p["nworld"] * (nstep // 100) * 100 / elapsed,
    "mem_mib": wp.get_mempool_used_mem_high(wp.get_device()) / 2**20,
    "overflow": OverflowType(peak["overflow_mask"]).name or "-",
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--nworld", type=int, nargs="*", default=[256, 1024, 4096, 16384])
  ap.add_argument("--iterations", type=int, nargs="*", default=[5, 10, 20])
  ap.add_argument("--nconmax", type=int, nargs="*", default=[16, 32, 64])
  ap.add_argument("--n_parts", type=int, nargs="*", default=[1, 4, 16])
  ap.add_argument("--dof", nargs="*", default=["3x2", "4x3", "5x4"], help="fingers x phalanges")
  ap.add_argument("--njmax", type=int, default=None, help="per-world constraint rows (MJWarp default if unset)")
  ap.add_argument("--settle", type=int, default=300)
  ap.add_argument("--nstep", type=int, default=500)
  ap.add_argument("--out", default="results")
  ap.add_argument("--_point", help=argparse.SUPPRESS)
  a = ap.parse_args()

  if a._point:
    print(json.dumps(run_point(json.loads(a._point), a.settle, a.nstep)))
    return

  base = {**BASE, "njmax": a.njmax}
  points = [("nworld", {**base, "nworld": n}) for n in a.nworld]
  points += [("iterations", {**base, "iterations": n}) for n in a.iterations]
  points += [("nconmax", {**base, "nconmax": n}) for n in a.nconmax]
  points += [("n_parts", {**base, "obj": "mesh", "n_parts": n}) for n in a.n_parts]
  for s in a.dof:
    f, ph = map(int, s.split("x"))
    points.append(("dof", {**base, "n_fingers": f, "n_phalanges": ph}))

  import warp as wp

  gpu = wp.get_device("cuda:0").name.replace(" ", "_")
  rows = []
  for axis, p in points:
    cmd = [sys.executable, __file__, "--_point", json.dumps(p), "--settle", str(a.settle), "--nstep", str(a.nstep)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
      row = {**p, "error": r.stderr.strip().splitlines()[-1]}
    else:
      row = json.loads(r.stdout.strip().splitlines()[-1])
    row["axis"] = axis
    rows.append(row)
    print(axis, {k: row.get(k) for k in ("nworld", "iterations", "nconmax", "n_parts", "nu", "sps", "mem_mib", "ncon_world_max", "nefc_world_max", "njmax", "overflow", "error")}, flush=True)

  out = Path(a.out)
  out.mkdir(exist_ok=True)
  (out / f"bench_{gpu}.json").write_text(json.dumps(rows, indent=1))
  cols = ["axis", "nworld", "iterations", "nconmax", "n_parts", "nu", "sps", "mem_mib", "ncon_world_max", "naconmax", "nefc_world_max", "njmax", "overflow", "capacity_overflow_worlds"]
  lines = [f"# Benchmark on {gpu}\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
  for r in rows:
    if "error" in r:
      lines.append(f"| {r['axis']} | {r['nworld']} | ERROR: {r['error']} |")
      continue
    fmt = {"sps": f"{r['sps']:,.0f}", "mem_mib": f"{r['mem_mib']:.0f}"}
    lines.append("| " + " | ".join(fmt.get(c, str(r.get(c))) for c in cols) + " |")
  (out / f"bench_{gpu}.md").write_text("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
