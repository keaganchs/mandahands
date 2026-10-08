"""Strategy A (rotating resident groups) vs C (all topology classes resident), physics steps/s and memory.

  python scripts/bench_rotation.py --worlds 8400            # 3060
  python scripts/bench_rotation.py --worlds 16800 50400     # A5000 (24 GB)

Each point runs in its own process (peak memory per point). Scenes are cached in data/cache/scenes/.
Writes results/bench_rotation_<gpu>.{json,md}.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def point(mode, worlds, groups_a):
  import warp as wp

  from bench_groups import time_multi
  from mandahands.sim.rotation import Rotation, RotationCfg, classes

  wp.config.quiet = True
  cfg = RotationCfg(worlds=worlds, n_groups=groups_a if mode == "A" else 10**6)
  rot = Rotation(cfg, disk_cache="data/cache/scenes")
  t = time.perf_counter()
  mg, ids = rot.build(0)
  build = time.perf_counter() - t
  r = time_multi(mg.groups, streams=True)
  out = dict(mode=mode, worlds=worlds, groups=len(ids), n_classes=len(classes(cfg)), build_s=build, **r)
  if mode == "A":  # cost of one rotation: drop the groups, build the next set
    del mg
    t = time.perf_counter()
    mg, _ = rot.build(1)
    out["rotate_s"] = time.perf_counter() - t
  out["mem_mib"] = wp.get_mempool_used_mem_high(wp.get_device()) / 2**20
  out["gpu"] = wp.get_device().name
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--worlds", type=int, nargs="*", default=[8400])
  ap.add_argument("--groups-a", type=int, default=16)
  ap.add_argument("--out", default="results")
  ap.add_argument("--_point", help=argparse.SUPPRESS)
  a = ap.parse_args()
  if a._point:
    print("RESULT " + json.dumps(point(*json.loads(a._point))))
    return
  rows = []
  for w in a.worlds:
    for mode in ("A", "C"):
      p = subprocess.run([sys.executable, __file__, "--_point", json.dumps([mode, w, a.groups_a])], capture_output=True, text=True)
      line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
      if line is None:
        rows.append(dict(mode=mode, worlds=w, error=(p.stderr or p.stdout)[-400:]))
      else:
        rows.append(json.loads(line[7:]))
      print(rows[-1], flush=True)
  gpu = next((r["gpu"] for r in rows if "gpu" in r), "unknown")
  out = Path(a.out)
  out.mkdir(exist_ok=True)
  tag = gpu.replace(" ", "_")
  (out / f"bench_rotation_{tag}.json").write_text(json.dumps(rows, indent=1))
  L = [f"# Strategy A vs C, {gpu}", "", "A: rotating subset of topology-class groups (one CUDA stream each). "
       "C: every topology class resident. Closed grasp on a 12 mm sphere, control step 10 x 2 ms, synchronized.", "",
       "| strategy | worlds | groups | physics steps/s | peak MiB | build s | rotate s | capacity overflow |", "|---|---|---|---|---|---|---|---|"]
  for r in rows:
    if "error" in r:
      L.append(f"| {r['mode']} | {r['worlds']} | - | failed: {r['error'][-120:].strip()} | | | | |")
      continue
    L.append(f"| {r['mode']} | {r['worlds']} | {r['groups']} | {r['sps'] / 1e6:.2f} M | {r['mem_mib']:.0f} | {r['build_s']:.0f} | "
             f"{r.get('rotate_s', float('nan')):.1f} | {r['capacity_overflow_worlds']} |")
  (out / f"bench_rotation_{tag}.md").write_text("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  main()
