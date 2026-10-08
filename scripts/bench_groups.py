"""Batching benchmark (M5): physics steps/s for 1 / 10 / 100 topology-class groups and the superset baseline.

Every world holds a small sphere in a closed grasp (object on its support weld), the contact-heavy steady state.
Total worlds are fixed; groups split them evenly. Each group is one GENHAND topology class (train split) with up
to 4 variants. Control step = 10 physics steps (2 ms, 50 Hz), captured as one graph per group.

  python scripts/bench_groups.py --worlds 8400            # writes results/bench_groups_<gpu>.{json,md}

ponytail: one benchmark process; peak memory is the mempool high-water mark of the whole run, reported once.
"""

import argparse
import collections
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIZE = 0.012  # sphere radius
NSUB = 10


def spec_to_scene(spec):
  from mandahands.sim.scripted import place, probe, start_state

  pr = probe(spec)
  c, cl = pr["candidates"][0]
  scene, *_ = place(spec, pr, c, "sphere", min(SIZE, 0.9 * cl))
  m = scene.compile()
  d = start_state(m, pr)
  d.ctrl[:] = pr["ctrl_close"]
  return scene, m, d


def class_groups(n_groups, max_variants, seed=0):
  """n_groups GENHAND topology classes (train split, >= 2 valid variants preferred), up to max_variants hands each."""
  from mandahands.embodiments import registry
  from mandahands.embodiments.procedural.generator import topology_key

  man = registry.manifest()
  by = collections.defaultdict(list)
  for h in man["hands"]:
    if h.get("split") == "train":
      by[topology_key(man["topologies"][h["t"]])].append(h["id"])
  keys = sorted(by, key=lambda k: (-min(len(by[k]), max_variants), k))
  rng = np.random.default_rng(seed)
  pool = [k for k in keys if len(by[k]) >= 2][: 3 * n_groups]
  pick = list(rng.choice(pool, n_groups, replace=False)) if len(pool) >= n_groups else keys[:n_groups]
  return [by[k][:max_variants] for k in pick]


def class_scenes(ids):
  """Worker: one class's hands -> union of pairs -> (MjModel, start MjData) per variant. No CUDA here."""
  from mandahands.embodiments import registry
  from mandahands.sim.groups import union_pairs

  hands = [registry.genhand(i)[0] for i in ids]
  union_pairs(hands)
  return [spec_to_scene(h)[1:] for h in hands]


def build_groups(classes, worlds_per_group):
  """Scenes per class (parallel) -> Groups, split by MJWarp-unbatchable signature (largest subset benchmarked)."""
  from mandahands.sim.groups import Group, signature

  with ProcessPoolExecutor(os.cpu_count()) as ex:
    per_class = list(ex.map(class_scenes, classes))
  groups, splits = [], 0
  for scenes in per_class:
    by_sig = collections.defaultdict(list)
    for m, d in scenes:
      by_sig[signature(m)].append((m, d))
    splits += len(by_sig) - 1
    sub = max(by_sig.values(), key=len)  # ponytail: benchmark the largest batchable subset of the class
    V = max(v for v in (4, 3, 2, 1) if v <= len(sub) and worlds_per_group % v == 0)
    groups.append(Group([s[0] for s in sub[:V]], [s[1] for s in sub[:V]], worlds_per_group, njmax=NJMAX))
  return groups, splits


def superset_group(nworld, n_variants=4, seed=0):
  """One maximal hand; per variant a random subset of fingers is locked (zero-width range) and collision-free
  (geom and pair impedance at the floor, so their contacts exert no force). Only hands that are sub-hands of the superset can be represented."""
  import mujoco

  from mandahands.embodiments.procedural.generator import build, sample_genome
  from mandahands.sim.builder import add_self_collision
  from mandahands.sim.groups import OFF_SOLIMP, Group

  topo = dict(layout="anthro", n_fingers=5, n_phalanges=3, coupling="full", tip="capsule", thumb="opposed",
              thumb_phalanges=3, abduction="all", abd_order="abd_first")
  spec, meta = build(topo, sample_genome(topo, 0, 0, 0))
  add_self_collision(spec)
  scene, m0, d0 = spec_to_scene(spec)
  rng = np.random.default_rng(seed)
  ms, ds = [], []
  for v in range(n_variants):
    keep = set(rng.choice(5, rng.integers(2, 6), replace=False).tolist()) | {4}  # thumb always kept
    m = scene.compile()
    for j in range(m.njnt):
      name = m.joint(j).name
      if name.startswith("f") and int(name[1]) not in keep:
        m.jnt_range[j] = [0, 0]
        m.jnt_limited[j] = 1
    for g in range(m.ngeom):
      n = m.geom(g).name
      if n.startswith("f") and n[1].isdigit() and int(n[1]) not in keep:
        m.geom_solimp[g] = OFF_SOLIMP
    for p in range(m.npair):
      names = [m.geom(m.pair_geom1[p]).name, m.geom(m.pair_geom2[p]).name]
      if any(n.startswith("f") and n[1].isdigit() and int(n[1]) not in keep for n in names):
        m.pair_solimp[p] = OFF_SOLIMP
    d = mujoco.MjData(m)
    d.qpos[:], d.ctrl[:], d.eq_active[:] = d0.qpos, d0.ctrl, d0.eq_active
    d.mocap_pos[:], d.mocap_quat[:] = d0.mocap_pos, d0.mocap_quat
    ms.append(m)
    ds.append(d)
  return Group(ms, ds, nworld, njmax=NJMAX), m0


def time_multi(groups, streams, settle=30, nctrl=60):
  import warp as wp

  from mandahands.sim.groups import MultiGroup

  mg = MultiGroup(groups, NSUB, streams)
  for _ in range(settle):
    mg.step()
  wp.synchronize()
  t = time.perf_counter()
  for _ in range(nctrl):
    mg.step()
    wp.synchronize()  # the policy reads the state every control step
  dt = time.perf_counter() - t
  stats = [g.stats() for g in groups]
  return dict(sps=mg.nworld * nctrl * NSUB / dt, ctrl_hz_per_world=nctrl / dt,
              capacity_overflow_worlds=sum(s["capacity_overflow_worlds"] for s in stats),
              ncon_world_max=max(s["ncon_world_max"] for s in stats), nefc_world_max=max(s["nefc_world_max"] for s in stats))


NJMAX = 400


def main():
  global NJMAX
  ap = argparse.ArgumentParser()
  ap.add_argument("--worlds", type=int, default=8400)
  ap.add_argument("--njmax", type=int, default=400)
  ap.add_argument("--out", default="results")
  a = ap.parse_args()
  NJMAX = a.njmax
  import warp as wp

  wp.config.quiet = True
  gpu = wp.get_device().name
  rows = []

  pool = class_groups(100, 4)
  # G = 1: ten classes, each alone with all worlds
  singles = []
  for ids in pool[:10]:
    gs, _ = build_groups([ids], a.worlds)
    r = time_multi(gs, False)
    singles.append(r["sps"])
    del gs
  rows.append(dict(config="1 group (mean of 10 classes, each alone)", groups=1, worlds=a.worlds,
                   sps=float(len(singles) / np.sum(1 / np.array(singles))), sps_min=min(singles), sps_max=max(singles)))
  for G in (10, 100):
    gs, splits = build_groups(pool[:G], a.worlds // G)
    for streams in (False, True):
      r = time_multi(gs, streams)
      rows.append(dict(config=f"{G} groups, {'one stream each' if streams else 'sequential'}", groups=G, worlds=a.worlds,
                       class_splits=splits, **r))
      print(rows[-1], flush=True)
    del gs
  g, m0 = superset_group(a.worlds)
  r = time_multi([g], False)
  rows.append(dict(config=f"superset (5 fingers x 3 + abduction, {m0.nv - 13} hand DoF, random finger subsets)", groups=1,
                   worlds=a.worlds, **r))
  mem = wp.get_mempool_used_mem_high(wp.get_device()) / 2**20
  out = Path(a.out)
  out.mkdir(exist_ok=True)
  tag = gpu.replace(" ", "_")
  (out / f"bench_groups_{tag}.json").write_text(json.dumps(dict(gpu=gpu, rows=rows, mem_high_mib=mem), indent=1))
  L = [f"# Batching benchmark, {gpu}", "", f"{a.worlds} worlds total, closed grasp on a {1000 * SIZE:.0f} mm sphere, "
       f"control step = {NSUB} x 2 ms, synchronized every control step. Peak pool memory {mem:.0f} MiB.", "",
       "| config | physics steps/s | control steps/s per world | max contacts / rows per world | capacity overflow |", "|---|---|---|---|---|"]
  for r in rows:
    extra = f" (range {r['sps_min'] / 1e6:.2f}-{r['sps_max'] / 1e6:.2f} M)" if "sps_min" in r else ""
    L.append(f"| {r['config']} | {r['sps'] / 1e6:.2f} M{extra} | {r.get('ctrl_hz_per_world', r['sps'] / r['worlds'] / NSUB):.0f} | "
             f"{r.get('ncon_world_max', '-')} / {r.get('nefc_world_max', '-')} | {r.get('capacity_overflow_worlds', '-')} |")
  (out / f"bench_groups_{tag}.md").write_text("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  main()
