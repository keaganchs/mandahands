"""Batching strategy A (user decision, M5): a few topology-class groups resident on the GPU, resampled every rotation.

Each rotation picks `n_groups` topology classes uniformly over classes (balanced by class, not by hand count),
up to `variants` hands per class, and optionally real hands (each its own group). Real hands are held out of
training by default (spec); `include_real=True` mixes them in at `real_fraction` of the groups.

Scenes are built once per hand and cached (per process), so a rotation only rebuilds the MJWarp groups.
Tendon coefficients are randomized per group at every rotation (`groups.tendon_dr`).

ponytail: the scene is still the M5 placeholder (sphere held at the hand's pinch point); M6/M7 replace
`scene_fn` with object sampling and episode resets.
"""

import collections
import copy
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mandahands.embodiments import registry
from mandahands.embodiments.procedural.generator import topology_key


@dataclass
class RotationCfg:
  n_groups: int = 16
  worlds: int = 16384
  variants: int = 4
  include_real: bool = False  # real hands are evaluation-only unless set
  real_fraction: float = 0.25  # share of groups that are real hands when include_real
  tendon_dr: float = 0.2  # per-group relative tendon coefficient randomization
  split: str = "train"
  dataset: str = "genhand-1k-v4"
  njmax: int = 400
  seed: int = 0
  real_ids: list = field(default_factory=list)  # default: every extracted real hand


def classes(cfg: RotationCfg) -> dict[str, list[str]]:
  man = registry.manifest(cfg.dataset)
  ids = set(registry.ids(cfg.split, cfg.dataset))
  by = collections.defaultdict(list)
  for h in man["hands"]:
    if h["id"] in ids:
      by[topology_key(man["topologies"][h["t"]])].append(h["id"])
  return dict(by)


def pick(cfg: RotationCfg, rotation: int) -> list[list[str]]:
  """Hand ids per group for one rotation (deterministic in seed and rotation)."""
  rng = np.random.default_rng([cfg.seed, rotation])
  by = classes(cfg)
  real = cfg.real_ids or registry.real_ids()
  n_real = round(cfg.real_fraction * cfg.n_groups) if cfg.include_real else 0
  keys = sorted(by)
  chosen = rng.choice(len(keys), min(cfg.n_groups - n_real, len(keys)), replace=False)
  groups = []
  for k in chosen:
    members = by[keys[k]]
    groups.append([str(x) for x in rng.choice(members, min(cfg.variants, len(members)), replace=False)])
  if n_real:
    groups += [[str(h)] for h in rng.choice(real, min(n_real, len(real)), replace=False)]
  return groups


def scenes(args):
  """Worker: hand ids of one group -> [(id, (MjModel, start MjData))]. GENHAND hands get the union of their whole
  class's pairs (built once per call), so any subset of the class can share a group."""
  ids, cfg = args
  from mandahands.sim.groups import union_pairs
  from mandahands.sim.scripted import place, probe, start_state

  if ids[0].startswith("gh-"):
    members = next(v for v in classes(cfg).values() if ids[0] in v)
    hands = {i: registry.genhand(i, cfg.dataset)[0] for i in members}
    union_pairs(list(hands.values()))
  else:
    hands = {i: registry.real(i)[0] for i in ids}
  out = []
  for hid in ids:
    spec = hands[hid]
    pr = probe(spec)
    c, cl = pr["candidates"][0]
    sc, *_ = place(spec, pr, c, "sphere", min(0.012, 0.9 * cl))
    m = sc.compile()
    d = start_state(m, pr)
    d.ctrl[:] = pr["ctrl_close"]
    out.append((hid, (m, d)))
  return out


class Rotation:
  """disk_cache: directory for pickled per-hand scenes (shared across processes), or None."""

  def __init__(self, cfg: RotationCfg, workers: int | None = None, disk_cache: str | None = None):
    self.cfg, self.workers, self.cache = cfg, workers, {}
    self.disk = Path(disk_cache) / cfg.dataset if disk_cache else None

  def _load(self, ids):
    if self.disk:
      for h in ids:
        f = self.disk / f"{h}.pkl"
        if h not in self.cache and f.exists():
          self.cache[h] = pickle.loads(f.read_bytes())

  def _save(self, res):
    if self.disk:
      self.disk.mkdir(parents=True, exist_ok=True)
      for h, v in res:
        (self.disk / f"{h}.pkl").write_bytes(pickle.dumps(v))

  def build(self, rotation: int, streams: bool = True, nsubstep: int = 10):
    """-> (MultiGroup, hand ids per group). Builds missing scenes in parallel, then the GPU groups."""
    from mandahands.sim.groups import Group, MultiGroup, tendon_dr

    ids = pick(self.cfg, rotation)
    self._load([h for g in ids for h in g])
    todo = [[h for h in g if h not in self.cache] for g in ids]
    todo = [t for t in todo if t]
    if todo:
      with ProcessPoolExecutor(self.workers) as ex:
        for res in ex.map(scenes, [(t, self.cfg) for t in todo]):
          self.cache.update(res)
          self._save(res)
    rng = np.random.default_rng([self.cfg.seed, rotation, 1])
    per = self.cfg.worlds // len(ids)
    groups = []
    for g in ids:
      ms = [copy.deepcopy(self.cache[h][0]) for h in g]  # tendon DR edits models in place
      ds = [copy.deepcopy(self.cache[h][1]) for h in g]
      tendon_dr(ms, ds, rng, self.cfg.tendon_dr)
      groups.append(Group(ms, ds, per - per % len(g), njmax=self.cfg.njmax))
    return MultiGroup(groups, nsubstep, streams), ids
