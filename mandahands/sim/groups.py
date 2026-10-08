"""Heterogeneous batching: groups of structurally identical scenes stepped as one MJWarp batch each.

A Group is one compiled model whose per-world parameters come from several variants (scenes with the same
bodies/joints/actuators/geom types but different geometry and dynamics). MJWarp indexes batched model fields as
field[worldid % batch], so worlds are laid out round-robin over the V variants and every field that differs
between variants gets a batch of V. Fields MJWarp cannot batch must be equal across a group; `signature()`
groups variants by them (e.g. tendon coefficients, `wrap_prm`, split tendon-coupled classes).

Self-collision pairs differ per variant (unreachable pairs are pruned per hand), so `union_pairs` gives every
variant of a class the union of their pairs, with the ones a variant does not own disabled by a large negative
pair margin (pair_margin is batchable).

MultiGroup steps several groups, one captured CUDA graph each.
"""

import dataclasses
import hashlib

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src import types
from mujoco_warp._src.warp_util import is_array_spec

from mandahands.sim.batch import capture_step, contact_stats

BATCHABLE = {f.name for f in dataclasses.fields(types.Model) if is_array_spec(f.type) and getattr(f.type, "shape", ("",))[0] == "*"}
UNUSED = {"dof_length"}  # differs between variants, but only read by sleeping, which we don't use
PAIR_OFF = -1e3  # pair margin that never produces a contact


def _fields(mw) -> dict:
  out = {}
  for f in dataclasses.fields(mw):
    v = getattr(mw, f.name)
    out[f.name] = v.numpy() if isinstance(v, wp.array) else v
  return out


def signature(mjm: mujoco.MjModel) -> str:
  """Hash of everything MJWarp cannot vary per world; scenes with equal signatures can share a group."""
  mw = mjw.put_model(mjm)
  h = hashlib.sha1()
  for name, v in sorted(_fields(mw).items()):
    if name in BATCHABLE or name in UNUSED:
      continue
    if isinstance(v, np.ndarray):
      h.update(name.encode() + str(v.shape).encode() + np.ascontiguousarray(v).tobytes())
    elif isinstance(v, (int, float, bool, str, tuple)):
      h.update(f"{name}={v}".encode())
  return h.hexdigest()[:16]


def union_pairs(hands: list[mujoco.MjSpec]) -> None:
  """In place: every hand gets the union of the hands' explicit pairs (same order); foreign pairs disabled."""
  key = lambda p: tuple(sorted((p.geomname1, p.geomname2)))
  own = [{key(p): p for p in h.pairs} for h in hands]
  union = sorted(set().union(*own))
  template = next(p for o in own for p in o.values()) if union else None
  for h, o in zip(hands, own):
    for p in list(h.pairs):
      h.delete(p)
    for k in union:
      src = o.get(k, template)
      h.add_pair(geomname1=k[0], geomname2=k[1], condim=src.condim, solref=list(src.solref), solimp=list(src.solimp),
                 margin=0.0 if k in o else PAIR_OFF)


class Group:
  """One MJWarp batch over V structurally identical scene variants; world w runs variant w % V."""

  def __init__(self, models: list[mujoco.MjModel], datas: list[mujoco.MjData], nworld: int,
               nconmax: int | None = None, njmax: int | None = None):
    sigs = {signature(m) for m in models}
    if len(sigs) != 1:
      raise ValueError(f"variants differ in fields MJWarp cannot batch ({len(sigs)} signatures); split with signature()")
    V = len(models)
    if nworld % V:
      raise ValueError(f"nworld {nworld} must be a multiple of the number of variants {V}")
    self.V, self.nworld, self.mjm = V, nworld, models[0]
    per = [_fields(mjw.put_model(m)) for m in models]
    self.batched = sorted(f for f in BATCHABLE if f in per[0] and isinstance(per[0][f], np.ndarray)
                          and any(not np.array_equal(p[f], per[0][f]) for p in per[1:]))
    self.m = mjw.put_model(models[0], batch_sizes={f: V for f in self.batched})
    for f in self.batched:
      ref = getattr(self.m, f)
      setattr(self.m, f, wp.array(np.concatenate([p[f] for p in per]), dtype=ref.dtype, device=ref.device))
    self.d = mjw.make_data(models[0], nworld=nworld, nconmax=nconmax, njmax=njmax)
    idx = np.arange(nworld) % V
    for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "eq_active"):
      arr = getattr(self.d, name)
      arr.assign(np.stack([getattr(datas[i], name) for i in idx]).astype(arr.numpy().dtype))
    self.graph = None

  def capture(self, nsubstep: int = 1, stream=None):
    self.stream = stream
    self.graph = capture_step(self.m, self.d, nsubstep, stream)
    return self.graph

  def stats(self) -> dict:
    return contact_stats(self.d)


class MultiGroup:
  """streams: launch each group's graph on its own CUDA stream so small groups can overlap on the GPU."""

  def __init__(self, groups: list[Group], nsubstep: int = 1, streams: bool = False):
    self.groups = groups
    for g in groups:
      g.capture(nsubstep, wp.Stream() if streams else None)

  def step(self):
    for g in self.groups:
      wp.capture_launch(g.graph, stream=g.stream)

  @property
  def nworld(self) -> int:
    return sum(g.nworld for g in self.groups)
