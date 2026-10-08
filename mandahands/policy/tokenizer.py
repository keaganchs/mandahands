"""Descriptors + state -> per-joint and per-actuator tokens, padded so hands with different DoF share a batch.

Observation tokens are per joint, action tokens per actuator (spec, Component 2). Each world indexes one
embodiment of a TokenTable; padded slots are zero and masked (mask True = real token). The coupling matrix
(A x J, closing-sign convention) is passed alongside for attention bias or incidence input.

Sign convention (see embodiments/descriptors.py): canonical coordinate = sign x MuJoCo coordinate, so for every
joint and actuator positive means closing. Actions are absolute position targets in [-1, 1]: -1 = the open
command, +1 = the closed command of each actuator.
"""

import numpy as np
import torch

from mandahands.embodiments.descriptors import ACT_FIELDS, JOINT_FIELDS, Descriptor, Normalizer

JOINT_STATE = ["q", "qd", "touch", "touch_mask"]  # q normalized to [-1, 1] over the range; qd in ranges/s
ACT_STATE = ["prev_action", "error", "force"]  # error = (command - actuator length) / ctrl span; force / closing limit

_J = {f: i for i, f in enumerate(JOINT_FIELDS)}
_A = {f: i for i, f in enumerate(ACT_FIELDS)}


def _pad(arrs: list[np.ndarray], n: int, fill=0.0) -> np.ndarray:
  out = np.full((len(arrs), n, *arrs[0].shape[1:]), fill, dtype=np.float32)
  for i, a in enumerate(arrs):
    out[i, : len(a)] = a
  return out


class TokenTable:
  """Static tokens of a set of embodiments, padded to the largest J and A."""

  def __init__(self, descs: list[Descriptor], norm: Normalizer, device="cpu"):
    self.J = max(len(d.joint) for d in descs)
    self.A = max(len(d.act) for d in descs)
    t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)
    self.joint_static = t(_pad([norm.joint(d.joint) for d in descs], self.J))
    self.act_static = t(_pad([norm.act(d.act) for d in descs], self.A))
    self.joint_mask = t(_pad([np.ones(len(d.joint)) for d in descs], self.J), torch.bool)
    self.act_mask = t(_pad([np.ones(len(d.act)) for d in descs], self.A), torch.bool)
    C = np.zeros((len(descs), self.A, self.J), np.float32)
    for i, d in enumerate(descs):
      C[i, : d.coupling.shape[0], : d.coupling.shape[1]] = d.coupling
    self.coupling = t(C)
    self.joint_sign = t(_pad([d.joint_sign for d in descs], self.J))
    self.act_sign = t(_pad([d.act_sign for d in descs], self.A))
    # raw (unnormalized) quantities needed for the dynamic state; padding gets harmless values
    self.lo = t(_pad([d.joint[:, _J["lo"]] for d in descs], self.J, -1.0))
    self.hi = t(_pad([d.joint[:, _J["hi"]] for d in descs], self.J, 1.0))
    self.ctrl_open = t(_pad([d.ctrl_open for d in descs], self.A))
    self.ctrl_close = t(_pad([d.ctrl_close for d in descs], self.A, 1.0))
    self.span = t(_pad([d.act[:, _A["ctrl_span"]] for d in descs], self.A, 1.0)).clamp_min(1e-6)
    self.f_close = t(_pad([d.act[:, _A["force_close"]] for d in descs], self.A, 1.0)).clamp_min(1e-6)

  def static(self, emb: torch.Tensor) -> dict:
    """Per-world static tokens; emb is a (B,) long tensor of embodiment indices."""
    return dict(joint=self.joint_static[emb], act=self.act_static[emb], joint_mask=self.joint_mask[emb],
                act_mask=self.act_mask[emb], coupling=self.coupling[emb])

  def joint_state(self, emb, q, qd, touch=None) -> torch.Tensor:
    """q, qd: (B, J) in each embodiment's joint order (MuJoCo coordinates, padded with anything). -> (B, J, 4)."""
    s, lo, hi, mask = self.joint_sign[emb], self.lo[emb], self.hi[emb], self.joint_mask[emb]
    rng = (hi - lo).clamp_min(1e-6)
    qn = 2 * (s * q - lo) / rng - 1
    qdn = s * qd / rng
    has = torch.zeros_like(q) if touch is None else torch.ones_like(q)
    touch = torch.zeros_like(q) if touch is None else touch
    out = torch.stack([qn, qdn, touch, has], -1)
    return out * mask[..., None]

  def act_state(self, emb, prev_action, ctrl, length, force) -> torch.Tensor:
    """prev_action in [-1, 1]; ctrl, actuator length and force in MuJoCo units, (B, A). -> (B, A, 3)."""
    s, mask = self.act_sign[emb], self.act_mask[emb]
    out = torch.stack([prev_action, s * (ctrl - length) / self.span[emb], s * force / self.f_close[emb]], -1)
    return out * mask[..., None]

  def to_ctrl(self, emb, action: torch.Tensor) -> torch.Tensor:
    """Action in [-1, 1] (-1 open, +1 closed) -> MuJoCo ctrl. Padded actuators give 0."""
    o, c = self.ctrl_open[emb], self.ctrl_close[emb]
    return (o + (action.clamp(-1, 1) + 1) / 2 * (c - o)) * self.act_mask[emb]

  def to_action(self, emb, ctrl: torch.Tensor) -> torch.Tensor:
    o, c = self.ctrl_open[emb], self.ctrl_close[emb]
    den = torch.where((c - o).abs() < 1e-9, torch.ones_like(c), c - o)
    return (2 * (ctrl - o) / den - 1) * self.act_mask[emb]
