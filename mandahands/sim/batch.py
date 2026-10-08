"""MJWarp batch creation, graph-captured stepping, contact-capacity accounting."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src.types import OverflowType

# solver hitting its iteration cap is a convergence signal, not lost contacts/constraints
_ITER = int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)


class ContactOverflow(RuntimeError):
  pass


def make_batch(mjm: mujoco.MjModel, nworld: int, nconmax: int | None = None, njmax: int | None = None):
  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=nworld, nconmax=nconmax, njmax=njmax)
  return m, d


def capture_step(m, d, nsubstep: int = 1, stream=None):
  """Capture nsubstep physics steps (one control step) as a CUDA graph (on `stream` if given)."""
  with wp.ScopedStream(stream) if stream is not None else _null():
    mjw.step(m, d)  # first call compiles kernels outside the capture
    with wp.ScopedCapture(stream=stream) as cap:
      for _ in range(nsubstep):
        mjw.step(m, d)
  return cap.graph


class _null:
  def __enter__(self):
    return self

  def __exit__(self, *a):
    return False


def contact_stats(d) -> dict:
  """Per-world contact/constraint usage vs capacity. Syncs the device; call outside hot loops."""
  nacon = int(d.nacon.numpy()[0])
  ncon_world = np.bincount(d.contact.worldid.numpy()[:nacon], minlength=d.nworld)
  overflow = d.overflow.numpy()
  return dict(
    nacon=nacon,
    naconmax=d.naconmax,
    ncon_world_max=int(ncon_world.max(initial=0)),
    nefc_world_max=int(d.nefc.numpy().max()),
    njmax=d.njmax,
    capacity_overflow_worlds=int((overflow & ~_ITER != 0).sum()),
    iter_limit_worlds=int((overflow & _ITER != 0).sum()),
    overflow_mask=int(np.bitwise_or.reduce(overflow)),
  )


def check_overflow(stats: dict):
  """Fail loudly: MJWarp drops contacts beyond capacity, which shows up as fingers passing through objects."""
  if stats["capacity_overflow_worlds"]:
    raise ContactOverflow(f"{OverflowType(stats['overflow_mask'] & ~_ITER)!r} in {stats['capacity_overflow_worlds']} worlds: {stats}")
