"""Procedural objects: primitives and composites (mugs, bowls, pans, bottles, hand/garden tools) as convex parts.

Composites are built from convex primitives directly, so they need no decomposition and have no decomposition
error. Seeded per (seed, family, index); sizes target single-hand grasping (bimanual-size objects: backlog B3).
"""

import numpy as np

from mandahands.objects.parts import bbox, part

FAMILIES = ("box", "sphere", "cylinder", "capsule", "ellipsoid", "mug", "bowl", "pan", "bottle", "tool")


def _q(axis, angle):
  axis = np.asarray(axis, float) / np.linalg.norm(axis)
  return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


def _ring(r, h, t, z0, k, tilt=0.0):
  """Wall of k boxes around the z axis: radius r (outer), height h, thickness t, base at z0, tilted outward."""
  out = []
  for i in range(k):
    a = 2 * np.pi * i / k
    q = _q([0, 0, 1], a)
    if tilt:
      tq = _q([0, 1, 0], tilt)  # outward: the wall top leans away from the axis
      q = np.array([q[0] * tq[0] - q[1:] @ tq[1:], *(q[0] * tq[1:] + tq[0] * q[1:] + np.cross(q[1:], tq[1:]))])
    rr = r - t / 2 + np.sin(tilt) * h / 2
    out.append(part("box", [t / 2, np.pi * r / k * 1.08, h / 2], [rr * np.cos(a), rr * np.sin(a), z0 + h / 2 * np.cos(tilt)], q))
  return out


def make(family: str, rng: np.random.Generator) -> list[dict]:
  u = rng.uniform
  if family == "box":
    return [part("box", u(0.01, 0.05, 3))]
  if family == "sphere":
    return [part("sphere", [u(0.012, 0.045)])]
  if family == "cylinder":
    return [part("cylinder", [u(0.01, 0.045), u(0.01, 0.08)])]
  if family == "capsule":
    return [part("capsule", [u(0.008, 0.035), u(0.01, 0.07)])]
  if family == "ellipsoid":
    return [part("ellipsoid", u(0.012, 0.05, 3))]
  if family == "mug":
    r, h, t = u(0.03, 0.045), u(0.06, 0.11), u(0.003, 0.006)
    ps = [part("cylinder", [r, t / 2], [0, 0, t / 2])] + _ring(r, h, t, 0, 8)
    hr, ht = u(0.004, 0.007), u(0.015, 0.025)  # handle: C of three capsules on +x
    z0, z1 = h * 0.25, h * 0.8
    ps += [part("capsule", [hr, ht / 2], [r + ht / 2, 0, z0], _q([0, 1, 0], np.pi / 2)),
           part("capsule", [hr, ht / 2], [r + ht / 2, 0, z1], _q([0, 1, 0], np.pi / 2)),
           part("capsule", [hr, (z1 - z0) / 2], [r + ht, 0, (z0 + z1) / 2])]
    return ps
  if family == "bowl":
    r, h, t = u(0.05, 0.08), u(0.03, 0.05), u(0.003, 0.006)
    return [part("cylinder", [r * 0.55, t / 2], [0, 0, t / 2])] + _ring(r, h, t, 0, 10, tilt=0.5)
  if family == "pan":
    r, h, t = u(0.07, 0.12), u(0.02, 0.04), u(0.003, 0.006)
    hl, hr = u(0.05, 0.09), u(0.007, 0.012)
    return ([part("cylinder", [r, t / 2], [0, 0, t / 2])] + _ring(r, h, t, 0, 10)
            + [part("capsule", [hr, hl], [r + hl, 0, h * 0.8], _q([0, 1, 0], np.pi / 2 - 0.15))])
  if family == "bottle":
    r, h = u(0.022, 0.04), u(0.05, 0.1)
    nr, nh = r * u(0.3, 0.5), u(0.015, 0.03)
    return [part("cylinder", [r, h / 2], [0, 0, h / 2]),
            part("cylinder", [r * 0.7, r * 0.25], [0, 0, h + r * 0.25]),
            part("cylinder", [nr, nh / 2], [0, 0, h + r * 0.5 + nh / 2]),
            part("cylinder", [nr * 1.15, 0.005], [0, 0, h + r * 0.5 + nh + 0.005])]
  if family == "tool":  # handle along x, lying flat; head at +x
    hr, hl = u(0.009, 0.018), u(0.05, 0.1)
    ps = [part("capsule", [hr, hl], [0, 0, hr], _q([0, 1, 0], np.pi / 2))]
    kind = rng.integers(3)
    x = hl + hr
    if kind == 0:  # trowel / spatula: thin blade
      bl, bw, bt = u(0.03, 0.07), u(0.02, 0.04), u(0.0015, 0.004)
      ps.append(part("box", [bl, bw, bt], [x + bl, 0, hr]))
    elif kind == 1:  # hammer: head across the handle
      hw, hh = u(0.04, 0.07), u(0.01, 0.018)
      ps.append(part("box", [hh, hw, hh], [x + hh, 0, hr]))
    else:  # rake / fork: bar plus tines
      n, tl, tr = int(rng.integers(3, 6)), u(0.02, 0.04), u(0.002, 0.004)
      bw = u(0.025, 0.05)
      ps.append(part("box", [0.004, bw, 0.004], [x + 0.004, 0, hr]))
      ps += [part("capsule", [tr, tl / 2], [x + 0.008 + tl / 2, y, hr], _q([0, 1, 0], np.pi / 2)) for y in np.linspace(-bw, bw, n)]
    return ps
  raise ValueError(family)


def centred(parts: list[dict]) -> list[dict]:
  """Shift so the bounding-box centre is the origin."""
  lo, hi = bbox(parts)
  c = (lo + hi) / 2
  return [{**p, "pos": (np.asarray(p["pos"]) - c).tolist()} for p in parts]


def generate(seed: int, family: str, i: int) -> list[dict]:
  return centred(make(family, np.random.default_rng([seed, FAMILIES.index(family), i])))
