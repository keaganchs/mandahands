"""Objects as unions of convex parts: MuJoCo primitives or convex-hull meshes (MuJoCo collides a mesh as its hull).

A part is a dict: type ("box" | "sphere" | "capsule" | "cylinder" | "ellipsoid" | "hull"), size (MuJoCo
convention: half extents, radius, [radius, half-length], ...), pos, quat (object frame), verts (hull only).
Object frame: origin at the bounding-box centre, z up in the object's natural resting orientation.

`inside()` is exact per part, so the union gives occupancy (for the shape autoencoder, M9), volume (masses of
overlapping parts), and the surface filtering used for the decomposition error.
"""

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

TYPES = ("box", "sphere", "capsule", "cylinder", "ellipsoid", "hull")
_GEOM = dict(box=mujoco.mjtGeom.mjGEOM_BOX, sphere=mujoco.mjtGeom.mjGEOM_SPHERE, capsule=mujoco.mjtGeom.mjGEOM_CAPSULE,
             cylinder=mujoco.mjtGeom.mjGEOM_CYLINDER, ellipsoid=mujoco.mjtGeom.mjGEOM_ELLIPSOID, hull=mujoco.mjtGeom.mjGEOM_MESH)


def part(type, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), verts=None) -> dict:
  assert type in TYPES
  p = dict(type=type, size=[float(x) for x in np.resize(size, 3)], pos=[float(x) for x in pos], quat=[float(x) for x in quat])
  if verts is not None:
    p["verts"] = np.asarray(verts, float).tolist()
  return p


def _rot(q):
  m = np.zeros(9)
  mujoco.mju_quat2Mat(m, np.asarray(q, float))
  return m.reshape(3, 3)


def inside(p: dict, pts: np.ndarray, tol: float = 0.0) -> np.ndarray:
  """Points (N, 3) in the object frame -> bool (N,): inside the part (grown by tol)."""
  x = (pts - p["pos"]) @ _rot(p["quat"])  # into the part frame
  s = np.asarray(p["size"])
  t = p["type"]
  if t == "box":
    return (np.abs(x) <= s + tol).all(1)
  if t == "sphere":
    return np.linalg.norm(x, axis=1) <= s[0] + tol
  if t == "capsule":
    z = np.clip(x[:, 2], -s[1], s[1])
    return np.linalg.norm(x - np.c_[np.zeros((len(x), 2)), z], axis=1) <= s[0] + tol
  if t == "cylinder":
    return (np.linalg.norm(x[:, :2], axis=1) <= s[0] + tol) & (np.abs(x[:, 2]) <= s[1] + tol)
  if t == "ellipsoid":
    return ((x / (s + tol)) ** 2).sum(1) <= 1
  eq = ConvexHull(np.asarray(p["verts"])).equations  # unit normals: n.x + d <= 0 inside
  step = max(1, 20_000_000 // len(eq))  # chunked: points x facets can exceed RAM (YCB hulls reach ~77k facets)
  return np.concatenate([(x[i : i + step] @ eq[:, :3].T + eq[:, 3] <= tol).all(1) for i in range(0, len(x), step)])


def occupancy(parts: list[dict], pts: np.ndarray) -> np.ndarray:
  return np.any([inside(p, pts) for p in parts], axis=0)


def bbox(parts: list[dict], n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
  """Axis-aligned bounds from surface samples of every part."""
  pts = np.concatenate([surface(p, n // len(parts) + 50, np.random.default_rng(0)) for p in parts])
  return pts.min(0), pts.max(0)


def volume(parts: list[dict], n: int = 200_000, seed: int = 0) -> float:
  """Union volume by Monte Carlo in the bounding box (overlapping parts counted once)."""
  lo, hi = bbox(parts)
  pts = np.random.default_rng(seed).uniform(lo, hi, (n, 3))
  return float(occupancy(parts, pts).mean() * np.prod(hi - lo))


def surface(p: dict, n: int, rng) -> np.ndarray:
  """n points on the part's surface, object frame."""
  import trimesh

  s = np.asarray(p["size"])
  t = p["type"]
  if t == "hull":
    m = trimesh.convex.convex_hull(np.asarray(p["verts"]))
  elif t == "box":
    m = trimesh.creation.box(2 * s)
  elif t == "sphere":
    m = trimesh.creation.icosphere(3, s[0])
  elif t == "capsule":
    m = trimesh.creation.capsule(2 * s[1], s[0])
  elif t == "cylinder":
    m = trimesh.creation.cylinder(s[0], 2 * s[1])
  else:  # ellipsoid
    m = trimesh.creation.icosphere(3, 1.0)
    m.apply_scale(s)
  pts, _ = trimesh.sample.sample_surface(m, n, seed=int(rng.integers(1 << 31)))
  return pts @ _rot(p["quat"]).T + p["pos"]


def union_surface(parts: list[dict], n: int, seed: int = 0) -> np.ndarray:
  """Points on the boundary of the union: part surface samples not strictly inside another part."""
  rng = np.random.default_rng(seed)
  out = []
  for i, p in enumerate(parts):
    pts = surface(p, n, rng)
    others = [q for j, q in enumerate(parts) if j != i]
    if others:
      # internal cut faces between neighbouring parts lie on both boundaries: anything within 1 mm of another part
      # is interior (dropping a sliver of true surface near seams is harmless)
      pts = pts[~np.any([inside(q, pts, 1e-3) for q in others], axis=0)]
    out.append(pts)
  return np.concatenate(out)


def add_to_body(spec: mujoco.MjSpec, body, parts: list[dict], density: float, prefix: str = "obj", **geom_kw):
  """Geoms for the parts. Densities are scaled so the total mass is density x union volume (MuJoCo would count
  overlaps twice). ponytail: uniform scaling, so overlaps still shift the inertia slightly."""
  vols = []
  for p in parts:
    s = np.asarray(p["size"])
    t = p["type"]
    vols.append(dict(box=lambda: 8 * s.prod(), sphere=lambda: 4 / 3 * np.pi * s[0] ** 3,
                     capsule=lambda: np.pi * s[0] ** 2 * (2 * s[1] + 4 / 3 * s[0]), cylinder=lambda: 2 * np.pi * s[0] ** 2 * s[1],
                     ellipsoid=lambda: 4 / 3 * np.pi * s.prod(), hull=lambda: ConvexHull(np.asarray(p["verts"])).volume)[t]())
  scale = volume(parts) / sum(vols)
  for i, p in enumerate(parts):
    kw = dict(name=f"{prefix}_g{i}", type=_GEOM[p["type"]], pos=p["pos"], quat=p["quat"], density=density * scale, **geom_kw)
    if p["type"] == "hull":
      spec.add_mesh(name=f"{prefix}_m{i}", uservert=np.asarray(p["verts"]).ravel().tolist())
      kw["meshname"] = f"{prefix}_m{i}"
    else:
      kw["size"] = p["size"]
    body.add_geom(**kw)
