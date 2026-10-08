"""YCB objects: mesh -> CoACD convex parts (cached), decomposition error, category.

Meshes: data/sources/ycb/<obj>/google_16k/nontextured.ply (scripts/fetch_ycb.py), metres, z up.
"""

import hashlib
from pathlib import Path

import numpy as np

from mandahands.objects.parts import part, union_surface

SRC = Path("data/sources/ycb")
CACHE = Path("data/cache/coacd_objects")
COACD = dict(threshold=0.04, max_convex_hull=16, preprocess_resolution=50, seed=0)  # recorded in the library

# fine semantic categories (for held-out-category splits), by YCB object name
CATEGORY = {
  "can": ["master_chef_can", "tomato_soup_can", "tuna_fish_can", "potted_meat_can"],
  "food_box": ["cracker_box", "sugar_box", "pudding_box", "gelatin_box"],
  "bottle": ["mustard_bottle", "bleach_cleanser", "windex_bottle"],
  "fruit": ["banana", "strawberry", "apple", "lemon", "peach", "pear", "orange", "plum"],
  "kitchenware": ["pitcher_base", "bowl", "mug", "skillet_lid", "plate", "sponge"],
  "cutlery": ["fork", "spoon", "knife", "spatula"],
  "tool": ["power_drill", "scissors", "padlock", "adjustable_wrench", "phillips_screwdriver", "flat_screwdriver", "hammer",
           "medium_clamp", "large_clamp", "extra_large_clamp", "large_marker"],
  "ball": ["mini_soccer_ball", "softball", "baseball", "tennis_ball", "racquetball", "golf_ball", "marbles"],
  "block": ["wood_block", "foam_brick", "dice", "colored_wood_blocks", "nine_hole_peg_test", "lego_duplo", "chain",
            "rubiks_cube", "toy_airplane", "cups"],
}


def category(obj: str) -> str:
  name = obj.split("_", 1)[1]
  return next((c for c, names in CATEGORY.items() if any(name.startswith(n) for n in names)), "other")


def objects() -> list[str]:
  return sorted(p.parent.parent.name for p in SRC.glob("*/google_16k/nontextured.ply"))


def load_mesh(obj: str):
  import trimesh

  return trimesh.load(SRC / obj / "google_16k" / "nontextured.ply", force="mesh")


def decompose(obj: str) -> tuple[list[dict], dict]:
  """-> (parts, info). Parts are convex hulls in the object frame (bbox centre at the origin)."""
  import coacd

  mesh = load_mesh(obj)
  raw = (SRC / obj / "google_16k" / "nontextured.ply").read_bytes()
  key = hashlib.sha1(raw + repr(sorted(COACD.items())).encode()).hexdigest()[:16]
  f = CACHE / f"{obj}_{key}.npz"
  if f.exists():
    z = np.load(f)
    hulls = [z[f"v{i}"] for i in range(int(z["n"]))]
  else:
    coacd.set_log_level("error")
    res = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **COACD)
    hulls = [np.asarray(v) for v, _ in res]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, n=len(hulls), **{f"v{i}": v for i, v in enumerate(hulls)})
  c = (mesh.bounds[0] + mesh.bounds[1]) / 2
  parts = [part("hull", [0, 0, 0], verts=v - c) for v in hulls]
  mesh.apply_translation(-c)
  return parts, dict(watertight=bool(mesh.is_watertight), mesh_volume=float(abs(mesh.volume)), mesh_faces=len(mesh.faces),
                     extents=mesh.extents.tolist(), sha256=(SRC / obj / "sha256").read_text().strip(), mesh=mesh)


def hausdorff(mesh, parts: list[dict], n: int = 20000, seed: int = 0) -> dict:
  """Symmetric Hausdorff and mean distance between the mesh surface and the union-of-hulls surface (m)."""
  from scipy.spatial import cKDTree

  a, _ = __import__("trimesh").sample.sample_surface(mesh, n, seed=seed)
  b = union_surface(parts, n, seed)  # n per part: sparse sampling alone would add ~mm of "error"
  dab = cKDTree(b).query(a)[0]
  dba = cKDTree(a).query(b)[0]
  return dict(hausdorff=float(max(dab.max(), dba.max())), hausdorff_p99=float(max(np.percentile(dab, 99), np.percentile(dba, 99))),
              mean=float((dab.mean() + dba.mean()) / 2))
