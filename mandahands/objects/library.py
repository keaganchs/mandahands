"""Object library: versioned manifest (registry/objects-<version>.json) -> parts for any object id.

Procedural objects regenerate from (seed, family, index); YCB objects from their cached CoACD decomposition
(recomputed from the mesh if the cache is missing; the manifest's mesh sha256 catches a changed download).
"""

import json
from functools import cache
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "registry"
VERSION = "objects-v1"


@cache
def manifest(version: str = VERSION) -> dict:
  return json.loads((REGISTRY / f"{version}.json").read_text())


def ids(split: str, version: str = VERSION, tier: str | None = None) -> list[str]:
  """split: train / test_instance / test_category (excluded objects have no split)."""
  return [o["id"] for o in manifest(version)["objects"] if o.get("split") == split and (tier is None or o["tier"] == tier)]


def parts(obj_id: str, version: str = VERSION) -> list[dict]:
  o = next(o for o in manifest(version)["objects"] if o["id"] == obj_id)
  if o["tier"] == "procedural":
    from mandahands.objects.procedural import generate

    return generate(manifest(version)["seed"], o["category"], o["index"])
  from mandahands.objects import ycb

  if (ycb.SRC / o["source_name"] / "sha256").read_text().strip() != o["sha256"]:
    raise RuntimeError(f"{obj_id}: local YCB mesh differs from the library's (sha256)")
  return ycb.decompose(o["source_name"])[0]
