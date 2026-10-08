"""Embodiment registry: GENHAND datasets (manifests committed in registry/) and the real hands.

A GENHAND hand is regenerated from (seed, topology index t, variant v); the manifest stores its split, filter
result and a hash of the generated MJCF so drift in the generator or numpy is caught on load.
"""

import hashlib
import json
from functools import cache
from pathlib import Path

import mujoco

from mandahands.embodiments.procedural.generator import GENERATOR_VERSION, build, sample_genome
from mandahands.sim.builder import add_self_collision

REGISTRY = Path(__file__).resolve().parents[2] / "registry"
REAL_HANDS = Path("data/real_hands")  # generated locally by scripts/extract_real.py


def spec_hash(spec: mujoco.MjSpec) -> str:
  return hashlib.sha1(spec.to_xml().encode()).hexdigest()[:16]


@cache
def manifest(name: str = "genhand-1k-v4") -> dict:
  return json.loads((REGISTRY / f"{name}.json").read_text())


def ids(split: str, name: str = "genhand-1k-v4") -> list[str]:
  """Hand ids in a split: train / test_id / test_ood_topology / test_ood_scale, or train_N for a scaling subset."""
  man = manifest(name)
  if split.startswith("train_"):
    return man["scaling_subsets"][split[len("train_") :]]
  return [h["id"] for h in man["hands"] if h.get("split") == split]


def genhand(hand_id: str, name: str = "genhand-1k-v4") -> tuple[mujoco.MjSpec, dict]:
  """Regenerate a GENHAND hand (with its self-collision pairs) and check it matches the manifest."""
  man = manifest(name)
  if man["generator_version"] != GENERATOR_VERSION:
    raise RuntimeError(f"{name} was made with generator v{man['generator_version']}, code is v{GENERATOR_VERSION}")
  h = next(h for h in man["hands"] if h["id"] == hand_id)
  topo = man["topologies"][h["t"]]
  spec, meta = build(topo, sample_genome(topo, man["seed"], h["t"], h["v"]))
  if spec_hash(spec) != h["hash"]:
    raise RuntimeError(f"{hand_id}: regenerated model differs from the manifest (generator or numpy changed)")
  add_self_collision(spec)
  meta.update(id=hand_id, split=h["split"], source="genhand")
  return spec, meta


def real(hand_id: str) -> tuple[mujoco.MjSpec, dict]:
  d = REAL_HANDS / hand_id
  return mujoco.MjSpec.from_file(str(d / "hand.xml")), json.loads((d / "meta.json").read_text())


# real-hand actuator origin -> descriptor provenance
_SOURCE = {"source_model": "datasheet", "estimated": "estimated", "converted_from_motor": "estimated"}


def descriptor(hand_id: str, name: str = "genhand-1k-v4"):
  """Descriptor of a GENHAND or real hand, with actuator provenance (procedural values are exact: "measured")."""
  from mandahands.embodiments.descriptors import SOURCES, describe

  if hand_id.startswith("gh-"):
    spec, meta = genhand(hand_id, name)
    # procedural values are exact, but deployment descriptors are not: each generated hand gets a provenance label
    # (seeded by its id), which sets its descriptor noise in training (descriptors.PROVENANCE_NOISE)
    src = SOURCES[int(hashlib.sha1(hand_id.encode()).hexdigest(), 16) % 3]
    acts = [dict(a, source=src) for a in meta["actuators"]]
  else:
    spec, meta = real(hand_id)
    acts = [dict(name=a["name"], velocity=a.get("velocity") or None, source=_SOURCE[a["origin"]]) for a in meta["actuators"]]
  return describe(spec, acts)


def real_ids() -> list[str]:
  """Real hands extracted locally (scripts/extract_real.py), sorted."""
  return sorted(p.parent.name for p in REAL_HANDS.glob("*/hand.xml"))
