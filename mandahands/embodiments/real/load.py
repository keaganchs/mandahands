"""Load a robot description (URDF or MJCF) into an MjSpec with resolved, MuJoCo-readable mesh paths."""

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

MESH_CACHE = Path("data/cache/meshes")
_NATIVE = {".stl", ".obj", ".msh"}


def find_packages(root: Path) -> dict[str, Path]:
  """ROS package name -> directory, from every package.xml under root."""
  out = {}
  for px in root.rglob("package.xml"):
    try:
      name = ET.parse(px).getroot().findtext("name")
    except ET.ParseError:
      continue
    if name:
      out[name.strip()] = px.parent
  return out


def _unique(fn: Path) -> Path:
  """Cached copy of a mesh under a name unique to its source path.

  MuJoCo's URDF importer names mesh assets by file stem, so gripper/base_link.STL and torso/base_link.STL
  collide and one link silently gets the other's geometry. Meshes MuJoCo can't read (e.g. .dae) become STL.
  """
  stem = hashlib.sha1(str(fn.resolve()).encode()).hexdigest()[:16]
  native = fn.suffix.lower() in _NATIVE
  out = MESH_CACHE / (stem + (fn.suffix.lower() if native else ".stl"))
  if not out.exists():
    out.parent.mkdir(parents=True, exist_ok=True)
    if native:
      out.symlink_to(fn.resolve())
    else:
      import trimesh

      trimesh.load(fn, force="mesh").export(out)
  return out.resolve() if not native else out.absolute()


def load_urdf(path: Path, packages: dict[str, Path]) -> tuple[mujoco.MjSpec, list[str], dict[str, float]]:
  """Returns (spec, notes, urdf velocity limits by joint). Geoms whose mesh file is missing are dropped and noted."""
  robot = ET.parse(path).getroot()
  notes, missing = [], set()
  for parent in list(robot.iter("visual")) + list(robot.iter("collision")):
    mesh = parent.find("geometry/mesh")
    if mesh is None:
      continue
    fn = mesh.get("filename", "")
    if m := re.match(r"package://([^/]+)/(.*)", fn):
      base = packages.get(m.group(1))
      p = base / m.group(2) if base else None
    elif fn.startswith("file://"):
      p = Path(fn[len("file://"):])
    else:
      p = (path.parent / fn).resolve()
    if p is None or not p.exists():
      missing.add(fn)
      parent.set("_drop", "1")
      continue
    mesh.set("filename", str(_unique(p)))
  for link in robot.iter("link"):
    for el in [e for e in link if e.get("_drop")]:
      link.remove(el)
  if missing:
    notes.append(f"{len(missing)} mesh files missing; their geoms were dropped (e.g. {sorted(missing)[0]})")
  for ext in robot.findall("mujoco"):
    robot.remove(ext)
  ET.SubElement(ET.SubElement(robot, "mujoco"), "compiler",
                strippath="false", fusestatic="false", discardvisual="true", balanceinertia="true")
  vel = {j.get("name"): float(j.find("limit").get("velocity", 0) or 0)
         for j in robot.iter("joint") if j.find("limit") is not None}
  return mujoco.MjSpec.from_string(ET.tostring(robot, encoding="unicode")), notes, vel


def load_mjcf(path: Path) -> mujoco.MjSpec:
  """MJCF with mesh/texture paths made absolute, so the extracted hand can be saved anywhere."""
  s = mujoco.MjSpec.from_file(str(path))
  for assets, sub in ((s.meshes, s.meshdir), (s.textures, s.texturedir)):
    for a in assets:
      if a.file and not Path(a.file).is_absolute():
        a.file = str((path.parent / sub / a.file).resolve())
  s.meshdir = s.texturedir = ""
  return s


def load(path: Path, packages: dict[str, Path] | None = None):
  if path.suffix == ".urdf":
    return load_urdf(path, packages or {})
  return load_mjcf(path), [], {}
