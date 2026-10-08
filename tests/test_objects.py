import numpy as np
import pytest

from mandahands.objects.parts import inside, part, union_surface, volume
from mandahands.objects.procedural import FAMILIES, generate
from mandahands.sim.builder import SceneCfg, build_scene


def test_union_volume_and_mass_count_overlap_once():
  # two 4 cm cubes overlapping by half: union = 1.5 cubes
  ps = [part("box", [0.02] * 3, [0, 0, 0]), part("box", [0.02] * 3, [0.02, 0, 0])]
  assert volume(ps) == pytest.approx(1.5 * 0.04**3, rel=0.02)
  m = build_scene(SceneCfg(), obj_parts=ps, obj_density=1000.0).compile()
  assert m.body("object").mass[0] == pytest.approx(1000 * 1.5 * 0.04**3, rel=0.03)


@pytest.mark.parametrize("t,size", [("box", [0.01, 0.02, 0.03]), ("sphere", [0.02]), ("capsule", [0.01, 0.02]),
                                    ("cylinder", [0.01, 0.02]), ("ellipsoid", [0.01, 0.02, 0.03])])
def test_inside_matches_primitive(t, size):
  p = part(t, size, [0.1, 0, 0], [0.7071068, 0.7071068, 0, 0])  # rotated and offset
  pts = union_surface([p], 500)
  assert inside(p, pts, 1e-4).all() and not inside(p, pts * 1.05 + 0.002).all()


def test_hull_inside():
  verts = np.random.default_rng(0).normal(size=(40, 3)) * 0.02
  p = part("hull", [0, 0, 0], verts=verts)
  assert inside(p, np.zeros((1, 3)))[0] and not inside(p, np.array([[1.0, 0, 0]]))[0]


@pytest.mark.parametrize("family", FAMILIES)
def test_procedural_families_compile(family):
  for i in range(3):
    ps = generate(0, family, i)
    assert 1 <= len(ps) <= 16
    m = build_scene(SceneCfg(), obj_parts=ps).compile()
    assert m.body("object").mass[0] > 0


def test_hausdorff_ignores_internal_cut_faces():
  import trimesh

  from mandahands.objects.ycb import hausdorff

  mesh = trimesh.creation.icosphere(4, 0.04)
  v = mesh.vertices
  halves = [part("hull", [0, 0, 0], verts=v[v[:, 2] >= -1e-9]), part("hull", [0, 0, 0], verts=v[v[:, 2] <= 1e-9])]
  err = hausdorff(mesh, halves, n=20000)
  assert err["hausdorff_p99"] < 0.002, err  # the equatorial cut face is not surface
