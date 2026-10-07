import mujoco
import numpy as np
import trimesh

from mandahands.embodiments.real import extract as ex
from mandahands.embodiments.real.sources import HandSource
from mandahands.sim.builder import SceneCfg, simple_hand
from mandahands.sim.scripted import grasp_search

URDF = """<robot name="toy">
  <link name="arm"/>
  <link name="palm">
    <inertial><mass value="0.2"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><mesh filename="a/link.stl"/></geometry></collision>
  </link>
  <link name="f1">
    <!-- placeholder inertial: 1 kg m^2 on a 5 cm link -->
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><mesh filename="b/link.stl"/></geometry></collision>
  </link>
  <link name="f2">
    <inertial><mass value="0.01"/><inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.01 0.01 0.04"/></geometry></collision>
  </link>
  <joint name="arm_j" type="revolute"><parent link="arm"/><child link="palm"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/></joint>
  <joint name="f1_j" type="revolute"><parent link="palm"/><child link="f1"/><origin xyz="0 0 -0.02"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="1.5" effort="1e6" velocity="3"/></joint>
  <joint name="f2_j" type="revolute"><parent link="f1"/><child link="f2"/><origin xyz="0 0 -0.05"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="1.5" effort="1" velocity="3"/><mimic joint="f1_j" multiplier="0.5"/></joint>
</robot>
"""


def test_extract_toy_urdf(tmp_path, monkeypatch):
  # same file stem in two directories, different sizes: each link must keep its own mesh
  for sub, size in (("a", 0.06), ("b", 0.02)):
    (tmp_path / "toy" / sub).mkdir(parents=True)
    trimesh.creation.box([size, size, size]).export(tmp_path / "toy" / sub / "link.stl")
  (tmp_path / "toy" / "toy.urdf").write_text(URDF)
  monkeypatch.setattr(ex, "SOURCES", tmp_path)
  monkeypatch.setattr(ex, "_packages", lambda: {})
  monkeypatch.setattr("mandahands.embodiments.real.load.MESH_CACHE", tmp_path / "cache")

  spec, meta = ex.extract(HandSource("toy", "Toy", "talos", "toy/toy.urdf", "palm", "MIT"))
  m = spec.compile()

  assert meta["removed_arm_joints"] == ["arm_j"]
  assert meta["joints"] == ["f1_j", "f2_j"]
  assert [a["target"] for a in meta["actuators"]] == ["f1_j"]  # follower gets no actuator
  assert meta["coupling"] == [dict(follower="f2_j", leader="f1_j", polycoef=[0.0, 0.5, 0.0, 0.0, 0.0])]
  assert "effort_placeholder:1" in meta["flags"] and "inertia_from_geometry:1" in meta["flags"]
  assert meta["actuators"][0]["effort"] == ex.DEFAULT_EFFORT[mujoco.mjtJoint.mjJNT_HINGE] and meta["actuators"][0]["velocity"] == 3.0
  # f1's placeholder inertia replaced by a plausible one
  f1 = m.body("f1").id
  assert np.sqrt(m.body_inertia[f1].max() / m.body_mass[f1]) < ex.MAX_GYRATION
  # mesh stems collided in MuJoCo's URDF importer before; the two links must keep different geometry
  sizes = sorted(m.geom_rbound[m.geom_type == 7])
  assert sizes[0] < 0.5 * sizes[1], sizes


def test_scripted_close_and_lift_test_hand():
  pr, res = grasp_search(simple_hand(SceneCfg()), n_centres=4)
  assert pr["close_rule"] and all(v == "range" for v in pr["close_rule"].values())
  assert any(r["ok"] for r in res)


def test_self_collision_pairs_simple_hand():
  m = simple_hand(SceneCfg()).compile()
  pairs = {tuple(sorted((m.geom(m.pair_geom1[i]).name, m.geom(m.pair_geom2[i]).name))) for i in range(m.npair)}
  tips = ["f0_g1", "f1_g1", "f2_g1"]  # distal phalanges only; proximal links never pair
  assert pairs == {tuple(sorted(("palm", t))) for t in tips} | {tuple(sorted(p)) for p in [tips[:2], tips[1:], tips[::2]]}
  assert (m.pair_dim == 1).all()  # frictionless
