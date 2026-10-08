"""Pinned sources for real hands. Clone with scripts/fetch_real.py into data/sources/<repo>.

Converted assets are generated locally from these sources and never redistributed:
several licenses (CC-BY-SA, LGPL, NASA-1.3, CC-BY-NC-ND) carry redistribution conditions.
"""

from dataclasses import dataclass, field

# name: (url, commit, sparse paths or None for the whole repo)
REPOS = {
  "icub": ("https://github.com/robotology/icub-models", "dca8c618c264be70ce285bad38040411725e3551", ["iCub"]),
  "nasa": ("https://github.com/gkjohnson/nasa-urdf-robots", "1f89ecba9112361cde5beda617575bdd7b811b24", None),
  "talos": ("https://github.com/stack-of-tasks/talos-data", "dc636c94d307fa51322ebc0ea86b72b5df109724", None),
  "nao": ("https://github.com/ros-naoqi/nao_robot", "67476469a1371b00b17538eb6ea336367ece7d44", ["nao_description"]),
  "romeo": ("https://github.com/ros-aldebaran/romeo_robot", "e53c8f3d705739cfd00fd9f2b4418dc1f8a2b733", ["romeo_description"]),
  "walkman": ("https://github.com/ADVRHumanoids/iit-walkman-ros-pkg", "d38fe56a844f80f699afac583faf17df5e1d5d97", None),
  "jaxon": ("https://github.com/robot-descriptions/jaxon_description", "4a0cb7a4a737864312f8d6e3f89823a741539bfc", None),
  "jvrc": ("https://github.com/isri-aist/jvrc_mj_description", "8cfc65f5c5793582b55e0d5eb4726dfb3094a224", None),
  "unitree_ros": ("https://github.com/unitreerobotics/unitree_ros", "5994d4faef0a9cadd3287f8de0199a67eeb2a259", ["robots"]),
  "unitree_rl_gym": ("https://github.com/unitreerobotics/unitree_rl_gym", "276801e46c5d433564f24658bac64f254b7d2d4b", ["resources/robots/h1_2"]),
  "drake_models": ("https://github.com/RobotLocomotion/models", "f85076b4d95a8bdfec70153015e723e92312e4e2", ["atlas"]),
  "roboschool": ("https://github.com/openai/roboschool", "c8ee2812207ef620af618b065eca3bde10f5823c", ["roboschool/models_robot/atlas_description"]),
  "digit": ("https://github.com/adubredu/DigitRobot.jl", "bd3b99e2297c48c5f818f521791a9f8daf3ee303", None),
  "sigmaban": ("https://github.com/Rhoban/sigmaban_urdf", "d5d023fd35800d00d7647000bce8602617a4960d", None),
  "menagerie": ("https://github.com/google-deepmind/mujoco_menagerie", "f054586a8e90465d49ee5be15335c4a0c7f57caf", ["leap_hand", "wonik_allegro", "shadow_hand", "robotiq_2f85"]),
}

# Commercial use is required (user decision, 2026-10-07). Not legal advice: obligations to review before release.
COMMERCIAL = {
  "MIT": "yes",
  "BSD-2-Clause": "yes",
  "BSD-3-Clause": "yes",
  "Apache-2.0": "yes (notice)",
  "CC-BY-SA-4.0": "yes (attribution; share-alike on derived assets)",
  "LGPL-3.0": "yes (LGPL terms on modified model files)",
  "NASA-1.3": "yes (notice; modifications must be disclosed)",
}

# ROS packages that ship without a package.xml: name -> path under data/sources
EXTRA_PACKAGES = {
  "iCub": "icub/iCub",
  "r2_description": "nasa/r2_description",
  "val_description": "nasa/val_description",
  "jaxon_description": "jaxon",
}


@dataclass
class HandSource:
  id: str
  robot: str
  repo: str
  file: str  # relative to data/sources
  hand_root: str  # body whose subtree is the hand; its own joints (arm/wrist) are removed
  license: str
  source: str = "awesome-humanoid-learning"  # or "menagerie"
  notes: list[str] = field(default_factory=list)
  close: dict[str, str] = field(default_factory=dict)  # actuator -> "lo"/"hi" where the probe heuristics are wrong
  decompose: bool = False  # CoACD on collision meshes whose convex hull fills the grasp space


HANDS = [
  HandSource("icub_mk3_left", "iCub", "icub", "icub/iCub/robots/left_hand_mk3/model.urdf", "l_hand_palm_link", "CC-BY-SA-4.0",
             notes=["Only a left-hand mk3 model exists. Full iCub robot URDFs in this repo have no finger joints.",
                    "Real hand is tendon-coupled; URDF has no mimic/coupling, so joints are modeled independently."]),
  HandSource("r2_right", "Robonaut 2", "nasa", "nasa/r2_description/robots/r2c6_valve.urdf", "r2/right_wrist_yaw", "NASA-1.3"),
  HandSource("valkyrie_right", "Valkyrie", "nasa", "nasa/val_description/model/robots/forearm_right.urdf", "rightPalm", "NASA-1.3"),
  HandSource("talos_right", "TALOS", "talos", "talos/urdf/talos_full_v2.urdf", "gripper_right_base_link", "LGPL-3.0"),
  HandSource("romeo_right", "Romeo", "romeo", "romeo/romeo_description/urdf/romeo_generated_urdf/romeo.urdf", "r_wrist", "BSD-3-Clause"),
  HandSource("jaxon_right", "JAXON", "jaxon", "jaxon/urdf/jaxon_jvrc.urdf", "RARM_LINK7", "CC-BY-SA-4.0", decompose=True,
             notes=["Ring-shaped finger meshes: convex hulls fill the grasp space, so collision meshes are CoACD-decomposed."]),
  HandSource("jvrc1_right", "JVRC-1", "jvrc", "jvrc/xml/jvrc1.xml", "R_WRIST_Y_S", "BSD-2-Clause"),
  HandSource("h1_inspire_dfq_right", "Unitree H1", "unitree_ros", "unitree_ros/robots/h1_description/urdf/h1_with_hand.urdf", "R_hand_base_link", "BSD-3-Clause",
             notes=["Inspire DFQ hand; same hand as unitree_ros g1_description/inspire_hand/DFQ_*.urdf."]),
  HandSource("h1_2_inspire_ftp_right", "Unitree H1-2", "unitree_ros", "unitree_ros/robots/h1_2_description/h1_2.urdf", "R_hand_base_link", "BSD-3-Clause",
             notes=["Inspire FTP hand. The unitree_rl_gym h1_2.urdf carries the same hand; extracted once."]),
  HandSource("g1_dex3_1_right", "Unitree G1", "unitree_ros", "unitree_ros/robots/dexterous_hand_description/dex3_1/dex3_1_r.urdf", "right_hand_palm_link", "BSD-3-Clause"),
  HandSource("g1_dex1_1", "Unitree G1", "unitree_ros", "unitree_ros/robots/dexterous_hand_description/dex1_1/dex1_1.urdf", "base_link", "BSD-3-Clause"),
  HandSource("g1_brainco_right", "Unitree G1", "unitree_ros", "unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf", "right_base_link", "BSD-3-Clause"),
  HandSource("unitree_dex5_1_right", "Unitree (G1/H2)", "unitree_ros", "unitree_ros/robots/dexterous_hand_description/dex5_1/Dex5-URDF-R/Dex5-URDF-R.urdf", "base_link00", "BSD-3-Clause"),
  HandSource("unitree_dex2_5_right", "Unitree (G1/H2)", "unitree_ros", "unitree_ros/robots/dexterous_hand_description/dex2_5/Right_Hand.urdf", "right_0_link", "BSD-3-Clause"),
  HandSource("leap_right", "LEAP Hand", "menagerie", "menagerie/leap_hand/right_hand.xml", "palm", "MIT", source="menagerie"),
  HandSource("allegro_right", "Wonik Allegro", "menagerie", "menagerie/wonik_allegro/right_hand.xml", "palm", "BSD-2-Clause", source="menagerie"),
  HandSource("shadow_right", "Shadow Hand E3M5", "menagerie", "menagerie/shadow_hand/right_hand.xml", "rh_palm", "Apache-2.0", source="menagerie"),
  HandSource("robotiq_2f85", "Robotiq 2F-85", "menagerie", "menagerie/robotiq_2f85/2f85.xml", "base_mount", "BSD-2-Clause", source="menagerie"),
]

# Inspected and skipped: model -> reason
SKIPPED = {
  "WALK-MAN": "Pisa/IIT SoftHand modeled as a rigid mesh on fixed joints (soft_hand.urdf.xacro); no articulated hand.",
  "Atlas DRC v3 (drake models)": "Arm ends at the wrist (l/r_arm_lwy); no hand modeled.",
  "Atlas v4 (roboschool)": "Arm ends at the wrist (l/r_arm_wry2); no hand modeled.",
  "Digit": "No hand joints; no license listed (not usable commercially anyway).",
  "NAO": "Hand meshes (nao_meshes) are CC-BY-NC-ND: non-commercial, excluded. With guessed capsule geometry the hand was a "
         "mechanism-only embodiment that failed with any self-collision pair; dropped (user decision, 2026-10-08).",
  "SigmaBan": "No hand joints.",
  "Bolt, Cassie, Spryped, Upkie, Hector, SA01": "Bipeds without arms/hands; excluded by spec.",
}
