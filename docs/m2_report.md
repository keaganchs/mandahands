# Milestone 2: real gripper extraction

Reproduce: `python scripts/fetch_real.py`, `python scripts/extract_real.py` (about 20 s on 8 cores), `pytest tests`.
Outputs: `data/real_hands/<id>/{hand.xml, meta.json, grasp.png}` (local, gitignored), `docs/real_hands_report.md` (committed).

## Acceptance

| Check | Result |
|---|---|
| Every extractable hand passes a scripted close-and-lift on a primitive | **19 / 19 pass** (14 hands from the awesome-humanoid-learning models, 4 Menagerie hands, plus Robotiq 2F-85). Per-hand trial counts are in `real_hands_report.md`. |
| Models without hands are documented as skipped, with a reason | WALK-MAN, Atlas DRC v3, Atlas v4, Digit, SigmaBan, and the bipeds (table in the report) |
| Saved hands reload | All 19 `hand.xml` files reload and match the in-memory model: joints, actuators, equalities, masses, ranges and gains |
| Tests | `pytest tests`: 6 pass, including a toy-URDF extraction test (mimic coupling, arm-joint removal, placeholder sanitizing, mesh-name collision) and the self-collision pair rules |

## What was found vs the spec's guesses

| Spec model | Spec's guess | Found |
|---|---|---|
| iCub | Multi-finger, coupled | Full-robot URDFs have **no finger joints**. Only `left_hand_mk3` (11 joints) exists. The real hand is tendon-coupled, but the URDF has no coupling, so its joints are modeled independently. |
| Robonaut 2 | Dexterous | 18 joints. **Placeholder dynamics** (1 kg / 1 kg·m² per phalanx, effort 1e7, ranges ±4 rad), all replaced and flagged. |
| Valkyrie | Multi-finger | 13 joints, from the standalone `forearm_right.urdf` |
| TALOS | Underactuated, mimic | 1 actuator and 6 mimic joints, as expected |
| NAO | Single actuator | 1 actuator and 8 mimic joints. **Meshes are CC-BY-NC-ND**, so they're excluded under the commercial-use rule; capsule collision instead. Mechanism-only embodiment. |
| Romeo | Single actuator | 1 actuator and 12 mimic joints |
| WALK-MAN | Synergy hand, if modeled | **Not modeled**: the SoftHand is a rigid mesh on fixed joints. Skipped. |
| JAXON | Simple gripper | 2-joint hook gripper with ring-shaped fingers, which needed convex decomposition |
| JVRC-1 | Simple gripper | 6 joints, 5 coupled; the source actuator is a torque motor, converted to a position servo |
| Unitree G1 / H1 / H1-2 | Check variants | Five distinct hands: Dex3-1, Dex1-1 (parallel jaw), Inspire DFQ (from H1), Inspire FTP (from H1-2), BrainCo. Also the standalone Dex5-1 (20 DoF) and Dex2-5 from `unitree_ros`. Duplicates were extracted once. |
| Atlas v3 / v4 | Maybe no hand | No hand; the arm ends at the wrist |
| Digit, SigmaBan | Likely no hand | No hand |
| Menagerie (added per your OK) | — | LEAP, Allegro, Shadow E3M5, Robotiq 2F-85. These ship real actuator models; kept as is. |

## Pipeline (`mandahands/embodiments/real/`)

- `sources.py`: pinned repos, per-hand root link, licenses, and the few manual overrides (one closing direction, one decomposition flag)
- `load.py`: URDF/MJCF to MjSpec. Resolves `package://`, converts `.dae` to cached STL, makes MJCF asset paths absolute, and gives meshes unique names (see bug 1).
- `extract.py`: cuts the subtree, removes the arm joints, records mimic coupling, adds collision capsules where meshes are missing, adds estimated position servos for URDFs, sanitizes placeholders, and optionally runs CoACD. Everything it changes is flagged in `meta.json`.
- `mandahands/sim/scripted.py`: hand-agnostic close-and-lift. It probes each actuator's closing direction, finds enclosed candidate object centres from the open/closed hand geometry (`mj_geomDistance`), and runs the lift test. The procedural validity filter (M3) will reuse it.

## Bugs found and fixed (some silent, worth knowing)

1. **MuJoCo's URDF importer names mesh assets by file stem.** TALOS's gripper base silently got the *torso* mesh (`gripper/base_link.STL` vs `torso/base_link*`), filling the jaws. Fix: every mesh path gets a unique hashed filename in the cache.
2. **`mjWARN_BADQACC` silently resets `mjData`**, which re-enables or disables equality constraints, restores qpos0, and so on. A blown-up simulation looks like a teleport, not a NaN. The env (M7) must check `d.warning` / MJWarp state and treat it as a failed episode.
3. **The M1 test hand barely moved**: `MjSpec()` defaults to degrees, so its [-0.4, 1.2] rad limits were degrees. M1's "grasp" was a near-rigid pinch. Fixed (`compiler.degree = False` everywhere), and the M1 numbers have been re-run (see the correction in `m1_report.md`).
4. Attaching an already-compiled MjSpec corrupts equality ids, so the builder attaches a copy. `attach_body(prefix="")` writes an unnamed `<default>` that MuJoCo's own parser rejects, so those blocks are named on save.

## Decisions applied after review

- **`impratio=3`** is the default (confirmed). At 10, 29 of about 800 trials across 10 hands blew up (BADQACC); even after the contact standardization below, LEAP still blew up in 5 of 21. At 1 there were no blow-ups, but R2 and JVRC couldn't hold anything.
- **Commercial use required.** Every included license allows it (MIT, BSD, Apache-2.0, CC-BY-SA-4.0, LGPL-3.0, NASA-1.3). Obligations are listed per hand in `real_hands_report.md` (`COMMERCIAL` in `sources.py`; not legal advice):
  - CC-BY-SA: share-alike on derived assets
  - LGPL: modified model files stay LGPL
  - NASA-1.3: notice and disclosure of modifications

  NAO's CC-BY-NC-ND meshes are excluded and Digit (no license) stays skipped.
- **Limited self-collision, on.** Explicit contact pairs only (hand geoms still never collide through contype bits), defined per hand by `add_self_collision` in `sim/builder.py` and carried through `attach_body`:
  - Each fingertip pairs with the palm and with the fingertips of its two nearest fingers (by base position; for a 2-jaw gripper, the other jaw).
  - Fingertip = the deepest link with collision geoms on a finger; one geom per tip (the one farthest from its parent).
  - Palm = the mount plus every fixed single-child link below it.
  - Pairs penetrating at rest are skipped (overlapping source meshes: TALOS 4, JAXON 3). So are pairs that never come within 1 cm over 64 random joint configurations plus simulated open/mid/closed poses.
  - Pairs are frictionless (`condim=1`): they only need to stop interpenetration.
  - Result: 1–25 pairs per hand (LEAP 36). NAO is off: its guessed capsules collide where the real hand wouldn't (no pairs: passes; any pairs: fails).
  - **Cost is real for dexterous hands.** Measured at 4096 worlds on the 3060, best grasp held: LEAP 238k → 113k steps/s (2.1×), Shadow 277k → 186k (1.5×). A first version (every geom on the tip link, condim 3) gave LEAP 156 pairs, a 10× slowdown, and capacity overflow. A cheaper option if needed later: sphere proxies at the fingertips for self-collision only.

## Other physics changes (since M1)

- **One contact model for all hands.** Hand geoms get the same `solref/solimp/margin/gap/priority`, and friction stays per hand. Sources disagree wildly (LEAP uses `solref=1e-4`). Contact parameters get randomized in M7 anyway.
- **Integrator: `implicitfast`** (was Euler); Euler blew up the stiff servos on light links. MJWarp supports it.

## Caveats

- The scripted test is a physics and actuation sanity check, not grasp quality. It's open-loop with a fixed closing command. Many dexterous hands pass only by caging small objects (6–10 mm half-size boxes) in a closed fist. Pass counts per hand range from 1/24 (NAO) and 1/45 (Dex3-1) to 32/48 (TALOS).
- **14 of 19 hands have estimated actuators** (no actuator model in the URDF). Gains come from URDF effort over a quarter of the joint range. Several efforts are placeholders. These descriptors must be marked `estimated` and randomized widely (Component 2).
- The closing direction comes from rules: the end of an asymmetric range farther from zero, otherwise geometry. One manual override: NAO `RHand` 0 = closed (Aldebaran convention; Romeo's URDF is the opposite, found empirically).
- Collision approximations: NAO (all capsules), R2 (13 links), Romeo (1). JAXON uses CoACD parts. Other hands use the convex hulls of their source meshes. Decomposition is opt-in per hand to keep contact counts down.
- The inertia replacement for placeholder links is a box estimate (marked `ponytail:` in code).

## Open questions

1. Self-collision roughly halves throughput on the dexterous hands. Keep it as is, or switch to fingertip sphere proxies (cheaper, less exact) before the batching work in M5?
2. NAO is now a mechanism-only embodiment (guessed geometry). Keep it in the real held-out set, or move it to training-only / drop it?
