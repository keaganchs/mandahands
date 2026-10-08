# Milestone 3: procedural generator, validity filter, GENHAND-1K

Reproduce: `python scripts/gen_genhand.py` (about 6 min on 16 cores; needs `data/real_hands/` from `scripts/extract_real.py`), `pytest tests`.
Outputs (committed): `registry/genhand-1k-v2.json` (manifest: topologies, per-hand filter results, splits, hashes), `docs/genhand_report.md` (generated tables), `docs/genhand_coverage.png`, `docs/genhand_sheet.png`.

## Acceptance

| Check | Result |
|---|---|
| Rejection rate reported | **38.1%**: 991 of 1600 candidates valid (200 topology draws × 8 variants). Broken down by reason and by every topology field in `genhand_report.md` |
| Descriptor-space coverage reported | Hand-level summary vector (12 features), PCA plot, and nearest-neighbour distances from every real hand to the train set |
| Where the real hands fall | **12 of 18 within support** (nearest-neighbour distance below the train set's own 99th percentile). The 6 outside, and why, are listed below |
| Versioned dataset with splits | `genhand-1k-v2`: train 713, test_id 79, test_ood_topology 75, test_ood_scale 124, plus nested scaling subsets 8 / 32 / 128 / 512 / all. Each hand is regenerated from (seed, t, v) and checked against a stored model hash |
| Tests | `pytest tests`: 16 pass, 10 of them new (9 in `tests/test_procedural.py`, plus a range-sweep test in `tests/test_real.py`) |

The tests ran on the 3060 only: `pink_server` was unreachable ("No route to host") during this milestone.

## Generator (`mandahands/embodiments/procedural/generator.py`)

Sampling has two stages, so each topology class has many variants: `sample_topology(seed, t)`, then `sample_genome(topology, seed, t, v)`. Variants of a class have identical bodies, joints, actuators, equalities, tendons and geom types. A test checks this, because M5's per-world batching depends on it.

| Axis | What varies |
|---|---|
| Topology | Layout: anthro (fingers in a row; thumb none / opposed / coplanar), radial (fingers around the palm, e.g. 2- and 3-finger grippers), jaw (2 or 3 prismatic jaws). Also 2–5 fingers, 1–4 phalanges, abduction joints (none / all / thumb only) and their order relative to flexion, and fingertip shape (capsule, box, ellipsoid pad) |
| Geometry | Scale 0.6–1.8 (log-uniform; 1 ≈ adult hand), palm size, finger and phalanx lengths, distal taper, radii, base positions, splay, thumb placement, offset and spread |
| Kinematics | Axis perturbation (~3.5° s.d. per joint), flexion limits (−0.5 to 2.2 rad), abduction ±0.15–0.6 rad |
| Actuation | Coupling: full, mimic (distal joints follow the proximal one by equality), tendon_pull (one cable per finger that only pulls; joint return springs open the finger), tendon_bidir (one cable per finger driven both ways, no springs; underactuated, so the cable sets only the weighted sum of its joints), synergy (one actuator for the whole hand). Servo torque, stiffness, damping, armature, joint damping and frictionloss, velocity limit, backlash, latency |

- **Latency is simulated.** MuJoCo 3.15 has a native actuator `delay`, and MJWarp implements it, so generated latency (0–20 ms) goes straight into the model. The history buffer is fixed at `nsample=12`, so it is the same across variants.
- **Not simulated yet:** velocity limit and backlash have no native equivalent. They're stored as descriptor values for M7 to simulate.
- **MJWarp check:** a tendon hand with ellipsoid pads and delay gives CPU and MJWarp trajectories within 9e-5.

## Validity filter (`mandahands/embodiments/procedural/validity.py`)

A hand is rejected if:
1. Non-adjacent hand links overlap by more than 1 mm at rest, or at the open pose.
2. The scripted close-and-lift (from M2) lifts fewer than **k = 2** of 6 reference objects: sphere, cylinder and box at half-extents **12 and 25 mm**.

Rejections by reason are in `genhand_report.md`. Overlap at the open pose comes from abduction swinging adjacent fingers into each other.

**Calibrating the filter on the real hands changed the plan.** All 18 real hands are valid by definition, so a filter that rejects them is too strict.

- **Spec-like settings fail half the real hands.** With half-extents of 20/35 mm and k=3, 8 of 18 real hands were rejected. R2 and Valkyrie lifted nothing. The cause is the open-loop script, not the hands: full closure at full torque squirts objects out.
- **Closing (`sim/scripted.py`):** each actuator now ramps closed until a link it drives touches the object, then squeezes a further 15% of its range. Full closure is still tried as a second strategy, because neither strategy wins on every hand: contact-stop broke Valkyrie and JVRC.
- **Candidate centres:** "pinch" centres, where the closed fingertips converge plus points back toward the palm, are tried first. The old enclosure heuristic often put objects below the fingertips.
- **Blow-ups:** a trial with a BADQACC warning now fails. MuJoCo silently resets the state when that happens, which could make a blown-up trial look like a pass.
- **Result:** every real hand lifts at least 2 of the 6 objects, so k=2. Per hand: R2 2, Valkyrie 2, JVRC 3, and 4–6 for the rest.
- **Effect on M2:** the M2 search was re-run with these changes. All 18 still pass, mostly with more passing trials; for example Dex3-1 went from 1/45 to 15/51. `real_hands_report.md` is updated.

**Rejection by layout:** jaw 0%, radial 8%, anthro 52%. By coupling: full 38%, mimic 31%, tendon_pull 34%, tendon_bidir 29%, synergy 54%. Within anthro: coplanar thumb 65%, opposed 35%.

- **Coplanar thumbs:** a coplanar thumb without a rotation joint can't oppose the fingers, so the scripted grasp can't use it. A learned policy might, so the filter is likely too strict here. I've left it, and report it.
- **Prior re-weighted:** because anthro hands fail far more often, the layout prior is anthro 0.65 / radial 0.25 / jaw 0.10. The valid set comes out 54% anthro / 32% radial / 14% jaw, closer to the real hands (13 of 18 anthropomorphic).

Generator fixes made during calibration (each found by rendering the failures):
- Phalanges are at least 2.5 radii long, so a finger doesn't overlap itself.
- The palm is widened to fit the fingers.
- Gripper palms are flat.
- Splay signs were inverted, and are fixed.
- The opposed thumb's pivot is near the palm's bottom edge. It previously curled into the palm.
- The opposed thumb rests spread away from the fingers. The open aperture was about 1.5 cm.
- The coplanar thumb curls obliquely across the palm.

## Splits (`registry/genhand-1k-v2.json`, `mandahands/embodiments/registry.py`)

| split | hands | rule |
|---|---|---|
| train | 713 | the rest |
| test_id | 79 | random 10% of the rest |
| test_ood_topology | 75 | all variants of a random 10% of topology classes (16 of 163) |
| test_ood_scale | 124 | scale in the outer 7.5% of the log-scale range at each end (< 0.65 or > 1.65) |

Scaling subsets are nested prefixes of a seeded permutation of train. `registry.genhand(id)` regenerates a hand, adds its self-collision pairs, and fails loudly if the generator version or the model hash doesn't match the manifest.

## Coverage: where the real hands fall

The coverage features are 12 hand-level summary features computed the same way for real and generated hands: finger, joint and actuator counts, actuators per joint, slide fraction, fingertip span, reach and closure, mass, mean hinge range, servo force, and aperture. They are standardized by train statistics. This is a stand-in until M4's per-joint descriptors; M4 will repeat the comparison at token level.

Real-hand nearest-neighbour distance is compared with the train set's leave-one-out distances: median 1.07, 99th percentile 1.85.

| Out of support | Distance | Main reason |
|---|---|---|
| JAXON | 3.3 | 25 cm hook gripper: span, reach and closure beyond the scale range |
| Romeo | 3.2 | 1 actuator driving 12 joints (actuators per joint below the train range). Backlog B2 |
| R2 | 3.1 | Hinge ranges: full range of motion (unspecified in the URDF), and its thin approximated collision capsules let fingers fold far before they clearly intersect |
| JVRC-1 | 2.6 | Wide hinge ranges (±2.36 rad in the source) |
| Robotiq 2F-85 | 1.9 | 4-bar linkage gripper; the generator has no closed linkages. Backlog B2. Just outside |
| LEAP | 1.9 | Large span and hinge ranges. Just outside |

Inside support: Shadow, Allegro, Dex5-1, Dex3-1, Dex2-5, Dex1-1, both Inspire hands, BrainCo, iCub, Valkyrie, TALOS. The coverage plot shows the real hands inside the main procedural cloud, except JAXON.

The generator ranges were widened once from this comparison: scale up to 1.8, palm motor mass, flexion range to 2.2 rad, and more 2–3 phalanx fingers. Before that, real hands were heavier and larger than nearly all generated ones.

## Notes for later milestones

- **M5 batching:** 163 topology classes contain a valid hand (median 6 valid variants, max 40). In 121 classes, variants have **different numbers of self-collision pairs**, because unreachable pairs are dropped per variant. Per-world batching inside a class needs either the union of pairs, with unused ones made inert per world, or the reachability pruning done per class.
- **M7:** velocity limits and backlash are descriptor values only so far. Latency is in the model.
- **Validity cost:** about 3.2 s of CPU per candidate; 1600 candidates take 5.5 min on 16 cores.

## Review decisions (2026-10-08) and what changed

- **Tendons:** coupling now includes both spring-loaded one-directional tendons (`tendon_pull`: actuator force limited to [0, F], joint return springs) and bidirectional tendons (`tendon_bidir`: ±F, no springs). This is generator v2, so the dataset is now `genhand-1k-v2` (v1 removed).
  - The return springs had to be stiffer than first sampled (0.15–0.4 × torque per rad, was 0.02–0.1). Weaker springs lost to joint friction (up to 5% of torque), leaving fingers half closed. A test covers both tendon types.
- **Reference objects:** the primitives are kept for now. The object dataset (mugs, cooking pans, garden tools, ...) is M6; once it exists, the validity filter should be re-run with it.
- **Layout mix and coplanar-thumb rejection:** kept as is.
- **Backlog** (`docs/backlog.md`): bimanual policy for large boxes and objects; four-bar linkage grippers and one-motor many-joint hands; larger-object categories.
- **Real-hand joint ranges: full range of motion when unspecified.** A hinge with no usable range (unlimited or wider than a full turn) gets ±π. It is cut only where sweeping the joint makes its links clearly intersect the rest of the hand: more than 2 mm of new surface penetration, or a point of the moving link entering another part, which catches a link folding back through its parent. The old ±π/2 clamp is gone. Affected: R2 (18 joints, all cut by the sweep).

Implementing the range rule exposed three extraction bugs, now fixed:
1. **Estimated and converted actuators never got their force limits.** The lines were dead code after a `return` in `_servo`, so M2's servos were effectively unlimited.
   - With limits on, iCub dropped to 0 passes. Its URDF has placeholder damping (1 Nm s/rad on every finger joint), so its fingers couldn't close in time.
   - Damping too high for a joint to reach 3 rad/s at its actuator's effort is now replaced (`damping_placeholder`: iCub 11 joints, TALOS 1).
   - JVRC's source motor has no torque limit; a missing effort now defaults to 1 Nm per joint the actuator drives through coupling.
2. **Zero-width URDF limits (lower == upper) are fixed joints.** MuJoCo's importer turns them into hinges with range [0, 0], the same as a continuous joint. BrainCo's 10 fingertip frames had been extracted as 5 actuated joints (M2 clamped them to ±π/2). `load_urdf` now makes them fixed: BrainCo has 11 joints and 6 actuators, was 16 and 11.
3. As a result, every real hand was re-extracted. All 18 still pass, and per-hand counts are in `real_hands_report.md`.
