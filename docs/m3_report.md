# Milestone 3: procedural generator, validity filter, GENHAND-1K

Reproduce: `python scripts/gen_genhand.py` (about 6 min on 16 cores; needs `data/real_hands/` from `scripts/extract_real.py`), `pytest tests`.
Outputs (committed): `registry/genhand-1k-v1.json` (manifest: topologies, per-hand filter results, splits, hashes), `docs/genhand_report.md` (generated tables), `docs/genhand_coverage.png`, `docs/genhand_sheet.png`.

## Acceptance

| Check | Result |
|---|---|
| Rejection rate reported | **37.8%**: 995 of 1600 candidates valid (200 topology draws × 8 variants). Broken down by reason and by every topology field in `genhand_report.md` |
| Descriptor-space coverage reported | Hand-level summary vector (12 features), PCA plot, and nearest-neighbour distances from every real hand to the train set |
| Where the real hands fall | **10 of 18 within support** (nearest-neighbour distance below the train set's own 99th percentile). The 8 outside, and why, are listed below |
| Versioned dataset with splits | `genhand-1k-v1`: train 715, test_id 80, test_ood_topology 75, test_ood_scale 125, plus nested scaling subsets 8 / 32 / 128 / 512 / all. Each hand is regenerated from (seed, t, v) and checked against a stored model hash |
| Tests | `pytest tests`: 13 pass, 7 of them new (`tests/test_procedural.py`) |

The tests ran on the 3060 only: `pink_server` was unreachable ("No route to host") during this milestone.

## Generator (`mandahands/embodiments/procedural/generator.py`)

Sampling has two stages, so each topology class has many variants: `sample_topology(seed, t)`, then `sample_genome(topology, seed, t, v)`. Variants of a class have identical bodies, joints, actuators, equalities, tendons and geom types. A test checks this, because M5's per-world batching depends on it.

| Axis | What varies |
|---|---|
| Topology | Layout: anthro (fingers in a row; thumb none / opposed / coplanar), radial (fingers around the palm, e.g. 2- and 3-finger grippers), jaw (2 or 3 prismatic jaws). Also 2–5 fingers, 1–4 phalanges, abduction joints (none / all / thumb only) and their order relative to flexion, and fingertip shape (capsule, box, ellipsoid pad) |
| Geometry | Scale 0.6–1.8 (log-uniform; 1 ≈ adult hand), palm size, finger and phalanx lengths, distal taper, radii, base positions, splay, thumb placement, offset and spread |
| Kinematics | Axis perturbation (~3.5° s.d. per joint), flexion limits (−0.5 to 2.2 rad), abduction ±0.15–0.6 rad |
| Actuation | Coupling: full, mimic (distal joints follow the proximal one by equality), tendon (one cable per finger, return springs, underactuated), synergy (one actuator for the whole hand). Servo torque, stiffness, damping, armature, joint damping and frictionloss, velocity limit, backlash, latency |

- **Latency is simulated.** MuJoCo 3.15 has a native actuator `delay`, and MJWarp implements it, so generated latency (0–20 ms) goes straight into the model. The history buffer is fixed at `nsample=12`, so it is the same across variants.
- **Not simulated yet:** velocity limit and backlash have no native equivalent. They're stored as descriptor values for M7 to simulate.
- **MJWarp check:** a tendon hand with ellipsoid pads and delay gives CPU and MJWarp trajectories within 9e-5.

## Validity filter (`mandahands/embodiments/procedural/validity.py`)

A hand is rejected if:
1. Non-adjacent hand links overlap by more than 1 mm at rest, or at the open pose.
2. The scripted close-and-lift (from M2) lifts fewer than **k = 2** of 6 reference objects: sphere, cylinder and box at half-extents **12 and 25 mm**.

Rejections: 472 for grasp, 91 for overlap at the open pose (abduction swings adjacent fingers into each other), 4 for overlap at rest.

**Calibrating the filter on the real hands changed the plan.** All 18 real hands are valid by definition, so a filter that rejects them is too strict.

- **Spec-like settings fail half the real hands.** With half-extents of 20/35 mm and k=3, 8 of 18 real hands were rejected. R2 and Valkyrie lifted nothing. The cause is the open-loop script, not the hands: full closure at full torque squirts objects out.
- **Closing (`sim/scripted.py`):** each actuator now ramps closed until a link it drives touches the object, then squeezes a further 15% of its range. Full closure is still tried as a second strategy, because neither strategy wins on every hand: contact-stop broke Valkyrie and JVRC.
- **Candidate centres:** "pinch" centres, where the closed fingertips converge plus points back toward the palm, are tried first. The old enclosure heuristic often put objects below the fingertips.
- **Blow-ups:** a trial with a BADQACC warning now fails. MuJoCo silently resets the state when that happens, which could make a blown-up trial look like a pass.
- **Result:** every real hand lifts at least 2 of the 6 objects, so k=2. Per hand: R2 2, Valkyrie 2, JVRC 3, and 4–6 for the rest.
- **Effect on M2:** the M2 search was re-run with these changes. All 18 still pass, mostly with more passing trials; for example Dex3-1 went from 1/45 to 15/51. `real_hands_report.md` is updated.

**Rejection by layout:** jaw 0%, radial 8%, anthro 52%. Within anthro: coplanar thumb 65%, opposed 35%.

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

## Splits (`registry/genhand-1k-v1.json`, `mandahands/embodiments/registry.py`)

| split | hands | rule |
|---|---|---|
| train | 715 | the rest |
| test_id | 80 | random 10% of the rest |
| test_ood_topology | 75 | all variants of a random 10% of topology classes (16 of 163) |
| test_ood_scale | 125 | scale in the outer 7.5% of the log-scale range at each end (< 0.65 or > 1.65) |

Scaling subsets are nested prefixes of a seeded permutation of train. `registry.genhand(id)` regenerates a hand, adds its self-collision pairs, and fails loudly if the generator version or the model hash doesn't match the manifest.

## Coverage: where the real hands fall

The coverage features are 12 hand-level summary features computed the same way for real and generated hands: finger, joint and actuator counts, actuators per joint, slide fraction, fingertip span, reach and closure, mass, mean hinge range, servo force, and aperture. They are standardized by train statistics. This is a stand-in until M4's per-joint descriptors; M4 will repeat the comparison at token level.

Real-hand nearest-neighbour distance is compared with the train set's leave-one-out distances: median 1.06, 99th percentile 1.82.

| Out of support | Distance | Main reason |
|---|---|---|
| R2 | 9.7 | Placeholder joint ranges, clamped to ±π/2 in M2. A sanitization artifact, not a morphology gap |
| JAXON | 3.6 | 25 cm hook gripper: span, reach and force beyond the scale range |
| Romeo | 3.1 | 1 actuator driving 13 joints (actuators per joint below the train range) |
| TALOS | 2.5 | 1 actuator, 7 joints; a mimic-coupled gripper |
| JVRC-1 | 2.2 | Wide hinge ranges |
| Valkyrie | 2.2 | No single feature out of range; the combination is off |
| Robotiq 2F-85 | 1.9 | 4-bar linkage gripper; the generator has no closed linkages |
| LEAP | 1.9 | Large span and hinge ranges (just past p99) |

Inside support: Shadow, Allegro, Dex5-1, Dex3-1, Dex2-5, Dex1-1, both Inspire hands, BrainCo, iCub. The coverage plot shows the real hands inside the main procedural cloud, except R2 and JAXON.

The generator ranges were widened once from this comparison: scale up to 1.8, palm motor mass, flexion range to 2.2 rad, and more 2–3 phalanx fingers. Before that, real hands were heavier and larger than nearly all generated ones.

## Notes for later milestones

- **M5 batching:** 163 topology classes contain a valid hand (median 6 valid variants, max 40). In 121 classes, variants have **different numbers of self-collision pairs**, because unreachable pairs are dropped per variant. Per-world batching inside a class needs either the union of pairs, with unused ones made inert per world, or the reachability pruning done per class.
- **M7:** velocity limits and backlash are descriptor values only so far. Latency is in the model.
- **Validity cost:** about 3.2 s of CPU per candidate; 1600 candidates take 5.5 min on 16 cores.

## Decisions I made (please confirm or change)

1. Reference objects at half-extents 12/25 mm with k=2, calibrated so every real hand passes. The spec suggested "two sizes"; I chose the values.
2. Scripted grasp changes (contact-stop plus full closure, pinch candidates, BADQACC fails a trial). These also changed the M2 numbers.
3. Layout prior re-weighted so the valid set is about half anthropomorphic.
4. Split rules as in the table above. OOD topology classes are chosen at random, not by a semantic rule.
5. Latency simulated through native actuator delay; velocity limit and backlash deferred to M7.

## Open questions

1. **Out-of-support real hands:** should generator v2 add the missing mechanisms? Candidates: closed 4-bar linkage grippers (Robotiq), single-actuator hands with many mimic joints (Romeo, TALOS), and a larger scale range (JAXON). Or are these acceptable as genuinely out-of-distribution evaluation hands?
2. **Coplanar thumbs:** rejected 65% of the time by an open-loop script that can't use them. Keep the filter as is, or relax it for this class (e.g. k=1)?
3. **R2:** its distance is mostly an artifact of placeholder joint ranges. Should real-hand placeholder ranges be clamped tighter, e.g. to the ranges typical for that joint type?
