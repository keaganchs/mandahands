# Milestone 6: object library

Reproduce: `python scripts/fetch_ycb.py && python scripts/build_objects.py` (geometry ~16 min with 6 workers, contacts ~50 min on one A5000; `--contacts-only` reruns the contact stage on the existing manifest); `pytest tests`.
Code: `mandahands/objects/` (parts, procedural, ycb, library), `scripts/build_objects.py`, `scripts/fetch_ycb.py`; `build_scene(..., obj_parts=, obj_density=)` in `mandahands/sim/builder.py`.
Outputs: `registry/objects-v1.json` (manifest), `docs/objects_report.md` (per-object table), `docs/objects_sheet.png`.

## Acceptance

| check | result |
|---|---|
| decomposition error reported per object | yes: p99 / mean Hausdorff between the YCB mesh and its convex decomposition (procedural objects are exact unions of primitives) |
| objects excluded for decomposition error | **0** of 78 YCB (threshold p99 > max(5 mm, 8% of the bbox diagonal)); worst included p99 11.3 mm (mug) |
| contact counts reported per object × hand | yes: peak contacts and constraint rows per world over descend → close → lift → hold, 16 worlds per run, 2 hands (LEAP, G1 Dex3) |
| capacity overflow | **0** worlds across 448 runs (capacity nconmax 200 / njmax 1500 per world) |
| peak contacts / rows per world | contacts median 26, max 78; rows median 94, max 298 (≤ 40% of capacity) |

## Library

| tier | objects | excluded | train | test_instance | test_category | parts median / max |
|---|---|---|---|---|---|---|
| procedural | 150 | 0 | 104 | 16 | 30 | 1 / 12 |
| ycb | 78 | 4 | 50 | 10 | 14 | 16 / 16 |

- **Procedural** (10 families × 15, seed 0): box, sphere, cylinder, capsule, ellipsoid, and composites built directly from convex primitives — mug (wall ring + C handle), bowl (outward-tilted wall ring), pan (shallow ring + long handle), bottle (body, shoulder, neck, cap), tool (handle + trowel blade / hammer head / rake tines). Sizes target single-hand grasping, per the M6 focus on mugs, pans and garden tools.
- **YCB** (79 objects with `google_16k` meshes; 24 YCB items ship no such mesh and are skipped): CoACD, threshold 0.04, ≤16 hulls, seed 0; decompositions cached in `data/cache/coacd_objects`, and the manifest pins each mesh's sha256 so a changed download is caught.
- **Excluded (4):** mini soccer ball, colored wood blocks (a, the full set), toy airplane (a, assembled), pitcher base — all `too_large_single_hand` (smallest extent > 11 cm). They go to the bimanual/large-object backlog (B1/B3).
- **Splits:** per tier, two held-out categories (`test_category`) plus ~15% of the remaining instances (`test_instance`). Held out: **mug and pan** (procedural), **bottle and tool** (YCB). Note that the procedural mug/pan hold-out means the single-hand focus objects are evaluation-only in the procedural tier; YCB still trains on its mug, bowl and skillet lid.
- **Object frame:** origin at the bounding-box centre, z up in the resting orientation.
- **Physical DR ranges** (stored in the manifest, sampled per episode from M7): density 300–1500 kg/m³, friction 0.5–1.2. Contact runs use 600 kg/m³.
- **Mass with overlapping parts:** MuJoCo would count overlaps twice, so `parts.add_to_body` scales the density so the total mass is density × union volume (Monte Carlo); test passes within 3%.

## Decomposition error

The error is the Hausdorff distance between points on the original mesh and points on the *union boundary* of the hulls. Surface samples of a hull that lie inside (or within 1 mm of) another hull are internal cut faces and are dropped. Without that filter the first run excluded 35 YCB objects on cut faces alone (test: two halves of a sphere give < 2 mm).

Largest p99 errors among included objects (all under their thresholds): mug 11.3 mm (the hull most likely fills the handle loop), lego duplo g 11.2 mm, chain 11.1 mm, lego duplo e 10.1 mm, nine-hole peg test 9.2 mm, bowl 9.1 mm. Median p99 across included YCB: 3.7 mm.

## Contact protocol and results

Each included object × {leap_right, g1_dex3_1_right}: the hand is oriented fingers-down by its canonical frame, centred over the object at the centroid of its closed fingertips, descends until the palm is 5 mm above the object top (or the lowest open fingertip is 5 mm above the floor), closes, lifts 10 cm and holds. 16 worlds per run with random yaw and ±1 cm offset.

| hand | peak contacts / world (median, max) | peak rows / world (median, max) | overflow worlds | lifted (worlds) | objects lifted at least once |
|---|---|---|---|---|---|
| leap_right | 26, 78 | 96, 298 | 0 | 0.5% | 18 / 224 |
| g1_dex3_1_right | 25, 70 | 94, 273 | 0 | 32% | 150 / 224 |

The heaviest loads come from many-hull objects pressed by many links: chain, scissors, pans, mugs, skillet lid (60–78 contacts, 220–300 rows). Capacity 200 / 1500 leaves ≥ 2.5× headroom on contacts and 5× on rows. Per-object numbers: `docs/objects_report.md`.

## Problems found and fixed

- **`parts.inside()` crashed the local machine twice:** a hull's half-space test allocated points × facets (a YCB hull reached 76,610 facets → 114 GiB). Now chunked. The build moved to `pink_server`.
- **Decomposition error counted internal cut faces** (above): 35 false exclusions in the first server run, which was killed.
- **LEAP descended beside the object:** its wrist origin sits 6 cm from the palm, so placing the wrist over the object put the hand off to the side (LEAP touched only the floor and itself in the first contact run). The hand is now centred on its closed-fingertip centroid. Centring on the palm instead cut G1's lift rate, so the grasp centre is the right point.

## Known limits (ponytail)

- **Open-loop top grasp is not a grasp planner.** LEAP's open pose puts the index tip 14 cm below the palm, so the descent stops at the floor while the fingers close at palm height, above most objects. Its lift rate is 0.5% of worlds (G1: 32%). The protocol exists to measure contact load, not grasp success; M7's scripted/RL grasps exercise LEAP properly.
- **Uniform density scaling** for overlapping parts gives the right mass but slightly shifts the inertia.
- **Procedural bowls** show small gaps at the rim between tilted wall segments (visible in the sheet). Harmless for grasping; widen the segments if the shape autoencoder (M9) minds.
- **YCB categories are coarse:** "block" lumps nesting cups, chain, foam brick, toy airplane parts and duplo (30 objects).
- **YCB license:** CC-BY-4.0 per the ai-habitat Hugging Face mirror's LICENSE. The official site (ycbbenchmarks.com) returned HTTP 500 when checked; re-confirm at the source before any release.
- **Manifest `git` field:** written on `pink_server`, which has no git checkout; restamped locally after commit.
- **Scene:** still the placeholder support/table from M1.
