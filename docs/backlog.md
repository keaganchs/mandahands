# Backlog milestone (after M11)

Work deliberately deferred until the 11 planned milestones are done. Each item lists where it came from.

## B1. Bimanual manipulation of large objects

Source: user, 2026-10-08 (M3 review).

- A bimanual policy for moving larger boxes and large objects in general.
- Current milestones stay single-hand grasping of small objects (mugs, cooking pans, garden tools, ...).
- Likely needs: two floating wrists (or a torso with two arms) in the scene builder; larger objects and their
  contact capacity; a command interface for both hands; and a hierarchy interface change (ask first, per the spec).

## B2. More procedural mechanisms

Source: M3 coverage (8 of 18 real hands outside the procedural support); user, 2026-10-08.

- **Closed four-bar linkage grippers** (Robotiq 2F-85-like): a subset of generated hands. MuJoCo needs equality
  connect constraints to close the loop; check the MJWarp support and the per-world batching implications (M5).
- **Single-actuator hands with many coupled joints** (Romeo and TALOS: 1 actuator for 7–13 joints): a subset with
  higher joints-per-actuator than the current synergy hands.
- Re-run the coverage comparison afterwards (`scripts/gen_genhand.py`).

## B3. Larger-object categories

Source: user, 2026-10-08.

- Larger and heavier objects than the single-hand set (boxes, bulky items), mainly for B1.
