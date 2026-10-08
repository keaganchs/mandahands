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

- **Closed four-bar linkage grippers** (Robotiq 2F-85-like): a subset of generated hands. Robotiq 2F-85 is out of
  the real-hand set until this exists (the coupling descriptor can't represent its closed loop). MuJoCo needs equality
  connect constraints to close the loop; check the MJWarp support and the per-world batching implications (M5).
- **Single-actuator hands with many coupled joints** (Romeo and TALOS: 1 actuator for 7–13 joints): a subset with
  higher joints-per-actuator than the current synergy hands.
- Re-run the coverage comparison afterwards (`scripts/gen_genhand.py`).

## B3. Larger-object categories

Source: user, 2026-10-08.

- Larger and heavier objects than the single-hand set (boxes, bulky items), mainly for B1.

## B4. Radially symmetric hands with varying finger counts

Source: user, 2026-10-08 (M4 review).

- Exactly radially symmetric hands (e.g. 3, 4, 6 or more identical fingers evenly spaced), as a generator family
  next to the current `radial` layout (2-5 fingers with jittered spacing and per-finger variation).
- Check the canonical frame for them: symmetric closing leaves the frame's y axis to a tie-break (`descriptors._frame`).

## B5. Real-hand geometry coverage

Source: M4 coverage; user, 2026-10-08: revisit after M7.

- Remaining token-level gaps are mostly geometric: finger-base layout, tilted joint axes, link mass.
- After M7, measure whether the gap hurts transfer before pushing the generator further.
