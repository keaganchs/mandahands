# Milestone 5: batching strategy

Reproduce: `python scripts/bench_groups.py --worlds 8400` (3060) or `--worlds 16800` (A5000); `pytest tests`.
Code: `mandahands/sim/groups.py` (Group, MultiGroup, union_pairs, signature).
Raw results: `results/bench_groups_<gpu>.{json,md}` (local, gitignored).

## Acceptance

| config | physics steps/s | control steps/s per world | max contacts / constraint rows per world |
|---|---|---|---|
| 1 group: mean of 10 classes, each run alone | **0.83 M** (range 0.38–1.34 M) | 10 | – |
| 10 groups, sequential | 0.64 M | 8 | 17 / 84 |
| 10 groups, one CUDA stream each | **0.86 M** | 10 | 16 / 88 |
| 100 groups, sequential | 0.13 M | 2 | 19 / 84 |
| 100 groups, one CUDA stream each | **0.46 M** | 6 | 19 / 88 |
| superset: 5 fingers × 3 + abduction (19 hand DoF), random finger subsets per world | **0.60 M** | 7 | 9 / 44 |

- **Setup:** RTX 3060 Laptop, 8400 worlds in total. Every world holds a 12 mm sphere in a closed grasp, with the object on its support weld.
- **Stepping:** the control step is 10 × 2 ms, synchronized every control step (the policy reads the state). Groups are GENHAND topology classes from the train split, up to 4 variants each.
- **Capacity:** no contact-capacity overflow anywhere (njmax 400). Peak memory 2.8 GiB.
- **RTX A5000** (`pink_server`, 16800 worlds, run during M6): 1 group 2.01 M (range 0.93–3.29 M); 10 groups 1.43 M sequential, **2.09 M** with streams; 100 groups 0.26 M sequential, **1.02 M** with streams; superset 1.56 M. No overflow; peak memory 5.5 GiB. Full table: `results/bench_groups_NVIDIA_RTX_A5000.md`.

## What was built

**Group** (strategy 1): one compiled model per topology class, holding several variants (different geometry, masses, joint axes and ranges, gains, delays, coupling ratios).
- **Per-world variants:** MJWarp reads batched model fields as `field[worldid % batch]`. So worlds are laid out round-robin over the V variants, and only the fields that actually differ get a batch of V. In GENHAND classes these include body and geom poses and sizes, masses and inertias, joint axes and ranges, and actuator gains, delay and force range.
- **Correctness test:** each world of a mixed group follows the CPU MuJoCo trajectory of its own variant (error < 1e-3 over 300 steps).
- **Self-collision pairs:** these differ per variant, since unreachable pairs are pruned per hand. `union_pairs` gives every variant the class's union of pairs; pairs a variant doesn't own get the minimum impedance (`solimp` 1e-4), so they exert no force. A test checks this.
  - Gap won't work: in MJWarp 3.15 a contact is active iff `dist < margin`, and gap only widens detection.
  - Margin won't work: a negative margin is rejected on box/mesh pairs while multi-contact CCD is on, and that's what gives box and mesh contacts multiple points, which stable grasps need.
- **Fields that can't vary per world** must be equal within a group. `signature()` hashes them, and a class whose variants differ is split into several groups.
  - In GENHAND the only such field is **`wrap_prm`, the tendon coefficients**. Every tendon-coupled class splits into one group per variant, 50 extra groups across the 100 benchmarked classes.
  - Other models carry fields that differ but that MJWarp doesn't use (`dof_length` is only for sleeping; the BVH fields aren't in MJWarp's model).

**MultiGroup** (strategy 2): one captured CUDA graph per group, launched sequentially or each on its own CUDA stream.

**Superset** (strategy 3): one maximal hand. Per world, unused fingers are locked (zero-width range) and their geoms and pairs get the minimum impedance. It can only represent hands that are sub-hands of the superset, and every world pays for all 19 DoF.

## Findings

1. **Streams matter:**
   - With sequential launches, 10 groups lose 23% and 100 groups lose 85%: graph launches serialize, and small groups leave the GPU idle.
   - With one stream per group, 10 groups cost nothing.
   - 100 groups (84 worlds each) still lose 45%.
2. **The superset is competitive on cost.** At 19 DoF it runs 0.60 M, against 0.83 M for the average single class, whose hands average about 9 joints. Its problem is coverage, not speed: one superset represents only hands derived from it.
3. **Topology classes are numerous.** GENHAND-1K v4 has 160+ classes, plus one group per tendon-hand variant, and 17 real hands, each of its own structure. Having all of them resident at once means about 100+ groups: the slow end of the table.

## Options (your decision, per the working rules)

- **A. Rotating class subsets (recommended).**
  - Keep about 10–20 groups resident, one stream each. That's the near-free end of the table.
  - Every K PPO iterations, resample which classes are resident, balanced by topology class and real vs procedural (spec, Component 7).
  - Rebuilding groups takes seconds with a warm kernel cache.
  - Cost: diversity within a single batch is limited to 10–20 classes; across training, every class is covered.
- **B. Family supersets.**
  - One superset per layout (anthro, radial, jaw) × tip shape. Inside it, phalanx counts and fingers vary per world through locked joints and shrunken geometry; mimic and synergy coupling vary per world through `eq_active` (world data) and `eq_data`.
  - Covers most GENHAND topologies in about 9 groups.
  - Cost: more engineering, every world pays the maximal DoF, and tendon hands still need fixed coefficients.
- **C. All classes resident (100+ groups).** Simple, but about half the throughput on this GPU.

**Generator option, independent of A/B/C:** make tendon coefficients a topology-level parameter (generator v5), so tendon classes batch like the others instead of splitting into one group per variant. Tendon diversity across classes is unchanged, but variants within a class would share their coefficients.

## Decisions (2026-10-08) and what was added

1. **Strategy A: rotating class subsets** (`mandahands/sim/rotation.py`).
   - Each rotation picks `n_groups` topology classes uniformly over classes, with up to `variants` hands each, and builds one Group per class with one CUDA stream each.
   - Scenes are built once per hand and cached on disk (`data/cache/scenes/`).
   - Tests check that groups never mix classes, that picks are deterministic per rotation and resampled across rotations, and that the real-hand flag works.
2. **Tendon coefficients are domain-randomized per batch** (`groups.tendon_dr`). Every variant of a group shares one sampled set of coefficients: the first variant's, × U(0.8, 1.2) per joint wrap. It's resampled at every rotation.
   - Command ranges and start commands are rescaled to the new coefficients, and derived constants are recomputed.
   - Tendon classes are now batchable like the others; a test confirms it.
   - Per batch rather than per hand, because MJWarp can't vary `wrap_prm` per world.
3. **Real hands are evaluation-only by default.** `RotationCfg.include_real=True` mixes them into training as their own groups, at `real_fraction` of the groups (default 25%).

### A vs C (`scripts/bench_rotation.py`)

| GPU | strategy | worlds | groups | physics steps/s | peak MiB | build s | rotate s |
|---|---|---|---|---|---|---|---|
| RTX 3060 Laptop | A | 8400 | 16 | **0.78 M** | 865 | 23 | 12 |
| RTX 3060 Laptop | C | 8400 | 142 (every train class) | 0.38 M | 701 | 52 | – |
| RTX A5000 | A | 16800 | 16 | **1.82 M** | 1730 | 226 (cold cache) | 47 (cold) |
| RTX A5000 | C | 16800 | 142 | 0.81 M | 1432 | 152 | – |
| RTX A5000 | A | 50400 | 16 | **2.00 M** | 5191 | 9 (warm cache) | 5 (warm) |
| RTX A5000 | C | 50400 | 142 | 1.15 M | 4318 | 42 | – |

- **Cost of a rotation:** "rotate s" is one rotation, mostly building scenes for classes not yet cached (cold cache here). Rotating every few minutes of training costs little.
- **Memory:** A's peak includes building the next rotation while the previous one is still allocated.
- **A5000 confirms A** (run during M6, after the driver was fixed by a reboot into kernel 7.0.0-38): A is 2.2× faster than C at 16800 worlds and 1.7× at 50400, for about 20% more memory. No capacity overflow in any run.
