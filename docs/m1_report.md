# Milestone 1: skeleton and benchmark

## Correction (2026-10-07, found during M2)

**The M1 test hand barely moved.** `MjSpec()` defaults to degrees, so its joint limits of [-0.4, 1.2] and [0, 1.5] rad were read as degrees: about 0.02 rad of travel. The M1 "grasp" was a near-rigid pinch of fingers that already almost touched the box. The benchmark and equivalence numbers below therefore describe a stiffer, more contact-heavy scene than intended. Fixed with `compiler.degree = False` in every MjSpec we build.

Other scene changes since M1 (details in `m2_report.md`): integrator `implicitfast` (was Euler), `impratio` 3 (was 10, confirmed by you), and one standardized contact model for hand geoms.

**Re-run, RTX A5000, njmax=160** (`results/bench_NVIDIA_RTX_A5000.md`):

| nworld | SPS (re-run) | MiB | SPS (original) |
|---|---|---|---|
| 256 | 0.59 M | 45 | 0.54 M |
| 1024 | 1.77 M | 182 | 1.37 M |
| 4096 | 2.99 M | 726 | 1.97 M |
| 16384 | 3.35 M | 2904 | 2.13 M |
| 32768 | 3.38 M | 5809 | 2.13 M |
| 65536 | 3.37 M | 11617 | 2.12 M |

At 4096 worlds:

| Axis | Values | SPS (re-run) | Notes |
|---|---|---|---|
| object convex parts | 1 / 4 / 16 | 2.34 / 1.80 / 1.49 M | |
| hand DoF | 6 / 12 / 20 | 2.93 / 2.64 / 0.98 M | |
| solver iterations | 5 / 10 / 20 | 2.94 / 2.95 / 2.95 M | Iteration cap still doesn't affect speed |
| nconmax | 16 / 32 / 64 | 2.99 / 2.96 / 2.93 M | Memory only |

After M2 added limited self-collision (6 frictionless fingertip pairs on this hand), the plateau is 3.29 M (−3%), and 4096 worlds give 2.90 M; the 20-DoF hand gives 1.22 M.

Without those pairs, the plateau is about 3.4 M physics steps/s, roughly 67k control steps/s at 50 Hz. The closed grasp settles at 6 contacts per world instead of 12, which is the main reason it's faster. No capacity overflow at njmax=160. A new signal: the line search often hits its iteration cap (`LS_ITERATIONS`). That's a convergence flag, not lost constraints. Per your instruction, iterations stay at 10 until there's a real need.

**Equivalence test, revised.** With fingers that actually move, CPU and MJWarp trajectories agree to about 1e-10 until contact onset, then diverge (float32 vs float64 timing of contact events). The test is now two checks:
1. Free-space trajectory match with the object out of reach: observed 6.5e-7 rad over 2 s, tolerance 1e-5.
2. Same close-and-lift outcome with contact: both lift the object to 0.1129 m, tolerance 1.5 cm.

Both pass on the 3060 and the A5000.

---

*Original M1 report below, unchanged.*

Stack: `mujoco==3.15.0`, `mujoco-warp==3.15.0`, `warp-lang==1.18.0`, Python 3.12.
Scene: a floating wrist (free body welded to a mocap target with `solref=(0.02, 1)`), a test hand with 3 fingers × 2 phalanges (6 position actuators), and a 6 cm box on a plane. Elliptic cones, `impratio=10`, `condim=4` on the object, 2 ms timestep.

Reproduce: `python scripts/bench.py --njmax 160` (raw output goes to `results/`), and `pytest tests`.

## Acceptance

| Check | Result |
|---|---|
| Benchmark: steps/s and memory for ≥3 world counts | 6 counts on the A5000 and 4 on the 3060 (tables below) |
| MJWarp vs CPU MuJoCo equivalence (`tests/test_sim.py`) | Passes on both GPUs. Over a 2 s close-and-lift, max error is about 1e-4 rad on hand joints and 1e-5 m on object and wrist position; the test tolerances are 10× that. Both backends lift the object 10 cm. |
| Overflow detection fails loudly | `check_overflow` raises `ContactOverflow`, tested with `njmax=8` |

## Throughput (physics steps/s summed over worlds, closed-grasp steady state, njmax=160)

| nworld | A5000 SPS | A5000 MiB | 3060 Laptop SPS | 3060 MiB |
|---|---|---|---|---|
| 256 | 0.54 M | 45 | 0.42 M | 45 |
| 1024 | 1.37 M | 181 | 0.71 M | 181 |
| 4096 | 1.97 M | 725 | 0.81 M | 725 |
| 16384 | 2.13 M | 2898 | 0.83 M | 2898 |
| 32768 | 2.13 M | 5796 | – | – |
| 65536 | 2.12 M | 11592 | – | – |

Throughput on the A5000 saturates at about 2.1 M physics steps/s from 16k worlds. That is about 42k control steps/s at 50 Hz (decimation 10 at 2 ms per physics step), and about 0.18 MiB per world.

Other axes, at 4096 worlds on the A5000:

| Axis | Values | SPS | Notes |
|---|---|---|---|
| solver iterations | 5 / 10 / 20 / 50 | 1.92 M at every cap | The solver exits early, so the cap only affects accuracy. At 5 it hits the cap. |
| nconmax | 16 / 32 / 64 / default (48) | 1.93–1.92 M | Memory only: 330 to 922 MiB. Peak use is 12 contacts per world. |
| object convex parts (mesh) | 1 / 4 / 16 | 2.85 / 2.25 / 1.50 M | A 16-part object costs about half of the 1-part throughput. Supports a cap of ≤16 parts. |
| hand DoF | 6 / 12 / 20 | 1.93 / 1.87 / 1.09 M | The 20-DoF case (5 fingers × 4 phalanges) is the expensive one. |

Full tables: `results/bench_NVIDIA_RTX_A5000.md` and `results/bench_NVIDIA_GeForce_RTX_3060_Laptop_GPU.md`.

## Findings to report

1. **MJWarp's default `njmax` (64) is too small.** The 16-part object and the 12- and 20-DoF hands overflowed constraint rows (`NEFC`) in every world at the default, and that also stopped the solver converging (`ITERATIONS`). The overflow flag is sticky, so it catches transient peaks between samples. Capacity has to be sized per group from measurements, as the spec already planned. 160 is clean for everything tested.
2. **116 Model fields can be batched per world** (`put_model(batch_sizes=...)`). These include `geom_size/pos/quat`, `body_pos/quat/mass/inertia`, `jnt_axis/range/pos`, `actuator_gainprm/biasprm/delay`, and `geom_dataid` (a different mesh per world). Topology-class grouping (strategy 1) therefore looks able to vary geometry, not just physics, within one compiled model. Milestone 5 will verify this.
3. **A box primitive is slower than a 1-part mesh box** (1.9 M vs 2.85 M SPS): capsule–box collision produced 12 contacts versus 4. This needs revisiting when the object library defines its primitives.
4. **MjSpec's default weld `data` is not zero.** It gives anchor and relpose of (0, 1, 0), which pinned the wrist. The builder now sets `data` explicitly.
5. **The 2-finger variant of the test hand does not hold the box.** It doesn't matter for M1, but it previews the validity-filter work in M3.

## Not done / open questions

- **Cluster:** Sinabox (`pink_cluster`, 2× A5000) has no `sbatch` or `det` on PATH, so the SLURM and Determined launch scripts are deferred to Milestone 7 cluster integration. Which GPU type do the real training clusters use?
- **Config:** a `SceneCfg` dataclass for now. Hydra or a full config tree comes when there is more than one consumer.
- **Solver iterations:** 10 converges for the 6-DoF hand. I'd default to 20 for the bigger hands, since it costs nothing measurable here.
