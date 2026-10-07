# Milestone 1: skeleton and benchmark

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
