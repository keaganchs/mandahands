# Milestone 4: descriptors and tokens

Reproduce: `python scripts/gen_genhand.py && python scripts/descriptor_report.py`, then `pytest tests`.
Code: `mandahands/embodiments/descriptors.py`, `mandahands/policy/tokenizer.py`, `registry.descriptor(id)`.
Outputs: `registry/descriptor_stats_genhand-1k-v4.json` (normalizer), `docs/descriptor_report.md` and `docs/descriptor_tokens.png` (token-level coverage).

## Acceptance

| Check | Result |
|---|---|
| Unit tests | `pytest tests`: 22 pass, 6 of them new (`tests/test_descriptors.py`) |
| Known values on a hand-built toy hand | Joint position and axis, range, depth and joints-below, link mass, damping, actuator gains, forces, span, latency, velocity and backlash, and the mimic coupling row [1, 0.5] |
| Invariance | The toy hand authored with the opposite sign convention, rotated under the wrist, or both, gives identical descriptors, coupling and tokens |
| Tokens batch across hands with different DoF and coupling | A toy hand plus GENHAND hands with full, tendon_pull and synergy coupling (different J and A) in one batch. Checked: shapes, masks, zeroed padding, coupling blocks, and normalized values. Actions: +1 gives the closed command and −1 the open one, and the round trip is exact. Joint state is +1 at each joint's closing limit |
| Normalization | Fitted on the train split; train tokens have mean 0 and std 1 per varying field; save/load round trip |
| Descriptor noise and out-of-support utility | Noise stays within bounds and leaves coupling and axes exact. A hand with 1000× gains is flagged out of support |

## Reference code read first (as the working rules require)

`one_policy_to_run_them_all` @ `7c18194` (URMA, JAX) and `embodiment-scaling-laws` @ `62e43ce` (PyTorch distillation), cloned to `data/refs/`.

- **Description vectors are computed from the model**, per joint and per foot, in the trunk frame:
  - joint position (normalized by the robot's bounding box) and axis;
  - nominal position;
  - torque and velocity limits, damping, armature, stiffness and frictionloss;
  - range;
  - plus robot-level gains, mass and dimensions appended to every joint.
- **Normalization uses fixed hand-set constants** (`x / (max/2) − 1`), not dataset statistics.
- **Descriptor values are the domain-randomized "seen" values**, so the description is noisy during training.
- **Dynamic joint state** is position minus nominal, velocity, and previous action.
- **Joint = actuator.** There is no notion of coupling.
- **Encoder:** a per-joint MLP over the description produces a softmax over 64 latent slots (softmax over slots, not over joints), with a learned temperature. Each joint's state latent is weighted into those slots and **summed over joints**. The core MLP takes that latent plus the general state.
- **Decoder:** shared per joint. Its input is the core latent, the joint's description latent, and its own state latent (with gradient stopped). Log-std comes from the description.
- **Batching is per robot type.** Each type keeps its exact joint count, so joints are never padded or masked inside the policy; only the flat observation and action vectors are padded.
- **ESL** uses the same model with 3 heads and larger widths. It is trained by distilling per-embodiment PPO experts in supervised batches, one embodiment per batch.

What we take, and where we differ:
- **Model-computed descriptors:** taken.
- **Normalization:** dataset statistics with log transforms, as our spec asks. Hands span orders of magnitude in gains and masses, so log scales matter more than for legged robots.
- **Descriptor noise:** taken.
- **Joints vs actuators:** separate joint (observation) and actuator (action) tokens with a coupling matrix. URMA has no answer for underactuated hands.
- **Padding:** pad plus mask. Under URMA's sum aggregation a masked token must contribute zero, which M8's Variant A must enforce. The alternative is per-group batching, which M5's grouping gives anyway.

## Descriptors (`descriptors.py`)

**Joint tokens** (30 fields):
- position, axis and finger-base position;
- slide flag, range (lower, upper, span);
- link mass and bounding-box half extents;
- depth from the palm and number of joints below;
- vector and distance to the reachable fingertip, and fingertip position;
- damping, armature, frictionloss and stiffness.

**Actuator tokens** (18 fields):
- kp and kv;
- closing and opening force limits (one-directional tendons have opening force 0);
- command span, gear and velocity limit;
- latency and backlash;
- number of driven joints, coupling sum, slide fraction, and mean position of the driven joints;
- a provenance one-hot (measured / datasheet / estimated).

**Coupling:** `C[a, j]` gives joint motion per unit of actuator command. It comes from joint gear, tendon coefficients, and linear joint equalities propagated through chains.

Two conventions make tokens comparable across hands (the descriptors are new; the conventions are my design):
- **Canonical hand frame:** origin at the palm's centre of mass; z toward the open fingertips; y the direction the fingertips close toward. Source root frames are arbitrary (real hands' URDF palm frames point anywhere), so source-frame coordinates would not be comparable.
- **Closing-sign convention:** every joint and actuator coordinate is signed so positive = closing, using the probe's open and closed poses. Actions are absolute targets in [−1, 1]: −1 = the open command, +1 = the closed command. The tokenizer applies the same signs to the per-step state.

Spec deviations:
- **No finger index:** finger ordering is arbitrary across hands, so the finger-base position identifies the finger geometrically instead.
- **Touch is a placeholder for now:** each joint token has touch and touch-mask slots, always zero with mask 0, until M7 adds touch sensors.
- **Robotiq's linkage isn't in the coupling matrix:** its four-bar closure (connect constraints) can't be written as ratios, so its passive joints have no coupling entry. The hand is flagged `closed_loop_not_in_coupling`; four-bar grippers are backlog B2.

## Coverage: real hands vs train, token level

Out of support means a hand's mean nearest-neighbour distance is above the 99th percentile of train hands' own leave-one-hand-out distances, for joints or for actuators.

| | hands | out of support |
|---|---|---|
| test_id | 75 | 0 |
| test_ood_topology | 83 | 1 |
| test_ood_scale | 131 | 12 (9%) |
| **real** | 18 | **9** |

- **Out of support:** JAXON, JVRC-1, LEAP, R2, Robotiq, Romeo, TALOS, Allegro, Dex3-1.
- **Inside:** Shadow, Dex5-1, Dex2-5, Dex1-1, both Inspire hands, BrainCo, iCub, Valkyrie.
- **Remaining field gaps:** finger-base and fingertip positions, axis tilt, link mass and ranges. Per-field shares are in `descriptor_report.md`.
- **Hand level (M3 summary features):** 13 of 18 real hands are now in support (Romeo, R2, JAXON, JVRC and Robotiq are not).

Getting from 0 of 18 real hands in support to 9 took two kinds of fix. Each was found by attributing nearest-neighbour distance to fields.

**Descriptor fixes:**
- The finger base was resolved wrongly for hands whose palm link hangs under the mount.
- The origin was the source root frame; it's now the palm centre.
- Unknown real-hand latency and backlash were stored as 0; they now get priors.
- URDF placeholder velocities (1000 rad/s) were taken at face value; anything over 50 rad/s is now unknown.
- Unlimited mimic followers read as ±π; they now take the range implied by their leader.

**Generator fixes (v3, v4):**
- 20% of hands have no servo kv and damp in the joints instead (Menagerie style).
- 30% have no joint damping and 50% no frictionloss: real models mostly leave these unset.
- Abduction now reaches ±1 rad.
- Procedural actuators now get a provenance label: one per hand, seeded by its id. In training, actuator descriptor noise depends on it (±3% measured, 10% datasheet, 30% estimated).

GENHAND is now **genhand-1k-v4**: 965 of 1600 valid (39.7% rejected), with train 676, test_id 75, test_ood_topology 83 and test_ood_scale 131. The M3 conclusions are unchanged; `genhand_report.md` is regenerated. The superseded manifests v1–v3 are removed.

## Notes for later milestones

- **M5:** the tokenizer pads to the largest J and A in a table. Per-group tables (one per topology class) avoid padding inside a group.
- **M7 (env):**
  - gather each embodiment's joint qpos, qvel, ctrl, actuator length and force in descriptor order and call `TokenTable.joint_state` / `act_state`;
  - apply `noisy()` per episode;
  - add touch sensors;
  - simulate velocity limits and backlash.
- **M8 (Variant A):** masked tokens must contribute exactly zero to URMA's sum over joints.
- **M11 (sysid):** real hands' latency and backlash are priors until identified.

## Open questions

1. **The two conventions:** are you OK with the canonical frame and the "+1 = closed" action convention? They change how actions are defined for every hand. I think this falls within the descriptor design rather than the hierarchy interface, but it's your call.
2. **Remaining real-hand gaps:** after the backlog mechanisms (B2), most remaining gaps come from geometry: finger-base layout, axis tilt and link mass. Should the generator be pushed further now, e.g. tilted joint axes and heavier links, or should we revisit after M7 when we can measure whether the gap actually hurts transfer?
