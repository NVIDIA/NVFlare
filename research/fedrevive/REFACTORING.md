# FedRevive composition

This is Stage 5 of the [FedRevive roadmap](https://github.com/NVIDIA/NVFlare/issues/5572).
It uses the shared [FedBuff recipe](https://github.com/NVIDIA/NVFlare/pull/5573),
[knowledge distillation and synthesis helpers](https://github.com/NVIDIA/NVFlare/pull/5613),
and [reproducible SimEnv](https://github.com/NVIDIA/NVFlare/pull/5780) from NVFlare.
The [README](README.md) covers data preparation, CLI options, and experiment artifacts.

## What FedRevive composes

- **Global updates.** `FedBuffTrainingCoordinator` computes the client delta from its
  assignment base. `FedReviveUpdate` adds the arriving full model to the teacher
  buffer before revival, refreshes retained proxies when the client's second
  upload finalizes its estimate, and blends in a KD delta computed against the
  current global model. The synthesis gate opens at current version 50. The
  blend uses a zero KD delta through current version 100, so the client delta
  is still scaled by its staleness weight. A separate FedAvg transform checks
  that each publication uses contributions from a single base version.
- **Execution.** Every arrival mode uses `FedBuffRecipe` with the same model
  operations, observer, and update transform. `SimEnv` preflights a supplied or
  generated plan before training and compares actual logical events with the
  preflight during execution.
- **Paper scheduling.** The reference scheduler releases a logical client after
  its training phase, so it can receive a new assignment while an earlier upload
  is pending. The paper planner uses `FedBuffScheduleState` to generate ordered
  events and expected base versions before launch. It preserves the NumPy
  `RandomState` draw order, heap tie handling, the simulated-time cutoff, and the training seed formula
  `run_seed * 1_000_000 + assignment_id + 1`.
- **Distillation and synthesis.** The shared teacher buffer, ordinary KD loss,
  feature-statistics objective, and mean-probability probe supply reusable
  primitives. FedRevive owns its conditional generator, learned latent and
  embedding tensors, Reptile meta-update, image pool and augmentation, and the
  per-teacher batch sampling and weighting before each KD optimizer step.
  Those policies differ from the generic synthesizer's fixed latent batch and
  the generic trainer's shared teacher batch. The synthesis KL retains gradients
  through teacher predictions to the generated inputs.

## Behavioral differences from the reference workflow

The shared feature objective groups each layer's mean and variance losses before
summing. In one ResNet CPU comparison, the scalar changed from 299.679016 to
299.679047 with identical input gradients. Timed replay with preflight fails if a
planned acceptance has a physical call error. Errors outside the planned
acceptances are logged whether they arrive during replay or drain.
The simulator reclaims temporary uploads and bases after their last consumer
finishes. The run retains the plan, verified traces, metrics, and final model.

## Historical migration checks

On the earlier dependency checkout, seeded scheduling for all three presets
matched the reference through 110 small-model publications and a time cutoff.
Assignment IDs, logical clients, bases, timestamps, seeds, and final parameters
matched across one and two workers. In a separate two-teacher CPU check, one
synthesis step and two KD steps matched reference deltas, generator state, and
losses. Pool samples, RNG continuation, and two-upload proxy freezing also
matched. See the [Stage 5 experiment record](https://github.com/NVIDIA/NVFlare/issues/5572).

The earlier real-data CUDA comparison used the CIFAR-10 experiment with seed 10,
two physical workers, and one RTX 6000 Ada. It compared pre-migration
[`474c0e276`](https://github.com/ZiyueXu77/NVFlare/commit/474c0e27608ebda5457b92ca58a25ce565c1cf92)
with the composed workflow at
[`a85d0771b`](https://github.com/ZiyueXu77/NVFlare/commit/a85d0771b4adab6e07d1f2ae2c727bef3df31a35).
Both reached 250 publications through the synthesis and KD warmup gates.
All accepted assignments matched in identity, base, and simulated time at the
legacy log's 0.01-unit precision. Final test accuracy was 20.89% before migration
and 20.76% after it. The largest difference across 26 evaluations was 0.42
percentage points. This was one bounded comparison on the earlier revisions.
Full paper-length, multi-seed, and current-head CUDA validation remain outside
that evidence.
