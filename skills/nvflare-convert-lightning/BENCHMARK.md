# Benchmark Summary

Status: draft/internal pending Milestone 8 two-conversion-skill verification and
Milestone 11 runtime evaluation.

Skill version: 0.1.0
FLARE version: 2.8.0 minimum

## Initial Checks

| Check | Status | Notes |
| --- | --- | --- |
| Positive trigger | Draft | `lightning-convert-basic` covers natural Lightning-to-FL conversion with FedAvg simulation settings and the `flare.patch(trainer)` exchange. |
| DDP/multi-GPU trigger | Draft | `lightning-ddp-multigpu` covers external-process launch and the rank-synchronized round loop. |
| Evaluation-only trigger | Draft | `lightning-eval-only` covers PyTorch FedEval recipe selection with validate/test in the patched loop. |
| Adjacent negative trigger | Draft | `lightning-negative-plain-pytorch` routes plain PyTorch manual loops to `nvflare-convert-pytorch`. |
| Global negative trigger | Draft | `lightning-global-negative-kubernetes` routes Kubernetes deployment away from this skill. |
| Mandatory behavior | Draft | Behavior IDs cover inspect-first, patch-based exchange, model constructor argument auditing, PyTorch recipe discovery and selection from FL intent, scoped edits, standard generated layout, `/tmp/nvflare` runtime outputs, Lightning behavior preservation, external-process DDP launch, rank-synchronized loop, local validation, and validation evidence reporting. |
| Prohibited behavior | Draft | Behavior IDs prohibit manual `FLModel` default exchange, passing `input_model` into the `Trainer`, inventing a Lightning-only recipe, production submit, private data copying, and CLI-wrapper Python. |
| Process metrics | Draft | Metrics cover first-pass acceptance, turns to acceptable, user correction count, layout violations, and validation evidence completeness. |

## Known Gaps

- Repeated runtime agent-performance measurement has not been run yet; the
  Milestone 8 Stage 5 checkpoint compares PyTorch and Lightning conversion
  behavior with Codex and Claude.
- The seed Lightning skill reuses the PyTorch recipe family and does not define
  a Lightning-specific recipe family.
- DDP/multi-GPU behavior is validated through external process launch; SimEnv
  single-process validation covers conversion structure, not distributed
  scaling.
- Export validation uses NVFLARE job system arguments; no job-local export
  argument definition is required.
