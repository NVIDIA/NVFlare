# Benchmark Summary

Status: draft/internal pending Milestone 11 runtime evaluation.

Skill version: 0.1.0
FLARE version: 2.8.0 minimum

## Initial Checks

| Check | Status | Notes |
| --- | --- | --- |
| Positive trigger | Draft | `pytorch-convert-basic` covers natural PyTorch-to-FL conversion with FedAvg simulation settings. |
| POC handoff trigger | Draft | `pytorch-approved-poc-handoff` covers explicit post-simulation POC approval with a provided workspace. |
| Iterative rerun trigger | Draft | `pytorch-iterative-rerun` covers scoped hyperparameter and recipe changes followed by validation. |
| Recipe search trigger | Draft | `pytorch-recipe-search` covers bounded recipe comparison without unmeasured best-recipe claims. |
| Data distribution trigger | Draft | `pytorch-data-distribution-rerun` covers IID/non-IID split experiments with comparable result reporting. |
| Dataset URL rerun trigger | Draft | `pytorch-dataset-url-rerun` covers repeating experiments with a user-provided dataset URL. |
| Synthetic data trigger | Draft | `pytorch-synthetic-site-data` covers deterministic synthetic per-site generation and validation. |
| Site heterogeneity trigger | Draft | `pytorch-site-specific-training` covers per-site speed and hyperparameter simulation. |
| Adjacent negative trigger | Draft | Lightning prompt routes away from this skill. |
| Global negative trigger | Draft | Kubernetes deployment prompt routes away from this skill. |
| Mandatory behavior | Draft | Behavior IDs cover inspect-first, model constructor argument auditing, natural request parsing, recipe discovery, recipe selection from FL intent, scoped edits, standard generated layout, `/tmp/nvflare` runtime outputs, Client API exchange, local validation, validation evidence reporting, and failed-validation eval records. |
| Prohibited behavior | Draft | Behavior IDs prohibit production submit, private data copying, and CLI-wrapper Python. |
| Process metrics | Draft | Metrics cover first-pass acceptance, turns to acceptable, user correction count, layout violations, and validation evidence completeness. |

## Observed Process Runs

| Case | Status | Process Observation | Notes |
| --- | --- | --- | --- |
| AMES PyTorch FedAvg conversion | Informal/ad hoc | Required user correction | Functional conversion and simulation completed, but first pass used `fl_train.py` instead of `client.py`, wrote export/workspace artifacts under the project root, and required user correction. The skill was updated to add standard-layout, `/tmp/nvflare` output, and result-evidence guardrails. |

## Known Gaps

- Repeated runtime agent-performance measurement has not been run yet.
- The AMES row above is an informal process observation from a real conversion
  exercise, not a formal score.
- The seed skill targets standard PyTorch loops only, not Lightning,
  Hugging Face Trainer, TensorFlow, XGBoost, sklearn, or custom NumPy loops.
- Export validation uses NVFLARE job system arguments; no job-local export argument
  definition is required.
