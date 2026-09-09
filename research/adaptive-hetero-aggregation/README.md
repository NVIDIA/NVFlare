# Adaptive Heterogeneity-Aware Aggregation

## Abstract

This draft research implementation studies a conservative client-weighting policy for heterogeneous
federated learning. The policy starts from NVFlare's native local-step weighting and applies a
bounded adaptive correction only after persistent client-distribution heterogeneity and
client-performance disparity are both observed. The implementation is related to
[NVIDIA/NVFlare issue #5209](https://github.com/NVIDIA/NVFlare/issues/5209).

The pull request remains a draft. Engineering behavior is covered by focused tests and NVFlare
integration paths, while the maintainer-requested CIFAR-10 comparison and public method write-up
must still be completed before this should be treated as validated research evidence.

## Papers and Links

- NVFlare issue: [#5209](https://github.com/NVIDIA/NVFlare/issues/5209)
- Pull request: [#5273](https://github.com/NVIDIA/NVFlare/pull/5273)
- Preprint/workshop paper: not yet published; this remains a requirement for research acceptance.

## Objective

The objective is to test whether a federated server can preserve native aggregation under ordinary
conditions while giving controlled additional influence to clients that contribute persistently
under-represented data distributions and exhibit a material performance gap. The implementation is
opt-in and falls back to native weighting rather than forcing adaptive behavior.

## Method Summary

For each participating client `i`, the adaptive policy receives:

- an actual positive training-example count `n_i`;
- a non-negative distribution descriptor;
- a normalized higher-is-better client metric in `[0, 1]`;
- an optional quality-improvement value.

The adapters separately retain NVFlare's native local-step weighting signal from
`NUM_STEPS_CURRENT_ROUND`. `NUM_STEPS_CURRENT_ROUND` is the number of completed local optimizer
steps, not the number of training examples.

The adaptive candidate is conceptually:

```text
w_adaptive ∝ n_i^gamma * representation_i^beta * fairness_i * quality_i^delta
```

Client metrics are reliability-adjusted using actual sample counts before the performance-disparity
gate and fairness pressure are calculated:

```text
reliability_i = n_i / (n_i + metric_prior_strength)
adjusted_metric_i = reliability_i * metric_i + (1 - reliability_i) * federation_mean
```

Two gates control whether adaptation is eligible:

```text
candidate_blend = max_blend * heterogeneity_gate * performance_gap_gate
```

Activation is stateful. The default policy requires a warm-up period, consecutive qualifying
rounds, and a stable participating cohort. When adaptation is inactive, native local-step weighting
is preserved exactly. When adaptation is active:

```text
blended = (1 - blend) * w_native + blend * w_adaptive
final_weight = bounded_simplex_projection(blended)
```

If configured bounds are infeasible for the current cohort size, the round conservatively falls
back to native aggregation instead of failing.

### Default safeguards

```text
sample_exponent               = 0.65
representation_exponent       = 0.70
quality_exponent              = 0.0
fairness_strength             = 1.0
metric_prior_strength         = 100.0
heterogeneity_threshold       = 0.26
heterogeneity_temperature     = 0.04
heterogeneity_deadband        = 0.15
performance_gap_threshold     = 0.10
performance_gap_temperature   = 0.03
performance_gap_deadband      = 0.05
max_blend_factor              = 0.20
activation_warmup_rounds      = 3
activation_patience           = 2
require_stable_cohort         = True
min_weight                    = 0.0
max_weight                    = 1.0
```

The default bounds are intentionally unconstraining so they remain feasible for any positive cohort
size. Tighter bounds can be configured for experiments with a known client count.

## NVFlare Integration

The weighting policy is independent of the server optimizer. Two adapters expose the same policy:

- `AdaptiveHeterogeneityAggregator` implements the Shareable/DXO `Aggregator` interface;
- `AdaptiveHeterogeneityModelAggregator` implements the unified `ModelAggregator` interface used by
  the current FedAvg recipe.

The method is therefore presented as a generic aggregation policy rather than a FedOpt-specific
algorithm. The FedOpt path remains useful as an integration/reference smoke path.

Each adaptive contribution supplies:

```python
from nvflare.apis.dxo import MetaKey
from adaptive_hetero.nvflare_aggregator import AdaptiveMetaKey

dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, num_local_optimizer_steps)
dxo.set_meta_prop(AdaptiveMetaKey.SAMPLE_COUNT, num_training_examples)
dxo.set_meta_prop(AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR, descriptor)
dxo.set_meta_prop(AdaptiveMetaKey.CLIENT_METRIC, validation_accuracy)
dxo.set_meta_prop(AdaptiveMetaKey.QUALITY_IMPROVEMENT, baseline_loss - final_loss)
```

`NUM_STEPS_CURRENT_ROUND` is used only for native NVFlare weighting. Sample-based representation and
metric reliability use `AdaptiveMetaKey.SAMPLE_COUNT`.

Per-site final weights remain server-side and are not copied into aggregated model metadata, so a
participant does not receive every other participant's final aggregation weight through the global
model.

If no Shareable/DXO contribution is accepted, the adapter returns `ReturnCode.EMPTY_RESULT`. The
unified `ModelAggregator` returns an empty DIFF no-op in the analogous empty-round case.

## Current Engineering Validation

Focused checks cover:

- bounded-simplex projection and final active weight bounds;
- exact native fallback behavior;
- separate optimizer-step and sample-count semantics;
- metric reliability based on actual sample counts;
- warm-up, activation patience, and cohort-reset behavior;
- one-client and large-cohort bound feasibility;
- malformed or missing adaptive metadata;
- empty-round behavior for both aggregator interfaces;
- server-side-only per-site weight diagnostics;
- custom constructor argument serialization through exported NVFlare jobs;
- unified FedAvg `ModelAggregator` serialization;
- real `DXO`, `Shareable`, `FLContext`, `FLModel`, `WeightedAggregationHelper`, and Recipe paths;
- a lightweight `FedOptRecipe + SimEnv` smoke path;
- a real FedCE protocol smoke path;
- synthetic and scikit-learn digits development benchmarks;
- CIFAR experiment split/result helper tests.

These checks establish implementation behavior only. They are not final scientific evidence.

No project-specific GitHub Actions workflow is included. The research workload stays inside this
project directory and is run intentionally rather than gating unrelated repository merges.

## Standard CIFAR-10 Evaluation

`cifar10_evaluation/` implements the non-IID evaluation requested by the maintainers. It reuses
NVIDIA FLARE's `ModerateCNN`, CIFAR-10 utilities, Dirichlet splitter, recipes, and algorithm helpers.

The supported methods are:

- FedAvg;
- FedOpt as an optional full-participation reference;
- FedProx;
- SCAFFOLD;
- FedCE;
- adaptive heterogeneity-aware aggregation.

### Train/validation/test separation

For each `(n_clients, alpha, seed)`:

1. NVIDIA FLARE's standard Dirichlet splitter assigns CIFAR-10 **training** examples to sites.
2. Each site's assignment is deterministically split into a training subset and a held-out local
   validation subset; the default validation fraction is 10%.
3. Every compared method uses the same site training subset and the same held-out validation subset.
4. Adaptive performance metadata and FedCE leave-one-out metrics use only the held-out training
   validation subset.
5. The official CIFAR-10 **test set is not read during federated training**.
6. After training, the common evaluator loads the final server checkpoint and evaluates the test set.

Matched FedAvg/FedOpt/FedProx/SCAFFOLD clients preserve the corresponding NVIDIA FLARE algorithmic
training behavior while replacing per-round test-set evaluation with the held-out training
validation subset.

### Common post-training evaluator

`eval_split.py` creates deterministic site-specific partitions of the untouched CIFAR-10 test set.
Their class mixtures follow the original Dirichlet training assignments.

`evaluate_result.py` evaluates the final server checkpoint from every method using the same protocol:

- `global_accuracy`: accuracy on the complete CIFAR-10 test set;
- `client_accuracies`: accuracy on each site-specific test partition;
- `worst_client_accuracy`: minimum client accuracy;
- `mean_client_accuracy`, best-client accuracy, and client-accuracy gap.

This keeps the headline FedAvg/FedProx/SCAFFOLD/FedCE/adaptive result definitions identical.

### Reproducibility and pairing

The experiment seed controls:

- the Dirichlet assignment;
- the deterministic train/validation split;
- server-model initialization;
- each site's PyTorch/NumPy client RNG seed.

The result record therefore declares pairing by Dirichlet split, server initialization, and client
seed. Different algorithms can still consume randomness differently, so reproducibility does not
imply identical optimization trajectories.

### Protocol versioning

Every completed result row is tagged with a protocol version. The campaign's resume logic and the
summary script ignore rows from an older protocol version so stale experiments cannot silently enter
new confidence intervals.

### Confidence intervals

`summarize_results.py` reports, for global and worst-client accuracy:

- sample size;
- mean;
- sample standard deviation;
- two-sided 95% Student-t confidence interval.

It also reports paired adaptive-minus-baseline confidence intervals for common seeds.

## Setup

This research project uses APIs on the current NVIDIA FLARE `main` branch, including the unified
FedAvg `ModelAggregator` path. Until those APIs are part of a published release, install NVFlare from
this repository.

From the NVFlare repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r research/adaptive-hetero-aggregation/requirements.txt
```

The full CIFAR-10 campaign is substantially faster with CUDA-capable GPUs. The lightweight unit,
synthetic, and digits checks can run on CPU.

## Data Preparation

No dataset is committed to this project.

On first use, torchvision downloads CIFAR-10 under:

```text
/tmp/cifar10
```

The configured `--split_root` contains four deterministic products for a condition:

- the original Dirichlet training assignment;
- the common per-site training subset;
- the common per-site held-out training validation subset;
- the site-specific final-test partitions.

For a fixed `(n_clients, alpha, seed, validation_fraction)`, all compared methods reuse the same
split directories.

## Run Instructions

From `research/adaptive-hetero-aggregation/`:

### Focused tests

```bash
PYTHONPATH=src pytest -q tests
```

### Lightweight NVFlare smoke

```bash
python nvflare_smoke/job.py \
  --n_clients 3 \
  --num_rounds 5 \
  --train_samples 180 \
  --valid_samples 90 \
  --local_epochs 1
```

### One CIFAR-10 run

```bash
python cifar10_evaluation/run.py \
  --method adaptive \
  --alpha 0.1 \
  --seed 7 \
  --participation_rate 1.0 \
  --results_jsonl results/cifar10_runs.jsonl
```

A completed run evaluates the final server checkpoint, prints one `CIFAR10_EVAL_RESULT` JSON record,
and optionally appends it to the supplied JSONL file.

### Maintainer-requested comparison campaign

The default campaign is:

- methods: FedAvg, FedProx, SCAFFOLD, FedCE, adaptive;
- Dirichlet alpha: `0.1` and `0.5`;
- seeds: `7, 19, 31, 43, 57`;
- participation: `1.0` and `0.75`;
- 8 clients, 50 rounds, 4 local epochs;
- 10% held-out validation from each site's CIFAR-10 training assignment.

Run it intentionally with:

```bash
python cifar10_evaluation/run_campaign.py
```

Inspect the matrix without training:

```bash
python cifar10_evaluation/run_campaign.py --dry_run
```

The campaign is resumable. A current-protocol `(method, alpha, participation, seed,
validation_fraction)` row is skipped when it is already present in `results/cifar10_runs.jsonl`.
Use `--fresh` to start a new result file.

Add FedOpt as a full-participation reference with:

```bash
python cifar10_evaluation/run_campaign.py \
  --methods fedavg fedopt fedprox scaffold fedce adaptive
```

Generated artifacts are written to:

- NVFlare simulation workspaces: `/tmp/nvflare/adaptive_hetero_cifar10` by default;
- data splits: `/tmp/cifar10_splits/adaptive_hetero_eval` by default;
- run rows: `results/cifar10_runs.jsonl` by default;
- confidence-interval summary: `results/cifar10_summary.json` by default.

## Expected Results

No final CIFAR-10 numerical performance claim is checked in yet. A successful run produces a
`CIFAR10_EVAL_RESULT` record containing at least:

```text
protocol_version
method
alpha
participation_rate
seed
validation_fraction
global_accuracy
worst_client_accuracy
client_accuracies
checkpoint
test_data_used_during_training = false
```

A completed campaign should produce confidence-interval tables covering FedAvg, FedProx, SCAFFOLD,
FedCE, and adaptive aggregation under full and partial participation. Neutral and negative results
must be retained alongside improvements.

Previous checked-in development numbers were removed because they predated the current policy and
included blend factors that are unreachable with the current `max_blend_factor=0.20` setting.

## Evidence Still Required Before Research Acceptance

The requested implementation and evaluation infrastructure is present, but final evidence is not yet
claimed. Before the draft should be considered merge-ready under `research/`, it still needs:

1. completed matched CIFAR-10 runs against FedProx, SCAFFOLD, and FedCE;
2. confidence-interval tables generated from those completed runs;
3. partial-participation results in the main result tables;
4. transparent reporting of neutral or negative results;
5. a public method write-up such as a preprint or workshop paper.

## Existing Development Benchmarks

`benchmark.py` uses `sklearn.datasets.make_classification` for fast synthetic checks.

`digits_benchmark.py` uses scikit-learn's `load_digits` dataset and supports linear/MLP models,
multiple seeds, and full or partial participation. These remain development tools rather than
headline evidence.

## Relationship to Existing NVFlare Work

### FedAvg and FedOpt

The method is an aggregation policy rather than a server optimizer. The unified FedAvg path uses
`AdaptiveHeterogeneityModelAggregator`; the Shareable/DXO path uses
`AdaptiveHeterogeneityAggregator`. Both delegate weighting to the same policy.

### FedProx and SCAFFOLD

FedProx and SCAFFOLD address non-IID optimization drift through client/local optimization changes.
They are included as standard heterogeneity baselines.

### FedCE

FedCE estimates client contribution using update-direction and leave-one-out validation behavior.
`fedce_client.py` uses the real NVFlare FedCE helper contract and the same held-out training
validation protocol as the other methods.

### Auto-FedRL

Auto-FedRL learns aggregation behavior with reinforcement learning. It remains relevant context but
is not part of the maintainer-requested minimum comparison set for this draft.

## Privacy and Trust Considerations

Distribution descriptors, sample counts, and client metrics are summary metadata; they are not
privacy-preserving by default. Depending on the application, they may disclose information about
local data and may require approved summaries, secure aggregation, differential privacy, or another
privacy mechanism.

The server assumes supplied metadata is trustworthy and comparable across clients. Production use
would require an application-specific metadata contract and appropriate privacy and integrity
controls.

## Remaining Limitations

The method has not yet been established as broadly effective. Remaining work includes completing the
standard CIFAR-10 campaign, evaluating larger client populations and additional model/dataset
settings, quantifying runtime/communication overhead, and validating deployment-specific privacy and
metadata-trust controls.

No universal convergence, production-readiness, or universal accuracy-improvement claim is made.

## Repository Layout

```text
adaptive-hetero-aggregation/
|-- README.md
|-- benchmark.py
|-- digits_benchmark.py
|-- requirements.txt
|-- cifar10_evaluation/
|   |-- __init__.py
|   |-- adaptive_client.py
|   |-- baseline_sgd_client.py
|   |-- eval_split.py
|   |-- evaluate_result.py
|   |-- fedce_client.py
|   |-- fedprox_client.py
|   |-- local_data.py
|   |-- protocol.py
|   |-- run.py
|   |-- run_campaign.py
|   |-- scaffold_client.py
|   `-- summarize_results.py
|-- fedce_smoke/
|-- nvflare_smoke/
|-- src/adaptive_hetero/
|   |-- __init__.py
|   |-- model_aggregator.py
|   |-- nvflare_aggregator.py
|   `-- policy.py
`-- tests/
    |-- test_cifar_evaluation.py
    |-- test_model_aggregator.py
    |-- test_nvflare_aggregator.py
    |-- test_nvflare_smoke_client.py
    `-- test_policy.py
```

## Requirements

Project dependencies are listed in `requirements.txt`. This draft is developed against the current
NVIDIA FLARE `main` branch used by PR #5273 because it relies on current Recipe and
`ModelAggregator` behavior.

## License

The contribution is Apache License 2.0, consistent with NVIDIA FLARE. CIFAR-10 is downloaded at run
time and is not redistributed by this project. Third-party Python dependencies and downloaded data
remain subject to their respective licenses and terms.

## Citation

A preferred citation/BibTeX entry will be added when the public preprint or workshop paper required
for research acceptance is available. Until then, this draft should be referenced by the NVFlare
issue/PR rather than cited as a published method.
