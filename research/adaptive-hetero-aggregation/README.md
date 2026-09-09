# Adaptive Heterogeneity-Aware Aggregation

This draft research implementation addresses the adaptive client-weighting use case discussed in
[NVIDIA/NVFlare issue #5209](https://github.com/NVIDIA/NVFlare/issues/5209).

The weighting policy is independent of the server optimizer. Two adapters expose the same policy to
NVFlare workflows:

- `AdaptiveHeterogeneityAggregator` implements the Shareable/DXO `Aggregator` interface used by
  workflows such as the FedOpt integration in this project;
- `AdaptiveHeterogeneityModelAggregator` implements the unified `ModelAggregator` interface used by
  the current FedAvg recipe.

The pull request is intentionally kept in draft while the standard non-IID evaluation and public
method write-up are completed. It should not yet be treated as a validated NVIDIA FLARE research
example or as production evidence.

## Motivation

Native weighted aggregation is a strong default, but a client with much larger local training
volume can dominate when participating data distributions differ substantially. Always applying
fairness-oriented weighting can create the opposite problem by changing aggregation even when
clients are already behaving similarly.

The policy therefore separates two questions:

1. Is there persistent evidence of both distribution heterogeneity and performance disparity?
2. If so, how should client influence be adjusted conservatively?

## Method

For each participating client `i`, the adaptive policy receives:

- an actual positive training-example count `n_i`;
- a non-negative distribution descriptor;
- a normalized higher-is-better client metric in `[0, 1]`;
- an optional quality-improvement value.

The NVFlare adapters separately retain the framework's native local-step weighting signal from
`NUM_STEPS_CURRENT_ROUND`. This distinction is important: `NUM_STEPS_CURRENT_ROUND` is the number
of completed local optimizer steps, not the number of training examples.

The adaptive candidate is conceptually:

```text
w_adaptive ∝ n_i^gamma * representation_i^beta * fairness_i * quality_i^delta
```

The representation term combines distribution novelty with representation of less common mass in
the federation reference distribution. Client metrics are reliability-adjusted using actual sample
counts before the performance-disparity gate and fairness pressure are calculated.

Two gates control whether adaptation is eligible:

```text
candidate_blend = max_blend * heterogeneity_gate * performance_gap_gate
```

Activation is stateful. The default policy requires a warm-up period, consecutive qualifying
rounds, and a stable participating cohort. When adaptation is inactive, the adapter preserves the
native NVFlare local-step weighting exactly.

When adaptation is active:

```text
blended = (1 - blend) * w_native + blend * w_adaptive
final_weight = bounded_simplex_projection(blended)
```

Configured weight bounds are applied to the final active weights. If user-configured bounds become
infeasible for the current cohort size, the round conservatively falls back to native aggregation
rather than failing the job.

## Default Safeguards

The current policy defaults are:

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
size. Experiments may configure tighter bounds when the client count is known. Quality-based
weighting is neutral by default (`quality_exponent=0.0`).

### Metric reliability

Reliability uses the explicit client training-example count, not optimizer steps:

```text
reliability_i = n_i / (n_i + metric_prior_strength)
adjusted_metric_i = reliability_i * metric_i + (1 - reliability_i) * federation_mean
```

This avoids applying a sample-count prior such as `100` to a local-step count that may be only a
single-digit number.

### Cohort stability

When `require_stable_cohort=True`, a change in the participating client set resets the activation
streak. This prevents evidence from different client populations from being combined as if it came
from one stable cohort.

## NVFlare Metadata Contract

Each adaptive contribution provides:

```python
from nvflare.apis.dxo import MetaKey
from adaptive_hetero.nvflare_aggregator import AdaptiveMetaKey

dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, num_local_optimizer_steps)
dxo.set_meta_prop(AdaptiveMetaKey.SAMPLE_COUNT, num_training_examples)
dxo.set_meta_prop(AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR, descriptor)
dxo.set_meta_prop(AdaptiveMetaKey.CLIENT_METRIC, validation_accuracy)
dxo.set_meta_prop(AdaptiveMetaKey.QUALITY_IMPROVEMENT, baseline_loss - final_loss)
```

`NUM_STEPS_CURRENT_ROUND` is used only for the native NVFlare weighting fallback. The adaptive
sample exponent, federation reference distribution, and metric-reliability calculation use
`AdaptiveMetaKey.SAMPLE_COUNT`.

Per-site final weights are kept server-side and are not placed in aggregated model metadata, because
aggregated metadata may be copied into the global model and broadcast to participants. Returned
adaptive metadata contains only federation-level diagnostics such as mean heterogeneity, metric gap,
blend factor, and whether configured bounds were feasible.

The Shareable/DXO adapter returns `ReturnCode.EMPTY_RESULT` if no contribution is accepted. The
unified FedAvg `ModelAggregator` returns an empty DIFF no-op in the analogous empty-round case, so an
empty round does not fail solely because the adaptive component has nothing to combine.

## Current Engineering Validation

The project includes focused checks for:

- bounded-simplex projection;
- exact native fallback behavior;
- separate optimizer-step and sample-count semantics;
- metric reliability based on actual sample counts;
- warm-up, activation patience, and cohort-reset behavior;
- one-client and large-cohort bound feasibility;
- final active min/max constraints when bounds are configured;
- malformed or missing adaptive metadata;
- empty-round behavior for both aggregator interfaces;
- server-side-only per-site weight diagnostics;
- custom constructor arguments required by NVFlare `FedJob` serialization;
- exported FedOpt and unified FedAvg job configuration with non-default adaptive settings;
- real `DXO`, `Shareable`, `FLContext`, `FLModel`, `WeightedAggregationHelper`, and Recipe integration;
- a lightweight `FedOptRecipe + SimEnv` smoke path;
- a FedCE protocol smoke path;
- synthetic and scikit-learn digits development benchmarks.

These checks establish implementation behavior only. They are not the final scientific evaluation
requested for acceptance into `research/`.

No project-specific GitHub Actions workflow is added. Reproducibility commands and experiment
configuration stay inside this project directory so the research workload does not gate unrelated
repository merges.

## Standard CIFAR-10 Evaluation

`cifar10_evaluation/` implements the standard non-IID evaluation requested by the maintainers.
It reuses NVIDIA FLARE's `ModerateCNN`, CIFAR-10 data utilities, and Dirichlet partitioner.
For a fixed `(alpha, seed)`, every method receives the same training partition and the server model is
initialized from the same seed.

The comparison runner currently supports:

- FedAvg;
- FedOpt as a full-participation reference;
- FedProx;
- SCAFFOLD;
- FedCE;
- adaptive heterogeneity-aware aggregation.

### Common post-training evaluator

NVIDIA's stock CIFAR clients evaluate against the complete CIFAR-10 test set at every site. That is
appropriate for ordinary global accuracy but does not create a client-specific performance measure.
The adaptive method requires a meaningful client-performance disparity signal, and the requested
scientific comparison also needs a common worst-client metric.

`eval_split.py` therefore creates deterministic site-specific CIFAR-10 test partitions whose class
mixtures follow the corresponding Dirichlet training partitions. `evaluate_result.py` evaluates the
**final server checkpoint of every method through the same evaluator**:

- `global_accuracy`: accuracy on the complete CIFAR-10 test set;
- `client_accuracies`: accuracy on each identical site-specific evaluation partition;
- `worst_client_accuracy`: minimum client accuracy;
- `mean_client_accuracy`, best-client accuracy, and client accuracy gap.

This post-training evaluation is method-independent, so headline FedAvg/FedProx/SCAFFOLD/FedCE and
adaptive results are not mixed across different validation protocols.

### Reproducibility and pairing

The `seed` controls the Dirichlet split and server-model initialization. NVIDIA's stock CIFAR
baseline client scripts do not all expose a client-side RNG argument, so local minibatch and data-
augmentation stochasticity is treated as part of run-to-run variance. Paired comparisons are
therefore described as paired by **Dirichlet split and server initialization**, not as identical
client-side stochastic trajectories.

### One run

From `research/adaptive-hetero-aggregation/`:

```bash
python cifar10_evaluation/run.py \
  --method adaptive \
  --alpha 0.1 \
  --seed 7 \
  --participation_rate 1.0 \
  --results_jsonl results/cifar10_runs.jsonl
```

The command trains the federation, evaluates the final server checkpoint with the common evaluator,
prints a `CIFAR10_EVAL_RESULT` JSON record, and optionally appends that record to the supplied JSONL
file.

### Maintainer-requested comparison campaign

`run_campaign.py` is an intentional research workload, not a repository CI gate. Its defaults are:

- methods: FedAvg, FedProx, SCAFFOLD, FedCE, adaptive;
- Dirichlet alpha: `0.1` and `0.5`;
- seeds: `7, 19, 31, 43, 57`;
- participation: `1.0` and `0.75`;
- 8 clients, 50 rounds, 4 local epochs.

Run the full default matrix with:

```bash
python cifar10_evaluation/run_campaign.py
```

Inspect the matrix without training:

```bash
python cifar10_evaluation/run_campaign.py --dry_run
```

The campaign is resumable by default: completed `(method, alpha, participation, seed)` rows already
present in `results/cifar10_runs.jsonl` are skipped. Use `--fresh` to start a new result file.

FedOpt can be added as a full-participation reference:

```bash
python cifar10_evaluation/run_campaign.py \
  --methods fedavg fedopt fedprox scaffold fedce adaptive
```

### Confidence intervals

`summarize_results.py` consumes the JSONL rows emitted by the common evaluator and reports, for each
method/condition:

- sample size;
- mean;
- sample standard deviation;
- two-sided 95% Student-t confidence interval for global and worst-client accuracy.

It also reports paired adaptive-minus-baseline confidence intervals for common split/initialization
seeds. `run_campaign.py` calls the summarizer automatically after the requested runs complete.

## Evidence Still Required Before Research Acceptance

The evaluation infrastructure is now present, but quantitative claims must not be added until the
full matched campaign has actually completed. Before this draft should be treated as a merge-ready
`research/` contribution, it still needs:

1. completed CIFAR-10 runs against FedProx, SCAFFOLD, and FedCE under the documented protocol;
2. confidence-interval tables generated from those completed runs;
3. partial-participation results included in the main result tables;
4. transparent reporting of neutral or negative results as well as improvements;
5. a public method write-up such as a preprint or workshop paper.

The current repository does not claim that those final evidence items already exist.

## Existing Development Benchmarks

### Synthetic benchmark

`benchmark.py` uses `sklearn.datasets.make_classification` and remains useful for fast development
and reproducibility checks.

### Handwritten digits benchmark

`digits_benchmark.py` uses scikit-learn's bundled `load_digits` dataset (1,797 8x8 handwritten digit
images). It supports linear and MLP models, full or partial participation, deterministic seeds, and
held cohorts.

These small benchmarks are development tools. Previous checked-in numerical results were removed
because they predated the current policy and included blend factors that are unreachable under the
current `max_blend_factor=0.20`. New quantitative claims will be based only on the final policy and
the standard evaluation protocol.

## Relationship to Existing NVFlare Work

### FedAvg and FedOpt

The custom method is an aggregation policy rather than a server optimizer. The unified FedAvg path
uses `AdaptiveHeterogeneityModelAggregator`, while the FedOpt smoke/reference path uses
`AdaptiveHeterogeneityAggregator`. Both delegate client weighting to the same policy implementation.
When adaptive activation is off, native local-step weighting is preserved.

### FedProx and SCAFFOLD

FedProx and SCAFFOLD primarily modify local/client optimization to address non-IID optimization drift.
They are required baselines because they solve a related heterogeneity problem through a different
mechanism. The standard campaign runs them under the same Dirichlet partitions, architecture, and
training budget used for the adaptive method.

### FedCE

FedCE dynamically estimates client contribution using gradient/update and validation behavior. It is
the closest existing NVFlare contribution-aware aggregation baseline. `fedce_client.py` adapts the
real NVFlare FedCE leave-one-out validation contract to the same CIFAR-10 setup; it is not a
synthetic approximation of FedCE.

### Auto-FedRL

Auto-FedRL learns aggregation behavior through reinforcement learning. It remains relevant context,
but it is not currently part of the maintainer-requested minimum benchmark set for this draft.

## Privacy and Trust Considerations

Distribution descriptors, sample counts, and client metrics are summary metadata; they are not
privacy-preserving by default. Depending on the application, they may disclose information about
local data and may require approved summaries, secure aggregation, differential privacy, or another
privacy mechanism.

The server also assumes supplied metadata is trustworthy and comparable across clients. Production
use would require an application-specific metadata contract and appropriate privacy and integrity
controls.

## Remaining Limitations

The method has not yet been established as broadly effective. Remaining work includes:

- completing and reporting the standard non-IID CIFAR-10 campaign;
- larger client populations and longer training runs;
- additional datasets and model architectures;
- metadata privacy, trust, and cross-site comparability;
- runtime and communication overhead at larger scale;
- public documentation of the method in a preprint or workshop paper.

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
|   |-- eval_split.py
|   |-- evaluate_result.py
|   |-- fedce_client.py
|   |-- run.py
|   |-- run_campaign.py
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

## Running Focused Tests

From `research/adaptive-hetero-aggregation/`:

```bash
PYTHONPATH=src pytest -q tests
```

Run the lightweight NVFlare smoke job intentionally rather than as repository-wide CI:

```bash
python nvflare_smoke/job.py \
  --n_clients 3 \
  --num_rounds 5 \
  --train_samples 180 \
  --valid_samples 90 \
  --local_epochs 1
```

## License

Apache License 2.0, consistent with the NVIDIA FLARE repository.
