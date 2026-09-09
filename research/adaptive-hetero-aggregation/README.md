# Adaptive Heterogeneity-Aware Aggregation

This draft research implementation addresses the adaptive client-weighting use case discussed in
[NVIDIA/NVFlare issue #5209](https://github.com/NVIDIA/NVFlare/issues/5209).
It implements a generic NVFlare `Aggregator` for `DataKind.WEIGHT_DIFF` updates. The same
aggregation mechanism can be used with FedAvg-style aggregation or with FedOpt server-side
optimization; FedOpt is not a requirement of the weighting policy itself.

The pull request is intentionally kept in draft while the method is strengthened with a standard
non-IID benchmark, additional baselines, confidence intervals, partial-participation results, and a
public method write-up. It should not yet be treated as a validated NVIDIA FLARE research example
or as production evidence.

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

The NVFlare adapter separately retains the framework's native local-step weighting signal from
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

`src/adaptive_hetero/nvflare_aggregator.py` implements the generic NVFlare `Aggregator` interface
for `DataKind.WEIGHT_DIFF` contributions.

Each contribution provides:

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

Per-site final weights are kept server-side and are not placed in aggregated DXO metadata, because
aggregated metadata is copied into the global model and may be broadcast to participants. Returned
adaptive metadata contains only federation-level diagnostics such as mean heterogeneity, metric gap,
blend factor, and whether configured bounds were feasible.

If a round has no accepted contributions, the aggregator returns `ReturnCode.EMPTY_RESULT` rather
than raising an exception that aborts the job.

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
- all-rejected-round handling through `ReturnCode.EMPTY_RESULT`;
- server-side-only per-site weight diagnostics;
- custom aggregator constructor arguments required by NVFlare `FedJob` serialization;
- real `DXO`, `Shareable`, `FLContext`, `WeightedAggregationHelper`, and Recipe integration;
- a lightweight `FedOptRecipe + SimEnv` smoke path;
- a FedCE protocol smoke path;
- synthetic and scikit-learn digits development benchmarks.

These checks establish implementation behavior only. They are not the final scientific evaluation
requested for acceptance into `research/`.

No project-specific GitHub Actions workflow is added. Reproducibility commands and experiment
configuration stay inside this project directory so the research workload does not gate unrelated
repository merges.

## Required Evaluation Before Research Acceptance

The current draft is being prepared for the evaluation requested by NVIDIA FLARE maintainers:

1. use a standard non-IID benchmark, with Dirichlet CIFAR-10 as the primary target;
2. compare against FedAvg/FedOpt native weighting, FedProx, SCAFFOLD, and FedCE;
3. use matched datasets, partitions, model architecture, training budgets, and random seeds;
4. report confidence intervals, including paired adaptive-minus-baseline intervals where applicable;
5. include partial participation in the main result tables rather than treating it only as a smoke test;
6. report both global performance and client-level/worst-client behavior;
7. publish a public method write-up such as a preprint or workshop paper before treating the code as
   a reference research implementation.

NVIDIA FLARE already contains Dirichlet CIFAR-10 examples for FedAvg, FedOpt, FedProx, and SCAFFOLD.
The intended evaluation will reuse those conventions rather than create an unrelated benchmark
protocol. FedCE requires an equivalent CIFAR-10 adaptation for an apples-to-apples comparison.

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

The custom component is an aggregation policy rather than a server optimizer. It can therefore be
used with FedAvg-style weight-difference aggregation and can also feed FedOpt's server-side optimizer.
When adaptive activation is off, the adapter preserves the native local-step weighting signal.

### FedProx and SCAFFOLD

FedProx and SCAFFOLD primarily modify local/client optimization to address non-IID optimization drift.
They are required baselines because they solve a related heterogeneity problem through a different
mechanism. The final evaluation will compare them under the same non-IID partitions and training
budget.

### FedCE

FedCE dynamically estimates client contribution using gradient/update and validation behavior. It is
the closest existing NVFlare contribution-aware aggregation baseline and will be included in the
standard benchmark comparison rather than represented only by a protocol smoke test.

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

- standard non-IID CIFAR-10 evaluation against the requested baselines;
- statistically supported confidence intervals;
- partial-participation headline results;
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
|-- fedce_smoke/
|-- nvflare_smoke/
|-- src/adaptive_hetero/
|   |-- __init__.py
|   |-- policy.py
|   `-- nvflare_aggregator.py
`-- tests/
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
