# Adaptive Heterogeneity-Aware Aggregation for FedOpt

This opt-in implementation addresses the adaptive client-weighting use case discussed in
[NVIDIA/NVFlare issue #5209](https://github.com/NVIDIA/NVFlare/issues/5209).
It keeps FedOpt as the server optimizer and changes only how client weight differences
are combined when persistent heterogeneity and client-performance disparity justify an
adaptive correction.

The implementation is currently isolated under `research/` so the weighting policy,
metadata contract, safeguards, and integration behavior can be evaluated without
changing existing NVFlare aggregation defaults.

## Motivation

Local-volume weighting is a strong default, but a large client can dominate aggregation
when client distributions differ substantially. Always applying fairness-oriented
weighting can create the opposite problem by changing aggregation when clients are
already behaving similarly.

The policy therefore answers two separate questions:

1. Is there enough persistent evidence to depart from native FedOpt weighting?
2. If so, how should influence be redistributed while keeping final client weights bounded?

## Method

For each participating client `i`, the server receives:

- a positive local-volume/local-iteration proxy `n_i`;
- a non-negative distribution descriptor;
- a normalized higher-is-better client metric in `[0, 1]`;
- an optional quality-improvement value.

The native baseline is:

```text
w_base ∝ n_i
```

The adaptive candidate combines sub-linear local volume, distribution representation,
and reliability-adjusted fairness pressure:

```text
w_adaptive ∝ n_i^gamma * representation_i^beta * fairness_i * quality_i^delta
```

The candidate is projected onto a bounded simplex. Two gates then control whether it is
used:

```text
candidate_blend = max_blend * heterogeneity_gate * performance_gap_gate
```

Activation is stateful. The default policy requires a warm-up period, consecutive
qualifying rounds, and a stable participating cohort. If those conditions are not met,
the result is the exact native FedOpt/sample-weight fallback.

When adaptation is active:

```text
blended = (1 - blend) * w_base + blend * w_adaptive
final_weight = bounded_simplex_projection(blended)
```

The final projection is important: a dominant native base weight cannot bypass
`min_weight` or `max_weight` merely because the bounded adaptive candidate is blended
with it.

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
min_weight                    = 0.02
max_weight                    = 0.50
```

The standalone benchmarks use `max_weight=0.30` for their eight-client setup.
Quality-based weighting is neutral by default (`quality_exponent=0.0`).

### Metric reliability

Client metrics from small local volumes are shrunk toward the local-volume-weighted
federation mean before fairness pressure and the performance-gap gate are computed:

```text
reliability_i = n_i / (n_i + metric_prior_strength)
adjusted_metric_i = reliability_i * metric_i + (1 - reliability_i) * federation_mean
```

This reduces sensitivity to noisy small-client measurements without dropping their
distribution representation signal.

### Cohort stability

When `require_stable_cohort=True`, a change in the participating client set resets the
activation streak. This prevents evidence from different client populations from being
combined as if it came from one stable cohort.

## NVFlare Integration

`src/adaptive_hetero/nvflare_aggregator.py` adapts the policy to NVFlare's aggregator
interface and returns standard `DataKind.WEIGHT_DIFF` output for `FedOptRecipe`.

Each contribution provides:

```python
from nvflare.apis.dxo import MetaKey
from adaptive_hetero.nvflare_aggregator import AdaptiveMetaKey

dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, num_local_steps)
dxo.set_meta_prop(AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR, descriptor)
dxo.set_meta_prop(AdaptiveMetaKey.CLIENT_METRIC, validation_accuracy)
dxo.set_meta_prop(AdaptiveMetaKey.QUALITY_IMPROVEMENT, baseline_loss - final_loss)
```

`MetaKey.NUM_STEPS_CURRENT_ROUND` is used as the native FedOpt local-iteration/local-volume
weighting signal. It should not be interpreted as a literal sample count in all
applications.

The smoke client reports validation accuracy of the received global model before local
training as `adaptive_client_metric`, so the client metric has the same round semantics
across participants.

## Validation Coverage

The dedicated workflow and repository pre-merge checks cover:

- policy unit tests and bounded-simplex stress tests;
- regression coverage for final post-blend min/max weight constraints;
- malformed metadata and normalized-metric contract checks;
- empty validation-loader rejection with a descriptive `ValueError`;
- NVFlare `DXO`, `Shareable`, and `FLContext` integration;
- stateful warm-up, activation patience, and cohort-reset behavior;
- real `FedOptRecipe + SimEnv` execution;
- real NVIDIA FedCE `SimEnv` protocol execution;
- deterministic synthetic benchmark smoke runs;
- scikit-learn handwritten-digits runs with linear and MLP models;
- multiple random seeds and mild/severe/extreme heterogeneity settings;
- partial participation with held cohorts and an assertion that adaptive weighting
  actually activates;
- formatting, lint, license, wheel-build, unit-test, and coverage checks.

The partial-participation benchmark accepts `--cohort-hold-rounds` so a selected cohort
can remain stable long enough to satisfy activation patience. CI also uses
`--require-adaptive-activation`; a run that exercises only the FedOpt fallback path is
therefore not considered sufficient validation of the adaptive path.

## Benchmarks

### Synthetic benchmark

`benchmark.py` uses `sklearn.datasets.make_classification` with configurable sample
count, client count, rounds, random seeds, and heterogeneity settings. The default
research configuration uses eight clients and compares native-volume FedOpt weighting
with the adaptive policy under the same FedAdam-style server update.

### Handwritten digits benchmark

`digits_benchmark.py` uses scikit-learn's bundled `load_digits` dataset (1,797 8x8
handwritten digit images), so no external dataset download is required. It supports:

- linear and MLP models;
- full or partial participation;
- deterministic random seeds;
- configurable cohort hold duration;
- an optional assertion that the adaptive path activates.

Example partial-participation validation:

```bash
python digits_benchmark.py \
  --rounds 25 \
  --participation-rate 0.75 \
  --cohort-hold-rounds 5 \
  --require-adaptive-activation \
  --seeds 7 19 31 \
  --settings severe extreme \
  --models linear mlp \
  --methods fedopt adaptive
```

### Historical development results

`results/development_5seed_summary.json` contains an earlier development-policy run.
Those values were useful while designing the safeguards but predate the current
stateful activation, metric-reliability, and final post-blend bounding behavior. They
should not be treated as final performance claims for the current policy.

## Relationship to Existing NVFlare Work

### FedOpt

FedOpt remains the native baseline and server optimizer. This implementation adds an
opt-in client-weighting layer and preserves exact baseline weights whenever adaptive
activation is off.

### FedCE

FedCE uses contribution-related behavior including update-direction and validation
signals. This policy instead uses explicit distribution representation and
reliability-adjusted performance disparity. A real FedCE `SimEnv` protocol smoke run is
included as an integration reference, not as a claim that the two methods are
algorithmically equivalent.

### Auto-FedRL

Auto-FedRL learns aggregation behavior with reinforcement learning. This implementation
is deterministic and does not require a separate learned policy or policy-training
phase. A full numerical Auto-FedRL reproduction is not claimed here because a comparable
reproduction requires substantially different compute and experiment setup.

## Privacy and Trust Considerations

Distribution descriptors and client metrics are summary metadata, not automatically
privacy-preserving signals. Depending on the application, they may disclose information
about local data and may require approved summaries, secure aggregation, differential
privacy, or another privacy mechanism.

The server also assumes that supplied metadata is trustworthy and comparable across
clients. Production deployments should define and enforce the metadata contract for the
specific application.

## Remaining Limitations

Current validation is intentionally broader than the initial implementation, but it is
not production evidence for every federated workload. Important remaining questions
include:

- behavior with larger client populations and longer training runs;
- additional datasets and model architectures;
- high client-dropout and rapidly changing participation patterns;
- metadata privacy, trust, and cross-site comparability;
- communication and runtime overhead at larger scale;
- application-specific tuning of thresholds and bounds;
- whether the component should remain under `research/` or evolve into a reusable
  NVFlare component after maintainer review.

No universal convergence or universal accuracy-improvement claim is made.

## Repository Layout

```text
adaptive-hetero-aggregation/
|-- README.md
|-- benchmark.py
|-- digits_benchmark.py
|-- requirements.txt
|-- fedce_smoke/
|-- nvflare_smoke/
|-- results/
|-- src/adaptive_hetero/
|   |-- __init__.py
|   |-- policy.py
|   `-- nvflare_aggregator.py
`-- tests/
    |-- test_nvflare_aggregator.py
    |-- test_nvflare_smoke_client.py
    `-- test_policy.py
```

## Running Tests

From `research/adaptive-hetero-aggregation/`:

```bash
PYTHONPATH=src pytest -q tests
```

## License

Apache License 2.0, consistent with the NVIDIA FLARE repository.
