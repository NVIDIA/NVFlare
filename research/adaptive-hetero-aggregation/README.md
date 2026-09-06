# Adaptive Heterogeneity-Aware Aggregation for FedOpt

This research prototype implements the adaptive weighting idea discussed in
[NVIDIA/NVFlare issue #5209](https://github.com/NVIDIA/NVFlare/issues/5209).
The goal is not to replace FedAvg, FedOpt, FedCE, or Auto-FedRL. Instead, it
adds a client-weighting policy that stays close to ordinary sample weighting
when measured heterogeneity is low and gradually increases representation and
worst-client pressure when heterogeneity becomes severe.

The prototype is deliberately scoped as a research contribution. It does not
claim a new federated-learning algorithm, and the results below are from a
controlled external CPU benchmark rather than an official NVFlare CIFAR-10
run.

## Motivation

Sample-count weighting can allow a very large client to dominate aggregation
although a smaller client may contain underrepresented patterns. At the same
time, always using fairness-oriented weights can unnecessarily reduce average
performance in nearly IID settings. The prototype therefore separates two
questions:

1. **When should aggregation depart from ordinary sample weighting?**
2. **How should influence be redistributed once heterogeneity is high?**

FedOpt remains the server optimizer. Only the weighted mean of client weight
differences changes.

## Method

For each client `i`, the server receives a positive local-volume proxy `n_i`, a
non-negative distribution descriptor, a higher-is-better client metric, and an
optional quality-improvement value.

The policy computes:

- ordinary base weights `w_base ∝ n_i`;
- Jensen-Shannon divergence from the sample-weighted reference descriptor;
- a bounded representation score combining distribution novelty with a smaller
  rarity term for globally underrepresented descriptor bins;
- a minimax-style fairness multiplier that increases pressure on clients with
  weaker current metrics;
- a bounded quality term based on within-round update improvement;
- sub-linear sample weighting `n_i^gamma` so volume still matters without being
  automatically dominant.

The adaptive candidate is projected onto a bounded simplex, so the configured
minimum/maximum client weights remain valid while the weights sum to one. The
final weight is a smooth blend:

```text
lambda = sigmoid((mean_heterogeneity - threshold) / temperature)
final_weight = (1 - lambda) * sample_weight + lambda * adaptive_weight
```

This smooth transition avoids a hard threshold that could cause the server to
oscillate between two aggregation regimes across adjacent rounds.

Default research parameters used in the benchmark are:

```text
sample_exponent            = 0.65
representation_exponent    = 0.70
quality_exponent           = 0.40
fairness_strength          = 2.20
heterogeneity_threshold    = 0.20
heterogeneity_temperature  = 0.04
min_weight                 = 0.02
max_weight                 = 0.30  # benchmark uses 8 clients
```

## Repository Layout

```text
adaptive-hetero-aggregation/
|-- README.md
|-- benchmark.py
|-- requirements.txt
|-- results/
|   `-- development_5seed_summary.json
|-- src/adaptive_hetero/
|   |-- __init__.py
|   |-- policy.py
|   `-- nvflare_aggregator.py
`-- tests/
    `-- test_policy.py
```

`policy.py` is independent of NVFlare and contains the weighting logic and
bounded-simplex projection. `nvflare_aggregator.py` adapts the policy to the
`Aggregator` interface used by PyTorch `FedOptRecipe` with
`DataKind.WEIGHT_DIFF`.

## NVFlare Metadata Contract

Each client contribution must provide:

```python
from nvflare.apis.dxo import MetaKey
from adaptive_hetero.nvflare_aggregator import AdaptiveMetaKey

dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, num_local_steps)
dxo.set_meta_prop(AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR, descriptor)
dxo.set_meta_prop(AdaptiveMetaKey.CLIENT_METRIC, validation_accuracy)
dxo.set_meta_prop(AdaptiveMetaKey.QUALITY_IMPROVEMENT, baseline_loss - final_loss)
```

The descriptor is intentionally application-defined. A label histogram is easy
to reproduce for research, but production applications should assess whether a
chosen descriptor leaks sensitive distribution information and use an
appropriate privacy-preserving representation when needed.

The aggregator can then be passed to `FedOptRecipe` as its custom
`aggregator=` component. FedOpt's server-side optimizer remains unchanged.

## Controlled Benchmark

The development benchmark uses `sklearn.datasets.make_classification` with:

- 30,000 samples;
- 80 features;
- 10 classes;
- 8 federated clients;
- Dirichlet label partitions;
- client-specific covariate shifts;
- 40 federated rounds;
- seeds `7, 19, 31, 43, 57`.

Both FedOpt and the adaptive method use the same FedAdam-style server update.
The only difference is the client aggregation weights.

| Setting | Method | Global accuracy | Worst-client accuracy |
| --- | --- | ---: | ---: |
| Mild | FedOpt | 71.88 ± 0.81% | 64.67 ± 2.84% |
| Mild | Adaptive | **71.86 ± 0.78%** | **64.87 ± 2.97%** |
| Severe | FedOpt | 81.39 ± 1.16% | 62.36 ± 2.55% |
| Severe | Adaptive | **81.28 ± 1.19%** | **65.24 ± 1.98%** |
| Extreme | FedOpt | 84.46 ± 1.28% | 55.48 ± 3.71% |
| Extreme | Adaptive | **84.18 ± 1.08%** | **57.48 ± 3.86%** |

Relative to the FedOpt baseline, the adaptive policy changed mean global
accuracy by **-0.02 pp / -0.11 pp / -0.29 pp** in mild/severe/extreme settings,
while worst-client accuracy changed by **+0.20 pp / +2.88 pp / +2.00 pp**.

For mild heterogeneity the observed blend factors were only about `0.06-0.09`,
so the method behaved close to its FedOpt/sample-weight fallback. The complete
per-seed output is stored in `results/development_5seed_summary.json`.

Run the same benchmark with:

```bash
python benchmark.py \
  --samples 30000 \
  --rounds 40 \
  --seeds 7 19 31 43 57 \
  --settings mild severe extreme \
  --methods fedopt adaptive \
  --output results/development_5seed.json
```

## Tests

The policy tests cover weight normalization, mathematically valid bounded-simplex
projection, low-heterogeneity fallback behavior, high-heterogeneity minimax
pressure, malformed descriptors, and infeasible bounds.

```bash
PYTHONPATH=src pytest -q tests/test_policy.py
```

Development validation for this branch:

```text
5 passed
Python syntax compilation passed for policy, benchmark, tests, and NVFlare adapter
```

## What Still Needs to Be Validated Before an Upstream PR

This branch is intentionally a first implementation rather than a claim that the
feature is already production-ready. Before proposing it upstream, the strongest
next validation would be:

- run the adapter inside the current NVFlare simulator with the real
  `FedOptRecipe`;
- compare against NVFlare's actual FedCE implementation and the existing
  Auto-FedRL research example rather than only the controlled FedOpt baseline;
- add a public image benchmark such as CIFAR-10 with Dirichlet label skew;
- add partial-participation experiments;
- test descriptor privacy and metric comparability assumptions;
- add NVFlare integration tests for malformed/missing client metadata and
  changing client sets.

These checks are important because a client-local metric is not automatically
comparable across sites, and distribution descriptors can disclose information
unless designed carefully.

## License

Apache License 2.0, consistent with the NVIDIA FLARE repository.
