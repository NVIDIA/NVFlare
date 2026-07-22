# Auto-FL Experiment Comparability

Use this reference for Auto-FL reruns, recipe comparisons, data replacement,
data distribution, synthetic data, and site heterogeneity requests.

## Canonical Selection Metric

The per-run ledger score, extracted with the objective's
`metric_extraction_order`, is the canonical selection surface for baseline,
keep/discard, and best-candidate comparisons. Cross-site server-final
global-model scores from `cross_val_results.json` are diagnostic unless the
user explicitly requests selection on them.

## Iterative Reruns

When the user asks to change batch size, train args, number of rounds,
`min_clients`, site count, or recipe, update only the affected training args or
job configuration. For recipe changes, rerun
`nvflare recipe show <recipe-name> --format json` and verify the new recipe's
parameters before editing.

After each change, rerun local validation when possible. If an exported job was
previously produced and export remains in scope, export again so the job folder
matches the updated source.

## Recipe Search And Accuracy Comparison

When the user asks for the best recipe or best accuracy, first define the target
metric, validation split, maximum run budget, and compatible recipe set. Do not
promise that one recipe is best without measured evidence from comparable runs.

Keep dataset split, number of sites, rounds, epochs, seed, and evaluation metric
comparable unless the user asks to tune them. Report a small results table with
recipe, settings, metric value, command, status, and result path. If a candidate
cannot run, report the blocker instead of silently dropping it.

## Data Distribution Experiments

When the user asks to compare IID and heterogeneous data splits, define the split
strategy before editing. Examples include equal random IID shards, label-skewed
non-IID shards, quantity-skewed shards, or user-provided per-site partitions.

Keep recipe, rounds, epochs, batch size, seed, and metric comparable unless the
user asks to tune them. Do not copy private data into generated artifacts; prefer
split indices, deterministic samplers, or site-local path arguments.

Report split strategy, per-site sample counts or label summary when available,
metric value, command, status, and result path.

## Dataset Replacement Experiments

When the user provides a dataset URL and asks to repeat an experiment, record
the URL, dataset name when known, version or timestamp when available, expected
download size if known, license or access constraints when visible, and local
cache/path used for validation.

Do not hide download, preprocessing, schema, or label-mapping assumptions. Keep
recipe, site count, rounds, epochs, batch size, seed, split policy, and metric
comparable unless the user asks to tune them. If the new dataset requires a data
loader or preprocessing change, keep it scoped and report the changed files.

Follow the project's existing data-prep structure. If it already has
`download_data.py`, `prepare_data.py`, `prepare_data.sh`, or equivalent helpers,
extend those rather than creating a parallel structure. If no helper exists and
a new one is needed, use the established NVFLARE example convention of separate
download and prepare/split steps, and keep download paths, cache paths, and
per-site output directories explicit.

Use hello-world examples as the first convention reference for new helpers:
`examples/hello-world/hello-lr/download_data.py`,
`examples/hello-world/hello-lr/prepare_data.py`,
`examples/hello-world/hello-jax/prepare_data.py`, and shell-based examples such
as `examples/hello-world/hello-cyclic/prepare_data.sh`.

## Synthetic Per-Site Data

When the user asks for synthetic data per site, add a deterministic data
generation step only after the expected data schema is clear from the model
input, transforms, loss function, and data loader, or from a user-provided data
generation spec.

If modality, schema, label semantics, target distribution, missingness/noise,
site heterogeneity, expected metric, or generation library is not clear, ask the
user for a data generation spec or an approved generator/library before creating
data. The spec should name modality, shape or columns, label or target
definition, class balance or value distributions, missing-data and noise
assumptions, per-site distribution differences, sample counts, seed, and
expected metric interpretation.

Do not invent labels, features, class balance, missingness, site skew, or
expected accuracy. If the user supplies a generator or domain-specific synthetic
data tool, wire it into the existing data-prep flow and record the tool, version
or command, seed, and parameters used.

Prefer extending existing `prepare_data.py`, `prepare_data.sh`, or equivalent
helpers. If a separate generator is needed, keep it under the same data-prep
structure and call it from the prepare step.

Generated site data should be written to explicit per-site outputs that the job
can pass as site-specific data paths. Treat synthetic validation as a smoke test
of wiring and training execution unless the user provides a synthetic data spec
with meaningful expected metrics.

## Site-Specific Training Heterogeneity

When the user asks to simulate different site speeds or training
hyperparameters, prefer per-site arguments or per-site config in `job.py`.
Examples include per-site learning rate, batch size, local epochs, sleep/delay
for speed simulation, dataset shard, or workload size.

Only create site-specific training scripts when arguments/config cannot express
the requested behavior. If scripts are split, keep shared model and training
helpers common and report why script splitting was necessary. Report each
site's settings, command, status, metric, and result path.

## Common Gaps To Report

- The source training script has side effects at import time.
- The model has non-serializable state outside framework-native model state.
- The dataset path is site-specific and cannot be validated locally.
- The job file has no export path yet.
