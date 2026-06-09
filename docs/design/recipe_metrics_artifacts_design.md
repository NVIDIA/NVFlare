# Recipe Metrics Artifacts Design

## Summary

NVFlare aggregation and training job recipes should persist a standard, machine-readable metrics artifact when they report aggregated round metrics. Today the final aggregated metric values are generally visible only in server logs or experiment tracking backends. This makes automated benchmarking and reporting harder than necessary.

This design proposes a common server-side metrics writer that records:

- final aggregated metrics from the last completed round
- best metrics published by existing model-selection logic, when available
- per-site client-reported metrics for each round
- aggregation weights and provenance when available
- skipped metrics and skip reasons for unsupported or unsafe metric values

The writer must be generic across aggregation and training recipes and metric names. Metric names such as `auroc`, `accuracy`, `loss`, `dice`, `rmse`, or `f1` are user/client supplied keys, not hard-coded schema fields.

The writer is intentionally an artifact layer over existing workflow outputs. It should consume metrics that recipes and controllers already produce, such as aggregated `FLModel.metrics`, `AGGREGATION_RESULT` payloads, existing contribution weight metadata, and best-selection metadata from model selectors. It should not introduce a parallel aggregation path or best-round selection path.

## Current State

The current recipe surface does not persist a standalone `final_metrics.json` or `metrics.json` from normal training aggregation.

Relevant implementation points:

- `nvflare/job_config/base_fed_job.py` installs `ValidationJsonGenerator` and `IntimeModelSelector` in the shared `BaseFedJob` path.
- `nvflare/app_common/widgets/intime_model_selector.py` computes and logs a selected validation metric for best-model selection.
- `nvflare/app_common/widgets/validation_json_generator.py` writes `cross_site_val/cross_val_results.json`, but only for cross-site validation result events.
- `nvflare/app_common/workflows/fedavg.py` computes weighted aggregated `FLModel.metrics` for rounds in the InTime FedAvg path, but those metrics are not persisted as a metrics artifact.
- `nvflare/app_common/workflows/base_fedavg.py` fires `AppEventType.AFTER_AGGREGATION` with `AppConstants.AGGREGATION_RESULT` in the non-streaming BaseFedAvg path.

One important detail is that `IntimeModelSelector` and FedAvg round aggregation may not use identical weighting today. `IntimeModelSelector` defaults to equal client weights unless configured with `weigh_by_local_iter=True`; FedAvg's aggregated round metrics use `NUM_STEPS_CURRENT_ROUND` and optional site aggregation weights. The artifact should preserve the provenance of each reported value instead of pretending these streams are identical.

## Goals

- Provide a standard metrics artifact contract for aggregation and training recipes that already produce metrics.
- Preserve final aggregated metric values and official best metric values when existing selectors or workflows publish them.
- Preserve per-site client-reported metrics for every completed round.
- Keep the schema metric-agnostic.
- Avoid recipe-specific log scraping.
- Make the writer safe when metric names and values are provided by untrusted clients.
- Allow benchmarking/reporting tools to consume a stable JSON format.

## Non-Goals

- Do not recompute nonlinear metrics such as AUROC from pooled predictions.
- Do not require clients to report a specific metric name.
- Do not force recipes without metrics to invent placeholder metric values.
- Do not replace TensorBoard, MLflow, W&B, or existing analytics streaming.
- Do not change model aggregation semantics.
- Do not reimplement aggregation algorithms that already exist in controllers, aggregators, selectors, or workflow helpers.
- Do not infer best-round policy or select best rounds in the writer.
- Do not add metrics artifacts to no-aggregation workflows such as PSI or stats recipes.
- Do not merge cross-site validation into this artifact contract; it remains a separate workflow with its existing `cross_val_results.json` artifact.

## Proposed Artifacts

The writer should create a dedicated metrics directory under the server run directory:

```text
metrics/
  metrics_summary.json
  round_metrics.jsonl
```

`metrics_summary.json` is compact and intended for dashboards, benchmark reports, and quick inspection.

`round_metrics.jsonl` is append-friendly and stores one JSON object per completed round.

Round values are recorded verbatim from NVFlare workflow metadata, such as `AppConstants.CURRENT_ROUND` or `FLModel.current_round`. They are 0-based by default and honor nonzero `start_round`; the artifact does not renumber rounds.

No artifact is required for workflows that do not perform training aggregation or do not produce aggregation metrics. This includes PSI, stats-only jobs, and cross-site validation. For an aggregation workflow where the writer is installed but no metrics are reported, the writer should remain quiet or emit a bounded debug log rather than creating a synthetic metrics file.

## Schema: metrics_summary.json

Use arrays for dynamic metric names rather than JSON object keys. This avoids problems in downstream JavaScript tooling with special keys such as `__proto__`, `constructor`, or deeply nested object paths.

Example:

```json
{
  "schema_version": "1",
  "job_name": "ames_fedavg",
  "algorithm": "FedAvg",
  "status": "metrics_reported",
  "metric_source": "client_reported_flmodel_metrics",
  "key_metric": {
    "name": "auroc",
    "mode": "max",
    "mode_source": "IntimeModelSelector.negate_key_metric"
  },
  "final_round": 2,
  "final_aggregated_metrics": [
    {
      "name": "auroc",
      "value": 0.7421
    },
    {
      "name": "train_loss",
      "value": 0.492
    }
  ],
  "best_round": 0,
  "best_metrics": [
    {
      "name": "auroc",
      "value": 0.7500010132169
    }
  ],
  "best_metric_source": "IntimeModelSelector",
  "best_metric_detail_source": "initial_metrics",
  "aggregation": {
    "method": "weighted_average",
    "weight_key": "NUM_STEPS_CURRENT_ROUND",
    "metric_policy": "finite_numeric_metrics_only_per_key_denominator"
  },
  "round_metrics_file": "round_metrics.jsonl",
  "notes": [
    "Aggregated metrics are weighted averages of client-reported metric values.",
    "Nonlinear metrics are not recomputed from pooled predictions."
  ]
}
```

`key_metric.name` is published by existing recipe, selector, stop-condition, or workflow logic. It is not a fixed schema field.

`key_metric.mode` is also published by existing logic, not registered separately by users. Examples:

- `IntimeModelSelector` normally implies `max`; if `negate_key_metric=True`, raw metric selection is `min`.
- FedAvg `stop_cond` operators such as `>` or `>=` imply `max`; `<` or `<=` imply `min`.
- Custom workflows may expose their comparator or selected best-round metadata through the same event contract.

The writer records best fields only when an existing selector or workflow publishes explicit best-selection metadata. If no such metadata is available, the writer still records final metrics and round metrics but omits `best_round`, `best_metrics`, and `best_aggregated_metrics`.

## Schema: round_metrics.jsonl

Each line is a full JSON object for one completed round.

Example:

```json
{
  "round": 0,
  "aggregated_metrics": [
    {
      "name": "auroc",
      "value": 0.7500010132169
    },
    {
      "name": "train_loss",
      "value": 0.4855
    }
  ],
  "sites": [
    {
      "name": "site-1",
      "metrics": [
        {
          "name": "train_loss",
          "value": 0.4707
        },
        {
          "name": "auroc",
          "value": 0.7380791446479046
        }
      ],
      "weight": 2911,
      "weight_key": "NUM_STEPS_CURRENT_ROUND"
    },
    {
      "name": "site-2",
      "metrics": [
        {
          "name": "train_loss",
          "value": 0.5003
        },
        {
          "name": "auroc",
          "value": 0.7619228817858955
        }
      ],
      "weight": 2911,
      "weight_key": "NUM_STEPS_CURRENT_ROUND"
    }
  ],
  "aggregation": {
    "method": "weighted_average",
    "weight_key": "NUM_STEPS_CURRENT_ROUND",
    "metric_policy": "finite_numeric_metrics_only_per_key_denominator"
  },
  "key_metric": {
    "name": "auroc",
    "mode": "max",
    "mode_source": "IntimeModelSelector.negate_key_metric"
  },
  "skipped_metrics": [
    {
      "site": "site-1",
      "name": "debug_blob",
      "reason": "unsupported_type"
    }
  ]
}
```

The `metrics` arrays preserve client-reported metric names and values after safe normalization. `aggregated_metrics` contains official workflow/controller aggregated metrics after safe normalization; it is not recomputed by the writer from site records.

## Event Contract

The writer should be one common server-side component, for example `MetricsArtifactWriter`, installed by aggregation recipe setup paths where practical. Its responsibility is artifact recording: final metrics, official best metrics, per-round client metrics, provenance, and skipped values. Existing aggregators and best-model selectors remain responsible for computing aggregates and deciding which model is best.

`Widget` is a reasonable implementation base because it is already an event-driven `FLComponent` pattern used by components such as `ValidationJsonGenerator`. The public contract is "server-side metrics artifact component" rather than "user must register a widget."

To avoid recipe-specific parsing, the writer consumes these consistent events:

1. Per-client contribution accepted

   The writer needs the current round, client identity, client metrics, and contribution weight. Existing events such as `AppEventType.BEFORE_CONTRIBUTION_ACCEPT` can provide this in some workflows, but not all workflows expose a normalized `FLModel` result at this point.

2. Round aggregation completed

   Workflows should fire:

   ```python
   fire_event_with_data(
       AppEventType.AFTER_AGGREGATION,
       fl_ctx,
       AppConstants.AGGREGATION_RESULT,
       aggr_result,
   )
   ```

   where `aggr_result` is an `FLModel` or an equivalent object with a normalized metrics dictionary and round metadata.

3. Run ended

   On `EventType.END_RUN`, the writer finalizes `metrics_summary.json`.

4. Best model selected

   Model selectors that decide a new global best model should continue firing their existing best-model event and may attach explicit metrics metadata:

   ```python
   fl_ctx.set_prop(
       AppConstants.METRICS_SELECTION_INFO,
       {
           "source": "IntimeModelSelector",
           "metric_source": "initial_metrics",
           "key_metric": {
               "name": "auroc",
               "mode": "max",
               "mode_source": "IntimeModelSelector.negate_key_metric",
           },
           "best_round": current_round,
           "best_metrics": {"auroc": 0.7500010132169},
       },
       private=True,
       sticky=False,
   )
   fire_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_ctx)
   ```

   The writer records this metadata without comparing metric values.

Built-in aggregation workflows publish the same event contract when they have official round-level aggregated metrics. This includes FedAvg-derived workflows and non-FedAvg aggregation workflows that return an `FLModel` or `Shareable` aggregation result. The writer normalizes both payload shapes, but it does not synthesize aggregate metrics from analytics logs or per-site records.

For custom aggregators, the contract is event-based rather than a new mandatory aggregator base interface. Controllers fire the standard aggregation event with the `FLModel` returned by the aggregator. The easy user path is: custom aggregators return `FLModel(metrics=...)` as they do today. If they have custom metric weights or provenance, they can attach bounded metadata to `FLModel.meta`, for example under `metrics_aggregation_info`.

## Recipe Coverage

This PR installs the writer in these recipe setup paths:

- `BaseFedJob`, covering recipe paths built on the shared BaseFedJob configuration such as FedAvg, FedOpt, Scaffold, FedAvg with HE, and PyTorch, TensorFlow, NumPy, and sklearn variants
- LR FedAvg
- standalone XGBoost recipes
- Edge SAGE setup

Recipes that do not perform aggregation or do not produce aggregation metrics should remain valid and should not be forced to create metrics artifacts.

## Best Metric Selection

Best metric selection is not a writer responsibility. It belongs to existing model selectors, controllers, or workflows that already own comparison policy and model persistence.

Existing model selectors such as `IntimeModelSelector` can continue to select and persist best models. When they expose `METRICS_SELECTION_INFO`, the writer records:

- `key_metric`
- `best_round`
- `best_metrics`
- optional `best_aggregated_metrics`
- source/provenance fields

If a selector or workflow does not expose compatible metadata, the artifact omits best fields rather than inferring a policy.

## Aggregation Policy

The writer should not invent aggregation semantics. Its primary job is to persist what existing recipes, controllers, aggregators, and selectors already compute. `aggregated_metrics` should come from the workflow's own `AGGREGATION_RESULT` or equivalent normalized controller output.

Per-site records should include all safely normalized scalar metrics, including bool metrics. Site weights should be recorded as provenance when available, but they should not be used by the writer to compute a synthetic aggregate.

For nonlinear metrics, record a note that aggregated values are weighted averages of client-reported values, not pooled metric computations.

## Security Model

Clients are untrusted metric producers. The metrics writer is a server-side sink for untrusted data and must not interpret metric content as executable code, configuration, or filesystem paths.

Threats to consider:

- code execution through unsafe deserialization or dynamic evaluation
- path traversal through client-provided names
- resource exhaustion through huge metric payloads
- JSON poisoning for downstream JavaScript consumers
- invalid numeric values such as `NaN` and `Infinity`
- object side effects through `__str__`, `__repr__`, or custom serialization hooks

Required protections:

- Do not use `eval`, `exec`, dynamic imports, class loading, `pickle`, `yaml.load`, or shell commands.
- Do not call `str(value)` or `repr(value)` on arbitrary client-provided metric values.
- Serialize only explicitly normalized JSON-safe values.
- Use fixed output filenames.
- Never use metric names, site names, job names, or other client-controlled metadata as path components.
- Resolve output paths under the server run directory with `resolve_path_under_root`.
- Use `json.dump(..., allow_nan=False)`.
- Accept finite numeric values and bool values for official aggregated metrics.
- Bound metric name length, number of metrics per client, number of sites, skipped metric records, and JSON line size.
- Store dynamic metric names as values in arrays, not as object keys.
- Record skipped metrics by name and reason only after safe name normalization.

Suggested default limits:

```text
max_metric_name_length = 256
max_metrics_per_site_per_round = 512
max_sites_per_round = 10000
max_skipped_metrics_per_round = 1024
max_round_record_bytes = 1048576
max_summary_bytes = 1048576
```

Metric name normalization:

- accept only strings
- truncate to the configured maximum length
- reject or replace control characters
- do not interpret separators such as `.`, `/`, `[`, or `]`

Metric value normalization:

- accept `int`, `float`, bool, and bounded strings for raw per-site metric records
- convert numeric and bool values to plain Python JSON scalars
- reject non-finite numbers
- record only finite numeric values and bool values for official aggregated metrics
- skip dicts, lists, sets, tuples, objects, tensors, arrays, bytes, and oversized strings

## Failure Behavior

The writer should not fail the job for bad metric payloads. It should:

- skip invalid metrics
- record skip reasons in the round record
- log bounded warnings
- continue writing valid metrics

The writer may fail closed only for local server-side errors such as inability to resolve a safe output path. In that case, it should log an error and avoid writing outside the run directory.

## Implementation Sketch

Add a common server-side metrics artifact component. The first implementation can subclass `Widget` because the component is event-driven, but the design only requires `FLComponent` behavior:

```python
class MetricsArtifactWriter(Widget):  # or FLComponent if no Widget-specific behavior is needed
    def __init__(
        self,
        results_dir="metrics",
        summary_file_name="metrics_summary.json",
        round_file_name="round_metrics.jsonl",
        limits=None,
    ):
        ...
```

Behavior:

- On `EventType.START_RUN`, reset in-memory summary state.
- On contribution events, capture normalized per-site metrics and weights for the current round.
- On `AppEventType.AFTER_AGGREGATION`, capture normalized aggregated metrics and append one JSONL record for the current round.
- On `AppEventType.GLOBAL_BEST_MODEL_AVAILABLE`, capture normalized `METRICS_SELECTION_INFO` if the selector published it.
- Optionally read bounded `metrics_aggregation_info` metadata from `FLModel.meta` for custom aggregator provenance.
- On `EventType.END_RUN`, write `metrics_summary.json`.

State to keep in memory:

- current round per-site metrics until the round aggregate is known
- final round number and final aggregated metrics
- official best-selection metadata, when published by a selector or workflow
- skip counters and bounded skipped metric records

The writer should stream `round_metrics.jsonl` as rounds complete rather than keeping all rounds in memory.

## Recipe Integration

Current PR scope:

- Add the writer to `BaseFedJob` so the recipes already using this path get the artifact.
- Add the writer to standalone aggregation recipe setup paths that do not use `BaseFedJob` and have a standard aggregation workflow.
- Standardize built-in aggregation workflows on `AFTER_AGGREGATION` with `AGGREGATION_RESULT` for official round metrics, accepting both `FLModel` and compatible `Shareable` payloads.
- Document the artifact contract as part of the user-facing recipe documentation.
- Include metrics artifacts in normal job result downloads when they exist, and report their local paths through the existing `nvflare job download --format json` `artifacts` map.

## Test Coverage

Unit tests cover:

- summary file with final metrics
- official best-selection metadata passthrough
- JSONL file with one line per round
- dynamic metric names
- no hard-coded metric names
- no writer-side best-round selection or policy inference
- absent metrics in an aggregation workflow
- missing key metric
- partial metric coverage across sites
- skipped unsupported metric types
- bool metrics in raw records
- `NaN` and `Infinity` rejection
- metric names such as `__proto__`, `constructor`, paths, and control characters
- output path containment using `resolve_path_under_root`
- file size and per-round limit enforcement

## Resolved Design Choices

- Metrics artifact writing is a recorder concern. Aggregation, key-metric policy, and best-round selection remain owned by workflows, aggregators, and model selectors.
- Bool metrics should be recorded. In per-site records they remain bool values; in aggregated metrics they preserve the official workflow-provided value.
- Custom aggregators should not be forced into a new mandatory interface. The common contract is the controller-fired aggregation event. Custom aggregators can return `FLModel(metrics=...)` and optionally attach bounded `metrics_aggregation_info` metadata for weights and provenance.
