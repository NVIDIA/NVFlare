.. _recipe_metrics_artifacts:

Recipe Metrics Artifacts
========================

Built-in training aggregation recipes write standard metrics artifacts when the
server workflow reports round-level aggregation metrics. These files make recipe
results easier to consume from benchmark, reporting, and agent tooling without
scraping server logs.

The artifacts are written under the server run directory:

.. code-block:: text

   metrics/
     metrics_summary.json
     round_metrics.jsonl

Recipes or workflows that do not report training aggregation metrics do not need
to create these files. This includes PSI, stats-only jobs, and standalone
cross-site validation. Cross-site validation continues to use its existing
``cross_site_val/cross_val_results.json`` output.

Recipe Behavior
---------------

Users do not need to select this writer for supported built-in training
aggregation recipes. The recipe setup installs it as part of the server
configuration, and it writes files only when the workflow reports aggregation
metrics.

This release does not expose a recipe argument to disable metrics artifacts. For
custom jobs that should not write these artifacts, omit the metrics artifact
writer from the server configuration.

Recorder Semantics
------------------

The metrics artifact writer is a recorder. It persists metrics and metadata that
workflows, aggregators, and model selectors already produce:

* official aggregated metrics from the round aggregation result
* per-site metrics received from clients for each round
* official best metric metadata published by model-selection logic
* aggregation provenance, weights, and skipped values when available

It does not recompute metrics, select a best round, infer max/min policy, parse
logs, or compute nonlinear metrics such as AUROC from pooled predictions.

Metric names are dynamic. Names such as ``auroc``, ``accuracy``, ``loss``,
``dice``, ``rmse``, or ``f1`` are client or workflow metric keys, not hard-coded
schema fields.

Round numbers are recorded as provided by workflow metadata such as
``AppConstants.CURRENT_ROUND`` or ``FLModel.current_round``. They are 0-based by
default and are not renumbered by the writer.

``metrics_summary.json``
------------------------

``metrics_summary.json`` contains the final aggregated metrics from the last
completed metrics round and, when available, official best metric metadata from
the model selector.

Example:

.. code-block:: json

   {
     "schema_version": "1",
     "status": "metrics_reported",
     "job_name": "ames_fedavg",
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

Best metric fields are optional. They are present only when a selector or
workflow publishes explicit best-selection metadata. The writer does not infer a
best round from metric values.

``round_metrics.jsonl``
-----------------------

``round_metrics.jsonl`` contains one JSON object per completed metrics round.
Each line records official aggregated metrics, per-site client metrics, optional
aggregation metadata, and skipped metric values.

Example line:

.. code-block:: json

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
     "skipped_metrics": [
       {
         "site": "site-1",
         "name": "debug_blob",
         "reason": "unsupported_type"
       }
     ]
   }

Dynamic metric names are stored as ``name`` values in arrays rather than as JSON
object keys. This avoids treating client-provided names as object structure in
downstream tools.

Safe Metric Values
------------------

Clients are untrusted metric producers. The writer serializes only normalized
JSON-safe scalar values and writes to fixed filenames under the server run
directory.

Official aggregated metrics accept finite numeric values and bool values.
Per-site metrics accept finite numeric values, bool values, and bounded string
values. Unsupported objects, tensors, arrays, nested containers, oversized
values, ``NaN``, and ``Infinity`` are skipped and reported in
``skipped_metrics`` with a bounded reason record.

Downloaded Artifacts
--------------------

Metrics files are part of the normal downloaded job result when they exist. For
automation, use the job download JSON output to find the downloaded local paths
instead of constructing paths from the workspace layout:

.. code-block:: shell

   nvflare job download <job_id> -o ./downloads --format json

Example response excerpt:

.. code-block:: json

   {
     "schema_version": "1",
     "status": "ok",
     "data": {
       "download_path": "/abs/path/downloads/abc123",
       "artifact_discovery": "completed",
       "artifacts": {
         "metrics_summary": "/abs/path/downloads/abc123/workspace/metrics/metrics_summary.json",
         "round_metrics": "/abs/path/downloads/abc123/workspace/metrics/round_metrics.jsonl"
       },
       "missing_artifacts": []
     }
   }

``metrics_summary`` and ``round_metrics`` are reported only when those files
exist in the downloaded result. ``round_metrics`` is optional because older jobs
and jobs without aggregation metrics do not create a per-round metrics file.
