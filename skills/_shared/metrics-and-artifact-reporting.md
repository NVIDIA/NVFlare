# Metrics And Artifact Reporting

Use this reference when reporting local validation and simulation results.

## Primary Metric Alignment

When job documentation, task guidance, or source project guidance declares a
primary or target validation metric, configure the FL recipe or global metric
to that metric key so the converted client returns it in `FLModel.metrics` and
the server writes it to the metrics artifact. Report that scalar as the primary
validation evidence. Do not silently substitute a different metric.

## Simulation Metrics

After successful simulation, inspect the server workspace and logs for metrics
evidence. Standard NVFLARE aggregation recipes commonly write server-side
metrics under:

- `server/simulate_job/metrics/metrics_summary.json` for final or best
  aggregate metrics;
- `server/simulate_job/metrics/round_metrics.jsonl` for per-round or per-site
  metrics when present.

Report final/best metrics and round/per-site metrics separately. If either
category is absent, say so and fall back to bounded stdout/stderr, server logs,
or other job-produced artifacts. Do not invent metrics or treat missing round
metrics as a successful final metric.

## Round Progression Evidence

When round metrics are present, compare them across rounds before declaring the
training behavior healthy. If the reported metric is unchanged across rounds,
or if the best/final metric evidence appears to come only from initial or
pre-training evaluation, report this as a training-lifecycle concern and name
the artifact path. Do not hide the concern just because the final metric exists.

## Final Response Expectations

Report:

- final metric values, metric names, and where they came from;
- round/per-site metric paths or a clear missing-evidence note;
- model or checkpoint artifact paths when present;
- command, status, result directory, and dependency or data blockers;
- any mismatch between user-requested metrics and observed artifacts.
