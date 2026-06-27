# Metrics And Artifact Reporting

Use this reference when reporting validation, simulation, POC, or production
results.

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

## POC Or Production Downloaded Artifacts

For POC or production runs in a terminal state, use:

```bash
nvflare job download <job_id> -o <dir> --format json
```

Read artifact paths from the JSON response when present, including
`data.artifacts.global_model`, `data.artifacts.metrics_summary`, and
`data.artifacts.round_metrics`. `round_metrics` is optional. Missing artifact
categories should be reported from `data.missing_artifacts` without treating a
successful download as failed.

## Final Response Expectations

Report:

- final metric values, metric names, and where they came from;
- round/per-site metric paths or a clear missing-evidence note;
- model or checkpoint artifact paths when present;
- command, status, result directory, and dependency or data blockers;
- any mismatch between user-requested metrics and observed artifacts.
