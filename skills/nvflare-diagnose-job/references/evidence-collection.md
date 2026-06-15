# Evidence Collection

Diagnosis is mode-aware. Simulation failures usually have local artifacts and no
admin server lifecycle. POC and production failures use the FLARE job/system CLI
because a server and sites exist.

## Mode Decision

| Mode | Evidence In Prompt | Evidence Source |
| --- | --- | --- |
| Simulation | `job.py`, SimEnv, local stdout/stderr, exported job folder, failed `python job.py` | Local files and command output |
| POC/production | job ID, startup kit, POC workspace, admin context, server/client status | `nvflare job` and `nvflare system` CLI |
| Ambiguous | "failed job" with no ID/path/logs | Ask for mode plus one concrete evidence source |

Treat POC as production-mode diagnosis because the same server, client, admin,
job, and log surfaces are used, even when everything runs on localhost.

## Simulation Evidence

Use local evidence only:

- failed command and stdout/stderr, usually `python job.py`;
- `job.py`, recipe settings, generated config, and exported job folder when
  available;
- SimEnv workspace, result directory, and site/server logs produced by the
  simulation;
- server-side simulation metrics artifacts, when present, under the SimEnv
  workspace's `server/simulate_job/metrics/` directory. Use
  `metrics_summary.json` for final/best aggregate metrics and
  `round_metrics.jsonl` for per-round and per-site evidence;
- local dependency, dataset, and path evidence supplied by the user.

Use `nvflare --format json agent inspect <path>` when the user provides a
project, job, or exported-job path. Do not use `nvflare job` or
`nvflare system` commands for a pure simulation failure unless the user also
provides a POC/production job ID or startup-kit context.

## POC And Production Evidence

Use bounded CLI evidence. Prefer JSON envelopes for agent-readable output:

```bash
nvflare --format json job meta <job_id>
nvflare --format json job logs <job_id> --site all --tail 200
nvflare --format json job stats <job_id> --site all
nvflare --format json job download <job_id> -o <download-dir>
nvflare --format json system status
nvflare --format json system resources
```

When the user supplies a startup kit or registered kit ID, add the matching
`--startup-kit <path>` or `--kit-id <id>` option to each command that needs
server access. When logs are large or the failure window is known, prefer
`--since <timestamp>` or `--max-bytes <bytes>` in addition to or instead of
`--tail`.

For terminal POC/production jobs, use the JSON response from
`nvflare job download` as the source of truth for downloaded artifacts. Inspect
`data.artifacts.global_model`, `data.artifacts.metrics_summary`, and
`data.artifacts.round_metrics` when present, and report `data.missing_artifacts`
when metrics or model artifacts are unavailable. Do not infer production
artifact paths from simulation workspace conventions.

## Evidence Quality

Track whether each evidence source is complete, partial, unavailable, or
permission-denied. Treat these as findings:

- `PARTIAL_LOG_VISIBILITY`: one or more expected site logs are missing,
  permission-denied, unavailable, or not streamed to the server.
- `LOGS_TRUNCATED`: log output reports truncation or the selected bound may
  omit earlier context.
- `MODE_AMBIGUOUS`: the prompt does not identify simulation versus
  POC/production and no concrete evidence path or job ID is available.

Use source labels from the evidence when available: server, site name, client
name, subprocess, user training code, or FLARE runtime. If markers such as
`[USER_CODE_EXCEPTION]` or `[FLARE]` are present, preserve them in the evidence
summary. If no marker or source label is visible, mark the source as `unknown`
rather than inferring user code or FLARE runtime ownership.

## Diagnosis Report

Report the diagnosis as:

1. mode and evidence collected;
2. failure status and affected sites;
3. matched pattern and confidence;
4. likely cause with source-aware evidence;
5. recovery category;
6. concrete next action;
7. missing evidence or follow-up command when confidence is limited.
