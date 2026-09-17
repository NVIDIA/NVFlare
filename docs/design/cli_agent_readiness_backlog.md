# NVFLARE CLI Agent Readiness Backlog

This backlog tracks the remaining larger CLI enhancement groups after the
`--submit-token` retry/recovery work.

## Already Completed

- Per-command startup-kit selectors: `--kit-id` / `--startup-kit`
- `job logs --tail`, `--since`, and `--max-bytes`
- `job submit --submit-token` and `job list --submit-token`
- `nvflare job wait`
- `poc prepare` port and startup-kit preflight
- `poc start` readiness, bound addresses, port conflict metadata, and `--timeout`
- `recipe list --filter` for all documented built-in recipe variants, with
  list-time metadata for `framework`, `privacy`, `algorithm`, `aggregation`,
  and `state_exchange`
- `recipe show <name>` for all documented built-in recipe variants, with
  queryable metadata, constructor parameters, optional dependencies, privacy
  compatibility, and framework support
- `study list` identity, startup-kit metadata, visible-study details, and
  membership-level `can_submit_job`
- `--format jsonl` support for `nvflare job monitor`
- `job monitor` JSONL stream contract: progress events, `terminal: true`, and
  structured timeout event

## 1. CLI Contract

- Rich `--schema` fields beyond monitor `output_modes` and `streaming`:
  `idempotent`, `mutating`, and related command metadata.
- `recovery_category` in error envelopes.

## 2. Job Lifecycle

- `job download` artifact/path contract: `download_path`, `artifacts`, and
  `missing_artifacts`.

## 3. POC Workflow

- `poc stop --restore-kit`.

## 4. Recipe Discovery

- Richer recipe compatibility guidance and packaged template/example references
  as recipe classes add explicit metadata.

## 5. Study / Production Preflight

- Full authorization-policy introspection for production submit preflight when
  server-side policy reporting is added.

## 6. Cert / Provisioning Lifecycle

- `cert request` expiration metadata.
- `cert request-status`
- `cert renew`

## 7. Exported Job Validation

- `_export_manifest.json`
- `job_fingerprint.json`
- Required-file checks.
- Source freshness checks.
- `poc_validated` warning or approval behavior.
