# NVFLARE CLI Agent Readiness Backlog

This backlog tracks the remaining larger CLI enhancement groups after the
`--submit-token` retry/recovery work.

## Already Completed

- Per-command startup-kit selectors: `--kit-id` / `--startup-kit`
- `job logs --tail`, `--since`, and `--max-bytes`
- `job submit --submit-token` and `job list --submit-token`
- `nvflare job wait`
- `poc prepare` port and startup-kit preflight
- `poc start` readiness, bound addresses, and port conflict metadata

## 1. CLI Contract

- Rich `--schema` fields: `output_modes`, `streaming`, `idempotent`, `mutating`,
  and related command metadata.
- `--format jsonl` support for streaming commands.
- `recovery_category` in error envelopes.

## 2. Job Lifecycle

- `job monitor` JSONL stream contract: progress events, `terminal: true`, and
  structured timeout event.
- `job download` artifact/path contract: `download_path`, `artifacts`, and
  `missing_artifacts`.

## 3. POC Workflow

- `poc wait-ready` as an explicit readiness command for workflows that start with
  `poc start --no-wait`.
- `poc stop --restore-kit`.

## 4. Recipe Discovery

- `recipe show`
- `recipe list --filter`
- Structured recipe metadata: algorithm, privacy, framework, state exchange, and
  related selection fields.

## 5. Study / Production Preflight

- Enrich `study list` with role, capabilities, `can_submit_job`, and denial
  reason fields.

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

## 8. Diagnostics

- Log source boundary markers: `[USER_CODE_EXCEPTION]` and `[FLARE]`.
