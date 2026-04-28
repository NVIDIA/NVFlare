# NVFLARE CLI Enhancement Extra Requirements for Agent Readiness

**Date:** 2026-04-27

**Scope:** Existing NVFLARE CLI commands only. This note excludes the new
`nvflare agent ...` namespace and focuses on enhancements needed in existing
command groups such as `job`, `poc`, `recipe`, `study`, `system`,
`config kit`, and `cert`.

## Goal

Make the existing NVFLARE CLI reliable for coding agents, notebooks, CI, and
scripted workflows. Agents need stable JSON contracts, retry-safe operations,
bounded outputs, and explicit state. These requirements are mostly small
contract additions to commands that already exist or are already planned.


## 1. Shared CLI Schema Contract

Commands used by agents should support `--schema` and return a normalized JSON
object, not prose help text or raw argparse internals.

Suggested minimum shape:

```json
{
  "schema_version": "1",
  "command": "nvflare job submit",
  "description": "Submit a job to the selected FLARE server.",
  "arguments": [],
  "flags": [
    {"name": "job-folder", "short": "j", "type": "path", "required": true},
    {"name": "format", "type": "enum", "values": ["json"], "required": false},
    {"name": "idempotency-key", "type": "string", "required": false}
  ],
  "output_modes": ["json"],
  "streaming": false,
  "idempotent": false,
  "idempotency_key_supported": true,
  "mutating": true,
  "examples": [
    "nvflare job submit -j ./jobs/hello-pt --idempotency-key <uuid> --format json"
  ]
}
```

Why needed:
Agents and skill validators must discover command flags, output modes, and
retry safety without scraping `--help`. This also lets contract tests catch
breaking CLI changes before release.

Suggested behavior:
- `--schema` succeeds without required positional arguments.
- Every public agent-used command declares `streaming`, `output_modes`,
  `idempotent`, `idempotency_key_supported`, and `mutating`.
- Schema output remains backward-compatible within a schema version.


## 2. Shared JSON and JSONL Output Contract

Single-result commands should use:

```text
--format json
```

and emit exactly one JSON envelope on stdout.

Streaming commands should use:

```text
--format jsonl
```

and emit one complete JSON event per line.

Why needed:
Agents need to parse outputs deterministically. A single JSON envelope is good
for status and submit operations. JSONL is needed for progress streams such as
job monitoring.

Suggested behavior:
- Human progress and diagnostics go to stderr in JSON/JSONL mode.
- JSON envelopes include stable status/code/message/hint fields.
- Error envelopes include `recovery_category` so agents know whether to retry,
  fix config, fix code, ask the user, or stop.


## 3. Scoped Startup-Kit Selection for Existing Admin Commands

Existing online admin commands should support a non-mutating identity selector:

```text
--kit-id <id>
```
```text
--startup-kit <path>
```

This should apply to `nvflare job ...`, `nvflare system ...`, and
`nvflare study ...`.

Meaning:
The command uses the selected startup kit for this invocation only. It does not
change the global active kit stored in `~/.nvflare/config.conf`.

Why needed:
`nvflare config kit use` mutates global user state. That is unsafe for agents,
Jupyter notebooks, and concurrent workflows because one process can silently
change the identity used by another process.

Suggested behavior:
- Per-command selector overrides the human default active kit.
- The global config file is not modified.
- JSON output reports which kit identity was used.
- `config kit use` remains available as a human convenience, not as the agent
  workflow primitive.


## 4. `nvflare job submit -j`: Idempotency Key

Add:

```bash
nvflare job submit -j <job_folder>/<job_name> --idempotency-key <uuid> --format json
```

Meaning:
An idempotency key is a client-generated unique token for one intended submit
operation. If the client retries with the same key, the server returns the
already-accepted job instead of creating a duplicate.

Why needed:
Agents retry commands after timeouts and connection failures. If `job submit`
times out after the server accepted the job, a retry without an idempotency key
can submit the same job twice.

Suggested behavior:
- Server records accepted submit requests by idempotency key for a documented
  time window.
- Reusing the same key with the same job returns the existing `job_id`.
- Reusing the same key with different job content returns a clear error.
- Response includes `job_id` and `idempotency_key`.
- `--schema` marks the command as `idempotent: false` and
  `idempotency_key_supported: true`.


## 5. `nvflare job list` and `nvflare job meta`: Job Recovery

Add:

```bash
nvflare job list --idempotency-key <uuid> --format json
```

and include `idempotency_key` in:

```bash
nvflare job meta <job_id> --format json
```

Why needed:
If an agent crashes after sending submit but before receiving `job_id`, the
next session can recover by searching with the idempotency key written before
submit.

Suggested behavior:
- `job list --idempotency-key` returns matching jobs only.
- `job meta` includes `idempotency_key` when present.
- If no match exists, return a structured `JOB_NOT_FOUND` or equivalent code.


## 6. `nvflare job wait`: Single-Envelope Job Completion

Add or standardize:

```bash
nvflare job wait <job_id> --timeout <seconds> --format json
```

Meaning:
Wait blocks until the job reaches a terminal state or timeout, then returns one
JSON envelope.

Why needed:
Agents often need one final result rather than a progress stream. This keeps
single-result JSON simple and avoids forcing agents to parse streaming output.

Suggested behavior:
- Terminal states: `COMPLETED`, `FAILED`, `ABORTED`.
- Timeout returns a structured timeout status and exit code.
- Output includes final job status and key metadata.


## 7. `nvflare job monitor`: Streaming Progress Contract

Add or standardize:

```bash
nvflare job monitor <job_id> --timeout <seconds> --format jsonl
```

Meaning:
Monitor emits progress events as JSONL and exits when the job reaches a
terminal state or timeout.

Why needed:
Agents need progress updates for long-running jobs without violating the
single-envelope JSON contract.

Suggested behavior:
- Each line is one complete JSON object.
- Terminal states: `COMPLETED`, `FAILED`, `ABORTED`.
- Final event always includes `terminal: true`.
- Timeout emits a final event with `status: "TIMEOUT"` and `terminal: true`.
- Schema declares `streaming: true` and `output_modes: ["jsonl"]` or
  `["json", "jsonl"]` if both are supported.


## 8. `nvflare job logs`: Bounded Logs and Partial Availability

Add:

```bash
nvflare job logs <job_id> --site all --tail 200 --format json
```
```bash
nvflare job logs <job_id> --site site-1 --since <timestamp> --format json
```
```bash
nvflare job logs <job_id> --site all --max-bytes <bytes> --format json
```

Meaning:
Logs are returned in bounded slices and include per-site availability metadata.

Why needed:
Multi-site production logs can be huge and can exceed an agent's context
window. In multi-org deployments, some sites may not expose logs at all.
Agents must know when evidence is partial.

Suggested behavior:
- Without `--tail`, `--since`, or `--max-bytes`, return at most 500 lines per
  site.
- Include `logs_truncated: true` when output is capped.
- Include per-site availability:

```json
{
  "sites": {
    "site-1": {"available": true, "lines": 200},
    "site-2": {"available": false, "reason": "log_forwarding_disabled"}
  }
}
```

- Diagnosis skills should request explicit bounds such as `--tail 200`.


## 9. `nvflare job download`: Artifact Path Contract

Add:

```bash
nvflare job download <job_id> --output-dir <path> --format json
```

Default output directory should be `./<job_id>/` when not specified.

Meaning:
Download returns machine-readable paths to key artifacts instead of requiring
agents to guess directory layout.

Why needed:
After download, agents need to find the trained global model, metrics summary,
and logs. Exit code 0 alone does not tell an agent where the result is.

Suggested response data:

```json
{
  "download_path": "./job-abc123",
  "artifacts": {
    "global_model": "job-abc123/server/models/global_model.pt",
    "metrics_summary": "job-abc123/server/metrics_summary.json",
    "client_logs": {
      "site-1": "job-abc123/site-1/log.txt"
    }
  },
  "missing_artifacts": []
}
```

Suggested behavior:
- Use returned `data.artifacts.*` as the source of truth.
- Include missing expected artifacts in `missing_artifacts`.
- Do not require the agent to know internal workspace layout.


## 10. `nvflare poc prepare`: Port and Kit Preflight

Enhance:

```bash
nvflare poc prepare --format json
```

Requirements:
- Preflight default port availability when possible.
- Report conflicts before start.
- Report the prior active startup-kit id if POC prepare changes the human
  default kit.

Why needed:
Agents must know whether POC can start cleanly and must be able to restore the
user's prior identity after POC workflows.


## 11. `nvflare poc start`: Readiness and Bound Addresses

Enhance:

```bash
nvflare poc start --format json
```

Requirements:
- Either block until server/admin endpoints are ready, or pair with
  `nvflare poc wait-ready`.
- Return actual bound server/admin addresses.
- Return port conflict metadata.

Suggested response data:

```json
{
  "server_address": "localhost:6011",
  "admin_address": "localhost:6012",
  "default_port": 6009,
  "port_conflict": true,
  "warnings": ["Default port 6009 was in use; POC started on 6011"]
}
```

Why needed:
If POC falls back to alternate ports, an agent must not accidentally verify or
submit to a different running FLARE server.


## 12. `nvflare poc wait-ready`: Deterministic POC Readiness

Add, unless `poc start --format json` becomes blocking:

```bash
nvflare poc wait-ready --timeout <seconds> --format json
```

Meaning:
Wait until POC services are reachable or return a structured timeout.

Why needed:
POC start is a background-process workflow. Agents need a deterministic
readiness point before running online doctor or job submit.


## 13. `nvflare poc stop`: Restore Prior Startup Kit

Add:

```bash
nvflare poc stop --restore-kit --format json
```

Meaning:
Stop POC and restore the startup kit that was active before POC prepare/start.

Why needed:
POC workflows should not leave users on a stopped POC admin identity. This is
especially important when a user had a production kit active before running POC.

Suggested behavior:
- `poc prepare` records prior kit id.
- `poc stop --restore-kit` restores it.
- JSON output reports restored kit id and warnings.


## 14. `nvflare recipe show`: Queryable Recipe Metadata

Add:

```bash
nvflare recipe show <name> --format json
```

Why needed:
Agents cannot choose a recipe from names alone. They need structured metadata
about framework support, algorithm, privacy compatibility, and parameters.

Suggested fields:
- `name`
- `algorithm`
- `aggregation`
- `client_requirements`
- `framework_support`
- `heterogeneity_support`
- `privacy_compatible`
- `parameters`
- `optional_dependencies`
- `template_references`


## 15. `nvflare recipe list`: Filters

Enhance:

```bash
nvflare recipe list --filter framework=pytorch --filter privacy=differential_privacy --format json
```

Why needed:
Agents should query suitable recipes directly instead of listing everything and
guessing by keyword.

Suggested filters:
- `framework`
- `privacy`
- `algorithm`
- `aggregation`
- `state_exchange`


## 16. `nvflare study list`: Production Submit Preflight

Enhance:

```bash
nvflare study list --format json
```

Requirements:
- Include authorized studies visible to the selected identity.
- Include role/capability information.
- Include `can_submit_job: true|false`.
- Include reason when submit is not allowed.

Why needed:
Agents should fail before production submit if the selected identity cannot
submit to the target study. This avoids long workflows ending in a predictable
authorization error.


## 17. `nvflare system ...`: Scoped Identity and JSON/Schema Contract

Existing `system` commands used by agents, such as status/resources/version,
should support:

```text
--kit-id <id>
```
```text
--startup-kit <path>
```
```text
--format json
```
```text
--schema
```

Why needed:
Agents use system status/resources/version for readiness and diagnosis. These
checks must use the same selected identity as job submission and must not depend
on global active-kit state.


## 18. `nvflare config kit show/list`: Identity Discovery

Existing commands should provide stable JSON output:

```bash
nvflare config kit list --format json
```
```bash
nvflare config kit show --format json
```

Why needed:
Agents need to discover registered startup kits and current human defaults. But
they should not rely on `config kit use` for workflow execution because it
mutates global state.

Suggested behavior:
- `show/list` return kit id, path, identity, role/org/project when available,
  certificate expiration status, and stale-path findings.
- `config kit use` remains human-facing and should warn when used in an agent
  workflow.


## 19. `nvflare cert request`: Request Expiration Metadata

Enhance:

```bash
nvflare cert request ... --format json
```

Requirements:
- Return `request_dir`.
- Return `request_id`.
- Return `expires_at`.
- Return warning text when lead approval must happen before a deadline.

Why needed:
Distributed provisioning approval is asynchronous and can take hours or days.
Agents must know whether a request is still valid when resuming.


## 20. `nvflare cert request-status`: Async Approval State

Add:

```bash
nvflare cert request-status --request-dir <path> --format json
```

Requirements:
- Return status: `pending`, `approved`, `rejected`, or `expired`.
- Return `request_id`.
- Return `expires_at`.
- Return `time_remaining_hours`.

Why needed:
The site-side distributed provisioning workflow should stop after request
creation and resume only after approval. The site agent should not run
`cert approve`; that is the lead's action.

Suggested behavior:
- If less than one hour remains, return a warning/finding such as
  `CERT_REQUEST_EXPIRING_SOON`.
- If expired, return a structured error or status with guidance to regenerate
  the request.


## 21. `nvflare cert renew`: Certificate Renewal

Follow-on enhancement:

```bash
nvflare cert renew --kit-id <id> [--extend-days 90] --format json
```

Why needed:
`agent doctor --online` can warn that a startup kit certificate is expiring,
but warnings need an actionable recovery path.

Suggested behavior:
- Return renewed/replaced kit id and expiration.
- Fail with a clear authorization error if the selected identity cannot renew.
- This can be promoted when distributed-provisioning lifecycle work is ready.


## 22. Export Manifest Validation in `nvflare job submit`

Enhance `nvflare job submit -j` to inspect exported job folders for:

- `_export_manifest.json`
- `job_fingerprint.json`
- required files listed in the manifest
- source freshness hash when available
- `poc_validated` status for production/study submission

Why needed:
Agents submit exported job folders. They need early validation that the export
is complete, fresh, and suitable for the selected environment.

Suggested behavior:
- Missing manifest: warn initially; later make strict if adopted as a formal
  export contract.
- Missing required files: fail with `JOB_VALIDATION_FAILED`.
- `poc_validated: false` on production submission: warn or require explicit
  confirmation depending on safety policy.


## 23. Log Source Boundary for Diagnosis

When FLARE executes user training code, logs should distinguish user-code
exceptions from FLARE framework/runtime errors.

Suggested behavior:
- User training exceptions caught at the boundary are logged with
  `[USER_CODE_EXCEPTION]`.
- FLARE framework/runtime errors are logged with `[FLARE]`.
- Diagnosis output can set source to `user_code`, `flare_framework`, or
  `unknown`.

Why needed:
Without this boundary, agents may classify a user training exception as a FLARE
environment failure, or classify a framework/runtime issue as fixable user
code.


## Priority Summary

P0:
- Concrete `--schema` output format.

P1:
- `job submit --idempotency-key`
- `job wait`
- `job monitor --format jsonl` terminal/timeout contract
- bounded `job logs`
- `job download --output-dir` artifact paths
- POC readiness/bound-address behavior
- scoped `--kit-id` / `--startup-kit`
- recipe show/list filters
- study submit preflight

P2:
- cert request-status and request expiration
- cert renew
- export manifest validation in job submit
- log source boundary markers
