# Hierarchical FL BERT-144 Analysis and Remediation Report

## 1) Analysis Summary

The failure pattern is a startup-stability issue under hierarchical scale (1 server, 6 relays, 144 clients) with heavy model initialization (BERT), not a core aggregation math bug.

Observed chain:

1. Start-job fanout receives late/missing replies from a subset of clients during startup.
2. Server-side heartbeat sync sees some clients not reporting the new `JOB_ID` yet and marks them as "missing job on client."
3. Those clients are treated as dead/disconnected after grace handling.
4. Deployment policy check fails when alive clients drop below required `min_sites` (effectively 144 in this setup).
5. Run abort leads to follow-on errors: empty aggregation and downstream `None`-type issues.

This is why CIFAR10 can pass at higher client counts while BERT fails on the same topology: the startup latency profile is materially different.


## 2) Root Causes

### RC-1: Strict policy sensitivity at startup window

- Current job policy (`min_sites` effectively 144/144) leaves no tolerance for temporary startup lag.
- A small number of delayed/unstable clients immediately violates policy.

### RC-2: Premature missing-job classification during startup

- `_sync_client_jobs` can classify a client as missing the job before that client has ever positively reported the job in heartbeat.
- In high-latency startup, this can create false early dead-job reports.

### RC-3: Start-job reply handling not strict by default

- If start replies are delayed or partially failed, run start can continue without fully healthy confirmation unless strict validation is enabled.

### RC-4: Client metadata robustness gap

- `get_job_clients()` previously assumed `JOB_CLIENTS` exists and is iterable.
- When metadata is malformed/missing (often after abort/disturbed startup), client can crash with non-actionable `TypeError`.


## 3) Critical Issues (Prioritized)

### P0 - Run abort due to policy violation at scale

- Symptom: `Alive clients < required min` followed by system panic/abort.
- Impact: Training cannot progress past early rounds.

### P0 - Startup false negatives from heartbeat job-sync

- Symptom: `missing job on client` and dead-client handling during initial startup.
- Impact: Cascades into policy violation even when clients may be only delayed.

### P1 - Weak startup reply validation

- Symptom: late/no replies tolerated in legacy flow.
- Impact: System proceeds with unstable participant set.

### P1 - Poor client-side error resilience for job metadata

- Symptom: `TypeError: 'NoneType' object is not iterable`.
- Impact: Masks root trigger and reduces diagnosability.


## 4) Configuration-Only Recommendations (No Code Changes)

These are low-risk operational mitigations you can apply immediately.

### A. Relax startup-sensitive policy for large BERT runs

- Set `min_sites` below total planned clients (for example 135-140 for 144 total).
- Keep `num_clients` as desired sampling target, but avoid requiring 100% alive at all times.

### B. Increase startup and sync tolerance

- Increase runner synchronization windows (`runner_sync_timeout`, `max_runner_sync_timeout`, and retry count where configured).
- Increase dead-client grace and lead-time (`dead_client_grace_period`, `dead_client_check_lead_time`) to reduce false disconnect during cold start.

### C. Reduce relay/node startup contention

- Lower clients per GPU/node for BERT startup (operational test: halve concurrency and compare startup success).
- Stagger launch or warm cache/model artifacts where possible.

### D. Enable strict/debounced toggles in hierarchical environments

- Set `strict_start_job_reply_check=true` for production validation of startup health.
- Set `sync_client_jobs_require_previous_report=true` to prevent startup false positives.
- Keep defaults `false` for legacy compatibility in non-hierarchical environments unless needed.


## 5) Suggested Code Changes and Implementation (With Inline Comments)

Below are the recommended changes (already aligned with current patch direction), shown with inline comments explaining intent.

### 5.1 Add opt-in config vars for safer rollout

File: `nvflare/apis/fl_constant.py`

```python
# server: require all start-job replies to be non-timeout and OK before considering the run started
STRICT_START_JOB_REPLY_CHECK = "strict_start_job_reply_check"

# server: require prior positive job observation before reporting "missing job on client" as dead-job
SYNC_CLIENT_JOBS_REQUIRE_PREVIOUS_REPORT = "sync_client_jobs_require_previous_report"
```

Why:
- Keeps behavior opt-in and backward compatible.
- Avoids surprise regressions for standard/non-hierarchical deployments.

### 5.2 Harden start-job reply validation behind strict toggle

File: `nvflare/private/fed/server/admin.py`

```python
def check_client_replies(replies, client_sites, command, strict=False):
    # Legacy mode: preserve existing behavior unless strict mode is explicitly enabled.
    if strict:
        replies_by_client = {r.client_name: r for r in replies}
        missing_clients = [c for c in client_sites if c not in replies_by_client]
        if missing_clients:
            # Explicitly fail if any expected client did not reply.
            raise RuntimeError(...)

        for client_name in client_sites:
            r = replies_by_client[client_name]
            if not r.reply:
                # Fail on timeout/no-reply in strict mode.
                ...
            return_code = r.reply.get_header(MsgHeader.RETURN_CODE, ReturnCode.OK)
            if return_code != ReturnCode.OK:
                # Fail on non-OK return code in strict mode.
                ...
```

Why:
- Converts startup ambiguity into explicit health gating when enabled.
- Prevents starting a run with hidden startup failures.

### 5.3 Wire strict toggle at run-start call site

File: `nvflare/private/fed/server/job_runner.py`

```python
strict_start_reply_check = ConfigService.get_bool_var(
    name=ConfigVarName.STRICT_START_JOB_REPLY_CHECK,
    conf=SystemConfigs.APPLICATION_CONF,
    default=False,  # Preserve legacy behavior unless explicitly enabled.
)

check_client_replies(
    replies=replies,
    client_sites=client_sites_names,
    command=f"start job ({job_id})",
    strict=strict_start_reply_check,  # Controlled rollout through config.
)
```

Why:
- Centralized toggle-based control with safe default.

### 5.4 Debounce missing-job detection during startup

File: `nvflare/private/fed/server/fed_server.py`

```python
require_previous_report = ConfigService.get_bool_var(
    name=ConfigVarName.SYNC_CLIENT_JOBS_REQUIRE_PREVIOUS_REPORT,
    conf=SystemConfigs.APPLICATION_CONF,
    default=False,  # Legacy immediate behavior kept by default.
)

# Record first positive observation of job report from this client.
reported_clients = job_info.setdefault("_reported_clients", set())
reported_clients.add(client_token)

# Only notify dead-job immediately in legacy mode OR after prior positive observation.
if (not require_previous_report) or (client_token in reported_clients):
    self._notify_dead_job(client, job_id, "missing job on client")
```

Why:
- Removes startup false positive dead-job notifications.
- Still preserves legacy behavior unless toggle is turned on.

### 5.5 Harden client metadata parsing

File: `nvflare/private/fed/client/client_run_manager.py`

```python
job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
if not isinstance(job_meta, dict):
    # Fail fast with actionable message instead of TypeError later.
    raise RuntimeError(...)

job_clients = job_meta.get(JobMetaKey.JOB_CLIENTS)
if job_clients is None:
    # Clear error for missing metadata field.
    raise RuntimeError(...)
if not isinstance(job_clients, list):
    # Clear error for wrong metadata type.
    raise RuntimeError(...)
```

Why:
- Turns opaque crash into deterministic, diagnosable runtime error.


## 6) Test Strategy

Goal: prove each behavior in both positive and negative directions, and prove backward compatibility.

### Unit tests added/updated

1. `tests/unit_test/private/fed/server/admin_test.py`
   - Legacy mode allows timeout reply (backward compatibility).
   - Strict mode rejects timeout.
   - Strict mode rejects non-OK return code.
   - Strict mode rejects missing expected client reply.
   - Strict mode accepts reordered successful replies.

2. `tests/unit_test/private/fed/server/fed_server_test.py`
   - Legacy mode: missing job is reported immediately.
   - Debounced mode: missing job is reported only after prior positive job report.

3. `tests/unit_test/private/fed/server/job_runner_test.py`
   - Toggle disabled -> `check_client_replies(..., strict=False)`.
   - Toggle enabled -> `check_client_replies(..., strict=True)`.

4. `tests/unit_test/private/fed/client/client_run_manager_test.py`
   - Missing `JOB_CLIENTS` raises `RuntimeError`.
   - Wrong `JOB_CLIENTS` type raises `RuntimeError`.
   - Empty list is accepted as valid shape.

### Execution recommendation

- Primary targeted suite:
  - `tests/unit_test/private/fed/server/`
  - `tests/unit_test/private/fed/client/client_run_manager_test.py`
- Keep full-repo tests optional in constrained environments where unrelated native deps may cause segmentation faults.

### Additional integration validation (recommended)

For hierarchical staging:

1. Run with both toggles disabled (baseline legacy behavior).
2. Enable only `strict_start_job_reply_check`; verify startup fail-fast diagnostics.
3. Enable only `sync_client_jobs_require_previous_report`; verify reduced early dead-job false positives.
4. Enable both toggles; validate improved startup stability and reduced policy-triggered aborts.

Success criteria:

- Fewer/no early `missing job on client` dead reports before first positive heartbeat for each client.
- No silent startup with timeout/non-OK replies when strict mode is enabled.
- Reduced run aborts from transient startup lag under BERT load.


## 7) Rollout Guidance

- Phase 1: apply config-only mitigations first.
- Phase 2: enable debounced sync toggle in hierarchical deployments.
- Phase 3: enable strict start-reply check in controlled staging, then production.
- Keep defaults backward compatible for non-hierarchical workloads unless local policy requires strict startup guarantees.

