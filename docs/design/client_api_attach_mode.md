# Client API Attach Mode Implementation Design

## Status

Implementation design for `execution_mode="attach"`, updated 2026-07-30.

This design assumes the Cell-based external-process Client API from PR #4906:

- `ClientAPIExecutor` delegates to execution-mode backends.
- `CellClientAPI` implements the trainer-side `flare.init()`, `receive()`,
  `send()`, and `log()` behavior.
- tasks and results travel as Cell `Shareable` objects;
- FOBS and `ViaDownloader` choose inline or lazy transfer; and
- an accepted lazy-result source remains alive until the receiver confirms a
  terminal transfer outcome.

Attach adds an externally owned trainer without changing `nvflare/fuel/f3/**`.

## Decision Summary

The CJ-side class is `AttachBackend`. The external process still uses
`CellClientAPI`; there is no new public agent or exchanger API.

```text
federation network <-> CP <-> CJ <-> Attach listener <-> external trainer
                                      owned by CJ
```

The key deployment decision is that the two Cell connections are independent:

- `comm_config.json.internal` continues to configure CP-to-CJ communication.
- `comm_config.json.client_api_attach` configures a separate listener created by
  the running CJ for CJ-to-trainer communication.

The Attach protocol is driver-independent. The dedicated listener may use an F3
network driver or the existing `shared-file` driver. A shared-filesystem trainer
therefore needs no network access.

The external trainer and job share an `attach_id`. It is a rendezvous name, like
legacy `IPCAgent.agent_id`, not a password or bearer secret. The trainer Cell is
a child of the job Cell:

```text
<site>.<job_id>.-client_api_<attach_id>
```

This topology confines ingress to the CJ-owned listener and preserves normal
hierarchical routing for lazy-result pulls:

```text
server -> CP -> CJ -> trainer
```

For shared-file Attach, the external trainer may start before the job. It waits
on a leased record keyed by `(site_name, attach_id)`; the CJ atomically publishes
its dynamic listener URL and job-specific FQCN when it starts. A direct network
profile instead contains the listener URL and `cj_fqcn`, so it is necessarily
job-specific.

## Goals

- Let an independently started process use the ordinary Client API.
- Permit either process to start first for shared-filesystem Attach.
- Let different trainers wait for different jobs using distinct attach IDs.
- Keep CP-to-CJ and CJ-to-trainer driver/security choices independent.
- Support a trainer whose only shared resource with the CJ is a filesystem.
- Reuse the #4906 Cell/FOBS transfer and result-lifetime contracts.
- Bind a job to one trainer session and reject stale or foreign sessions.
- Retry transport failures without retrying semantic task rejection.
- Never launch, signal, kill, or `waitpid` an externally owned process.
- Keep the shared F3 implementation unchanged.

## Non-Goals

- Treating `attach_id` as an authentication secret.
- Adding HMAC, challenge/response, or an Attach credential broker.
- Supporting bare-CA, server-auth-only network TLS.
- Resuming arbitrary application state after trainer failure.
- Supporting multiple active trainers or tasks per executor.
- Removing the legacy IPC/Pipe classes in this change.
- Changing `shared-file` driver behavior or its filesystem trust model.
- Making LOG delivery reliable.

## Architecture

### Backend factoring

`CellBackendBase` owns behavior common to Attach and external-process:

- CJ Cell lookup and protocol callback registration;
- one-task execution admission;
- task/result correlation and status;
- lazy decode/pass-through handling;
- task exchange settings;
- heartbeat and analytics handling; and
- closed/finalized callback gates.

The concrete backends retain different ownership behavior:

```text
CellBackendBase
├── ExternalProcessBackend
│   ├── creates the launch bootstrap
│   ├── starts the process
│   ├── monitors process liveness
│   └── SHUTDOWN -> TERM -> KILL
└── AttachBackend
    ├── starts and removes a dedicated Attach listener
    ├── optionally publishes shared-file rendezvous state
    ├── monitors session/heartbeat liveness
    ├── optionally permits reconnect
    └── SHUTDOWN only; no process action
```

Attach must not subclass `ExternalProcessBackend`, because inheriting process
ownership would make teardown unsafe.

### Listener ownership and configuration

At `START_RUN`, `AttachBackend` reads the site-local communication configuration
already available to the CJ:

```json
{
  "internal": {
    "scheme": "tcp",
    "resources": {}
  },
  "client_api_attach": {
    "scheme": "shared-file",
    "resources": {
      "root_dir": "/absolute/shared/attach",
      "connection_security": "clear"
    }
  }
}
```

The backend passes `client_api_attach.scheme` and a copy of
`client_api_attach.resources` to
`cell.core_cell.communicator.start_listener()`. The returned handle, URL, and
actual connection parameters are authoritative. The backend uses those actual
parameters for its security decision and removes that exact connector during
unwind/finalize.

This is deliberately not `create_internal_listener=True`: the ordinary
`internal` listener remains the site's general child/backbone configuration,
whereas Attach needs an isolated, job-lifetime listener.

Missing or malformed `client_api_attach` configuration fails job initialization
before any session metadata is sent.

### Trainer identity and routing

The CJ derives:

```text
trainer_fqcn = <cj_fqcn>.-client_api_<attach_id>
```

The last segment begins with `-` to retain the ad-hoc leaf convention, while the
whole name remains a child of the CJ that owns the listener. The backend sends
`SESSION_OPEN` only to this exact FQCN.

For mTLS, an externally started trainer uses the provisioned client certificate
for the site, whose certificate CN is the site name. A normal local child would
otherwise resolve its expected identity to the leaf segment. The backend
therefore installs an exact, job-lifetime identity mapping:

```text
<site>.<job_id>.-client_api_<attach_id> -> <site>
```

It refuses a conflicting existing mapping and removes only a mapping it added.
F3 then verifies the trainer's certificate before registering its claimed FQCN.

The identity remains non-secret routing metadata. It may appear in Cell headers,
logs, and lazy references.

### Shared-filesystem rendezvous

A `shared-file` listener URL contains a dynamic listener directory, so a static
trainer profile cannot contain the final URL before the job exists. Attach uses
a small discovery record above the driver:

```text
<root_dir>/.nvflare/client_api_attach/<site>/<attach_id>.claim/
├── owner.json
├── endpoint.json
└── lease
```

The endpoint contains:

- schema version;
- publisher instance ID;
- site and attach ID;
- CJ FQCN;
- derived trainer FQCN;
- complete `shared-file://` listener URL;
- `connection_security="clear"`; and
- lease timeout.

Publisher rules:

1. Claim creation is atomic.
2. A live existing claim rejects a second job using the same attach ID.
3. A stale claim may be atomically renamed to a unique tombstone and removed.
4. Record replacement is atomic.
5. The CJ refreshes the lease while the Attach backend is alive.
6. Close removes only a claim whose owner instance still matches.

Reader rules:

1. The trainer reads only the deterministic site/attach-ID claim.
2. It accepts only a live record with the expected schema, site, and attach ID.
3. `cj_fqcn` must be exactly `<site>.<job_id>`.
4. `trainer_fqcn` must equal the value derived from that CJ FQCN and attach ID.
5. The endpoint must be a valid `shared-file` URL with clear connection security.
6. Discovery plus `SESSION_OPEN` share one `job_wait_timeout` budget.

The record is discovery, not authentication. Filesystem access to the configured
root is the trust boundary.

## Trainer Profiles

Typed profiles use bootstrap schema version 1 and
`execution_mode="attach"`. They require exactly one connection source.

### Shared-file profile

```json
{
  "schema_version": 1,
  "execution_mode": "attach",
  "attach_id": "trainer_a",
  "site_name": "site-1",
  "rendezvous_dir": "/absolute/shared/attach",
  "job_wait_timeout": null
}
```

`rendezvous_dir` must be absolute. CA or TLS fields are rejected. The trainer
waits for a record before constructing its Cell, so its immutable FQCN and parent
URL are both known at construction.

### Direct network profile

```json
{
  "schema_version": 1,
  "execution_mode": "attach",
  "attach_id": "trainer_a",
  "site_name": "site-1",
  "cj_fqcn": "site-1.01234567-89ab-cdef-0123-456789abcdef",
  "connect_url": "grpcs://site-1.example.com:8102",
  "connection_security": "mtls",
  "ca_cert": "/workspace/startup/rootCA.pem",
  "job_wait_timeout": null
}
```

`cj_fqcn` must have exactly two segments and be rooted at `site_name`. Because it
contains a job ID, this direct profile must be generated for that job. This mode
does not provide the either-starts-first discovery offered by shared-file
rendezvous unless an external deployment system provisions the job identity and
URL before starting the trainer.

The mTLS credential helper locates `client.crt` and `client.key` beside
`rootCA.pem`. All three files must exist and be readable before Cell
construction. `connection_security="tls"` is rejected.

## Transport and Security

Attach makes its admission decision from the concrete parameters returned by its
own listener, never from trainer-reported labels and never from the unrelated
CP-to-CJ connector.

Supported paths are:

- `shared-file`: accepted when the FileDriver-owned root, listener directory,
  connection directory, and owner marker form a protected filesystem boundary;
- mTLS network: accepted when the actual listener is mTLS;
- clear network: accepted only with `allow_insecure_attach=True`.

An unsafe shared-file listener is rejected even when the insecure flag is set.
Bare-CA TLS is always rejected.

`allow_insecure_attach` exists because a clear network driver authenticates
neither the trainer certificate nor the transport peer. The flag is an explicit
operator acknowledgement for a trusted development network. It:

- applies only to the dedicated CJ-to-trainer listener;
- does not change `comm_config.json.internal`;
- does not turn clear transport into secure transport; and
- emits a warning when used.

Shared-file Attach does not need this flag because filesystem permissions are its
peer-access boundary.

### Threat model

`attach_id` is intentionally not a credential. Anyone who can access an
unprotected Attach transport and claim the expected trainer FQCN could impersonate
the trainer. Therefore production network Attach requires mTLS, and clear network
Attach is explicit development-only behavior.

For shared-file Attach, an actor with access to the protected root can read or
modify transport and rendezvous files. Operators must:

- use a dedicated site account;
- restrict the shared group to intended trainer principals;
- avoid world-writable roots; and
- give the CJ and trainer the same absolute path.

The trainer accepts `SESSION_OPEN` only from the prebound CJ in a direct profile
or from the CJ published in a validated rendezvous record. Header-only message
interception rejects unauthorized Attach streams before FOBS decoding or lazy
payload acquisition.

## Session Protocol

The CJ initiates the session so the trainer may wait without knowing runtime task
configuration.

### Establishment

1. The trainer constructs and starts its Cell.
2. The CJ creates one random `session_id`.
3. The CJ retries the same `SESSION_OPEN` until accepted, aborted, or
   `attach_timeout` expires. `SESSION_OPEN` is a bounded control-plane request,
   not a blob-stream request, so an early driver-level unreachable result returns
   promptly to this retry loop.
4. The request carries the attach ID, job/site identity, trainer FQCN, protocol
   version, rank, heartbeat policy, task exchange, memory settings, and whether
   trainer-hosted lazy results are relayed through the CJ.
5. The trainer validates the entire request and registers framework decomposers
   before committing any session binding.
6. It returns `SESSION_ACCEPTED`.

Duplicate `SESSION_OPEN` for the same session is idempotent. A stray, malformed,
or second-origin open returns `SESSION_REJECTED` but never latches a fatal error
or wakes a waiting `init()`. This avoids a wrong-peer availability latch.

### Task delivery

The CJ sends:

```text
TASK_READY(session_id, task_id, attempt_id, task_name, model)
```

The trainer replies with `TASK_ACCEPTED` or `TASK_FAILED`. Transport failures are
retryable with the same IDs; semantic rejection is terminal and is not retried.
`TASK_STATUS` resolves an ambiguous reply.

The trainer keeps a bounded ledger. Entries evicted from the active ledger remain
represented by a stale watermark/set so delayed duplicates are rejected rather
than requeued as new work.

### Result delivery

The trainer publishes:

```text
RESULT_READY(session_id, task_id, attempt_id, result_id, model)
```

The CJ canonicalizes only the first valid attempt for the active task. Duplicate
publication is idempotent. If the acknowledgement is lost, the trainer probes
`RESULT_STATUS` before treating session shutdown as final.

`flare.send()` returns only after the result's lazy references reach
receiver-confirmed terminal success. Until then:

- the external trainer must remain alive and attached;
- the trainer must preserve canonical result transfers across ambiguous status
  failures or routine `SHUTDOWN`; and
- its lifecycle guard must not stop the Cell while a canonical source is still
  being served.

The expected source receiver is a session property supplied by the CJ. In a
secure relayed job it is the CJ; otherwise it is the ultimate receiver stamped
on the task. This choice is independent of the Attach listener driver and its
connection security: a clear `shared-file` trainer route may still feed a
secure CP/CJ route whose CJ performs the relay.

If tensor-streaming filters are also configured, a result that already contains
pass-through lazy references bypasses their separate tensor rendezvous. The
client stamps that decision in the result envelope because the server may eagerly
resolve the references before its tensor-stream filter runs. The filters preserve
the complete result envelope so its terminal consumer resolves the references
through the ordinary downloader path.

After acceptance, transient status-probe failure must not delete transfer
sources. Reusing a result ID for another `send()` is rejected explicitly.

### Liveness and reconnect

Heartbeat liveness replaces process monitoring. Attach requires either a positive
`heartbeat_timeout` or a finite `result_wait_timeout`, so a dead trainer cannot
leave post-acceptance result waiting unbounded.

If `allow_reconnect=False`, loss ends the session. If true and no task is active,
the backend creates a fresh session and attach-timeout budget for the same trainer
FQCN. It never silently replays an ambiguous active task.

## Lifecycle

Initialization owns:

- pass-through registration;
- the exact mTLS identity mapping, when added;
- the dedicated Attach listener;
- the shared-file endpoint claim, when used; and
- the session-monitor daemon thread.

Failure unwinds them in reverse without touching the external process.

Finalization:

1. marks the backend closed;
2. asks an established trainer to shut down its session;
3. waits briefly for the monitor thread;
4. removes the endpoint claim;
5. removes the exact listener connector;
6. removes the identity mapping it added; and
7. disables pass-through.

It never sends OS signals, terminates a process group, calls `waitpid`, or assumes
the trainer's PID exists.

## Public Configuration

`ClientAPIExecutor` Attach arguments include:

- `attach_id`;
- `attach_timeout`;
- `allow_reconnect`;
- `allow_insecure_attach`;
- heartbeat, task, and result timeouts; and
- ordinary Client API exchange settings.

Process-ownership arguments remain invalid:

- `command`;
- `launch_once`;
- `launch_timeout`;
- `shutdown_timeout`; and
- `stop_grace_period`.

The literal attach ID may be delivered through normal server-distributed job
configuration because it is only a name. No Attach secret is defined.

## Implementation Map

- `nvflare/app_common/executors/client_api/cell_backend.py`
  contains shared Cell protocol machinery.
- `nvflare/app_common/executors/client_api/attach_backend.py`
  owns the listener, rendezvous publisher, session retry/reconnect, and
  non-owning lifecycle.
- `nvflare/client/cell/attach.py`
  validates attach IDs, derives the trainer FQCN, and validates direct URLs.
- `nvflare/client/cell/attach_rendezvous.py`
  implements leased shared-file endpoint discovery.
- `nvflare/client/cell/bootstrap.py`
  validates direct and rendezvous profile forms.
- `nvflare/client/cell/attach_session.py`
  resolves discovery, validates `SESSION_OPEN`, and implements trainer-side
  task/result attempts.
- `nvflare/client/cell/api.py`
  constructs the Cell only after connection resolution and preserves Attach
  runtime ownership.

No change is required under `nvflare/fuel/f3/**`.

## Verification

Unit coverage must verify:

- `internal` and `client_api_attach` are independent;
- missing/malformed Attach listener configuration fails before `SESSION_OPEN`;
- listener classification uses returned concrete parameters;
- shared-file permission checks and claim collision/staleness behavior;
- direct profiles require a valid job-specific CJ FQCN;
- rendezvous records derive the same CJ-child trainer FQCN on both sides;
- mTLS requires readable CA/client cert/client key and exact identity binding;
- clear network requires the explicit flag, while shared-file does not;
- SESSION_OPEN rejection cannot poison `init()`;
- task delivery does not retry semantic rejection;
- reconnect, stale-session rejection, task deduplication, and result recovery;
- trainer-first and CJ-first rendezvous over each supported Attach driver;
- secure-job result relay is independent of clear shared-file Attach transport;
- canonical lazy sources survive uncertain status and routine shutdown; and
- Attach finalization never invokes process-ownership operations.

The fast integration test must run a real trainer subprocess and real Cells,
with both trainer-first and CJ-first startup, for:

1. a clear TCP listener with explicit insecure opt-in; and
2. clear gRPC and HTTP listeners with explicit insecure opt-in; and
3. `shared-file` with a trainer audit hook that rejects all socket operations.

All cases send a NumPy task, accept a lazy result, and have the CP pull the
result through the CJ from the trainer. This verifies the load-bearing
CJ-child routing and source-lifetime path.

Regression gates:

- external-process tests remain green;
- the project style check passes; and
- `git diff -- nvflare/fuel/f3` is empty.
