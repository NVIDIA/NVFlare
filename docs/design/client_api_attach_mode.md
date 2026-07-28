# Client API Attach Mode Implementation Design

## Status

**Proposed implementation design, 2026-07-21.**

This design assumes NVIDIA/NVFlare PR #4906 has merged. In particular, it assumes
the following are already the supported out-of-process Client API architecture:

- `ClientAPIExecutor` delegates to internal execution-mode backends.
- `ExternalProcessBackend` communicates directly with the trainer over Cell.
- `CellClientAPI` implements the trainer-side `flare.init()` / `receive()` /
  `send()` / `log()` behavior.
- Task and result `Shareable` objects travel directly in Cell requests. FOBS and
  `ViaDownloader` select inline or lazy transfer without a Client-API-specific
  payload wrapper.
- An accepted lazy result source remains alive until its receiver-confirmed
  `DownloadService` transfer outcome is terminal.

The goal of this change is to implement the reserved `execution_mode="attach"`.
It is not a redesign of the external-process data path and it does not remove the
legacy IPC or Pipe implementations.

## Motivation

Today an externally started trainer — one that NVFlare does not launch — must use
`IPCAgent` with a CJ-side `IPCExchanger`. That is a different programming model
(an agent object exposing `get_task`/`submit_result`) on the older `flare_agent`
/ Pipe transport, with its own retry and payload semantics.

Attach lets the same self-started process use the **ordinary Client API**
(`flare.init()/receive()/send()/log()`) — the identical script a user already
writes for in-process or launched (external-process) mode — while the CJ simply
declares `ClientAPIExecutor(execution_mode="attach")`. The payoff:

- **One API across all modes.** A training script no longer depends on who
  launches the process (in-process, external-process, or attach).
- **Modern transport for free.** Attach rides the #4906 Cell/FOBS path, so
  self-started trainers get the same lazy large-payload streaming, filters, and
  deduplication as the other modes rather than a parallel legacy stack.
- **A deprecation path.** Once attach and external-process coverage is green,
  `FlareAgent`/`IPCAgent`/`IPCExchanger`/Pipe can be removed in a later change.

The trade is explicit: V1 attach uses the same rendezvous-by-id and site-trust-boundary
assumption as legacy IPCAgent. Its secure network path is deliberately narrower:
network attach requires mTLS, while bare-CA one-way TLS is rejected. It is a
UX/consistency/maintenance win with an additive path to stronger authentication later.

## Decision Summary

The CJ-side implementation is named `AttachBackend`. The trainer continues to use
`CellClientAPI`; there is no new public `FlareAgent`, `IPCAgent`, or
`IPCExchanger` replacement class.

```text
ClientAPIExecutor(execution_mode="attach")
  -> AttachBackend                         (CJ side)
  -> Client API protocol over Cell
  -> CellClientAPI                         (externally owned trainer)
  -> flare.init()/receive()/send()/log()
```

**Attach adopts legacy IPCAgent's rendezvous and runs the Client API protocol
over it.** As with `IPCAgent`/`IPCExchanger`, the external trainer and the job
are independently provisioned with the same `attach_id`, and either may start
first. Concretely, matching how IPCAgent already works
(`nvflare/client/ipc/ipc_agent.py`, `nvflare/app_common/executors/ipc_exchanger.py`):

- The trainer names its own Cell from `(site_name, attach_id)` — a
  **job-independent** name it knows in advance — and connects to the site's Cell,
  exactly as `IPCAgent` connects with `parent_url=flare_site_url`. Its FQCN is
  fixed at Cell construction; nothing is renamed or "adopted" later.
- The CJ (`AttachBackend`) derives the **same** trainer FQCN from `attach_id`
  (which it holds from its executor config or job meta) and **reaches out** to it,
  exactly as `IPCExchanger` addresses the agent by
  `agent_site_fqcn(client_name, agent_id)` and waits for it. The trainer learns
  the CJ's FQCN from the origin of the CJ's first request.

Because `attach_id` is only a rendezvous name — not a credential, exactly as the
legacy `agent_id` was — it may live in the server-distributed job configuration.

Two consequences of adopting the IPCAgent topology keep V1 small:

1. **The trainer is a routable site-level cell, not a CJ child.** Its name is
   `<site>.-client_api_<attach_id>` (a child of the site, sibling of the CJ), the
   same shape `IPCAgent` uses (`agent_site_fqcn` = `<site>.-<agent_id>`). Because
   the trainer connects to the site's Cell, the site holds its routing agent, so
   the CJ can route requests to it — the direction `IPCExchanger`→`IPCAgent` proves
   in production. (Here the site cell `<site>` is the persistent client parent,
   "CP", that both the CJ and the trainer connect to.) The result data path is
   unchanged from #4906 and is conditional (see Cell Topology): the result
   *envelope* always flows trainer → CP → CJ → CP → server, while the payload
   *bytes* depend on filters — with a CJ filter the CJ materializes and re-serves,
   with none a downstream receiver pulls the lazy bytes directly (server → CP →
   trainer). The one new thing is that this no-filter direct pull is served from a
   CJ *sibling* rather than a *descendant* — a hard-gated integration test, not an
   already-proven path. Either way, no CJ-assigned identity, no FQCN adoption
   after connect, and no change to the shared Cell overlay or connection manager
   are required.

2. **V1 security uses the provisioned IPCAgent path over existing F3 (no F3
   change).** The
   secure-network path is a **provisioned-trainer mutual TLS** on the site's existing
   backbone (IPCAgent's credential path: `{CA_CERT}` + auto-discovered client cert),
   where Attach verifies that the CA, client certificate, and client key are all
   present and readable before Cell construction. F3's existing mTLS registration
   binds the cert to the trainer FQCN before it is routable; `attach_id` at
   `SESSION_OPEN` then selects which trainer. A **bare-CA** (CA-only) trainer is
   rejected in V1 because a one-way-TLS listener does not authenticate the peer
   before registration. There is no application-layer secret, cryptographic proof,
   or broker in V1; `attach_id` match plus transport is the bar. A hardened mode that
   adds a site-local shared secret and mutual proof is deliberately deferred (see
   Non-Goals); because rendezvous and the task/result protocol are independent of it,
   that mode is purely additive.

The legacy classes remain compatibility references only:

- `IPCExchanger` is a separate Executor and uses the legacy `flare_agent`
  protocol.
- `IPCAgent` exposes an agent-specific API rather than the standard Client API.
- `FlareAgent` and its Pipe stack have different retry and payload-lifetime
  semantics.

The attach implementation must not depend on those classes; it reuses their
*rendezvous shape*, not their code.

## Goals

- Let an externally started trainer use the normal Client API unchanged, so one
  training script runs in in-process, external-process, and attach modes.
- Permit either the external trainer or the NVFlare job to start first, using
  IPCAgent-style rendezvous by a shared `attach_id`.
- Address multiple independently waiting trainers by pre-provisioned attach IDs,
  with each job selecting the intended trainer.
- Reuse the #4906 Cell/FOBS data path and payload-lifetime contract unchanged,
  including its conditional pass-through: with a CJ filter the CJ materializes and
  re-serves the result; with none, the routable trainer serves the downstream pull
  directly.
- Keep Attach independent of the concrete F3 driver so a replacement `file://`
  driver can provide FilePipe-equivalent local transport without changing the
  attach protocol or backend.
- Match legacy `IPCAgent`'s trust model: `attach_id` is a rendezvous name, and
  transport security plus the site trust boundary provide protection.
- Bind a job to one trainer session and reject duplicate trainers, stale sessions,
  and messages from unexpected Cell origins.
- Add sender retry and receiver deduplication together.
- Keep large lazy result sources alive through uncertain control replies and
  receiver-confirmed terminal settlement.
- Allow an explicitly configured replacement trainer to attach after a lost
  session without silently replaying an ambiguous active task.
- Preserve external ownership: teardown closes the NVFlare session but never
  terminates the trainer process.

## Non-Goals

- Resuming arbitrary user training state after an external process crashes.
- Automatically delivering credentials to Kubernetes, Slurm, or an AV platform.
- Adding a public backend-extension registry.
- Adding a new payload manifest, transfer ID layer, or streaming protocol above
  FOBS/`DownloadService`.
- Removing `IPCAgent`, `IPCExchanger`, `FlareAgent`, CellPipe, or FilePipe in this
  change.
- Implementing the replacement `file://` F3 driver. It may land in a separate PR,
  but this design defines the contract Attach requires from it.
- Making LOG delivery reliable. LOG remains best-effort analytics traffic.
- Supporting more than one active trainer or active task per `ClientAPIExecutor`.
- **A stronger-than-IPCAgent authentication or confinement story.** V1 does not
  add an application-layer shared secret, HMAC/challenge-response proof, a
  secret-holding broker, or trainer confinement beyond what IPCAgent provides. A
  hardened, confined mode (for example a CJ-rooted trainer identity behind a
  dedicated CJ listener, and/or a site-local mutual-proof secret) is future work;
  it layers on top of the same rendezvous and task/result protocol without
  changing them.

## Architecture and Code Factoring

`ExternalProcessBackend` currently combines two kinds of behavior:

1. Cell session and task/result protocol behavior.
2. Owned-process behavior such as bootstrap-before-launch, `Popen`, process
   groups, output collection, and staged process termination.

Attach needs the first group but must never inherit the second. The implementation
should therefore extract a private common base rather than make `AttachBackend` a
subclass of `ExternalProcessBackend`.

```text
CellBackendBase                         (internal)
├── ExternalProcessBackend
│   ├── local/internal listener + CJ-launched child
│   ├── Popen and process-group liveness
│   ├── launch_once policy
│   └── SHUTDOWN -> natural exit -> TERM -> KILL
└── AttachBackend
    ├── derive trainer FQCN from (site, attach_id) and reach out (IPCExchanger-style)
    ├── attach_id rendezvous; no launch, no process ownership
    ├── heartbeat-only trainer liveness
    ├── optional reconnect to the same attach_id
    └── SHUTDOWN -> revoke session (no process action)
```

`CellBackendBase` should own:

- CJ Cell lookup and protocol callback registration;
- the lazy-decode pass-through for `GET_TASK` server-command replies (the #4906
  route the CJ decodes lazily), and its cleanup — this is not a trainer task-pull;
  tasks are pushed to the trainer via `TASK_READY`;
- one-task execution admission;
- task correlation and result waiting;
- `TASK_READY`, RESULT, LOG, and HEARTBEAT common handling;
- post-handshake origin/session validation;
- analytics forwarding;
- task-exchange configuration; and
- closed/finalized callback gates.

The base provides hooks for:

- creating or waiting for the mode-specific session;
- establishing the mode-specific peer binding;
- checking mode-specific liveness;
- handling session loss;
- requesting shutdown; and
- releasing mode-specific resources.

Session state shared by both modes should be separate from process state.

Driver selection remains below this backend layer. `AttachBackend` passes the
exported URL and connection properties to Cell/F3 and must not instantiate,
import, or branch on TCP, gRPC, FilePipe, or a future file-driver class. Cell/F3
selects a registered driver from the URL scheme. The driver owns URL parsing,
transport resource creation, connection lifecycle, and transport-specific security
validation; Attach owns rendezvous, session binding, and the Client API protocol.

```python
class _CellSession:
    trainer_fqcn: str          # derived from (site, attach_id); fixed at construction
    session_id: Optional[str]
    ready: threading.Event
    result_source_live: threading.Event
    last_peer_activity: Optional[float]
    shutdown_requested: threading.Event


class _OwnedTrainerSession(_CellSession):
    token: str
    process: Optional[subprocess.Popen]
    pgid: Optional[int]
    bootstrap_path: Optional[str]
    reaper_thread: Optional[threading.Thread]


class _AttachedTrainerSession(_CellSession):
    attach_id: str
    disconnected_at: Optional[float]
```

The extraction should be behavior-preserving for `ExternalProcessBackend` and land
with its existing unit and integration tests still green before attach-only
behavior is added.

## Attach Provisioning and Rendezvous

### Pre-provisioned Identity

Attach does not use a CJ-generated runtime bootstrap. Before either side starts,
the client-site operator or platform provisions one `attach_id` and supplies it to
both:

1. the external trainer's connection profile; and
2. the `ClientAPIExecutor(execution_mode="attach")` job configuration (or the job
   meta, exactly as `IPCExchanger` may read `agent_id` from job meta).

`attach_id` is not a secret, so putting it in the submitted job is acceptable — it
is the rendezvous name that pairs one trainer with one job, exactly as legacy
`agent_id` did. This is the same ordering-independent rendezvous pattern as
`IPCAgent`/`IPCExchanger`: the trainer may connect to the site and wait before the
job exists, and if the job starts first, `AttachBackend` reaches out and retries
until the trainer is reachable or `attach_timeout` elapses. No post-job-start file
copy, mount watcher, Kubernetes callback, or CJ-generated credential is required.

Different external trainers use different attach IDs. A job selects exactly one ID,
so multiple trainers can wait concurrently without ambiguity:

```text
trainer A <-> attach_id A <-> job A
trainer B <-> attach_id B <-> job B
```

Each rank-0 `CellClientAPI` context owns one attach endpoint. An external host that
serves several jobs runs separate trainer processes/contexts with separate profiles
and attach IDs; the process-global `flare` API does not multiplex several
simultaneous jobs through one context.

Two concurrently active jobs must not select the same attach ID. Sequential job
restart/recovery may reuse it when that behavior is intentional, but a new
trainer/job association should receive a newly provisioned ID.

The trainer selects its connection profile with either:

```python
flare.init(config_file="/path/client_api_attach.json")
```

or the existing `NVFLARE_CLIENT_API_BOOTSTRAP` environment variable. The file is a
pre-provisioned connection profile, not an artifact written by the CJ. Its
attach-specific fields are:

```text
execution_mode = "attach"
attach_id
site_name
connect_url (driver-selecting URL for the site's Cell — the same connection the
             legacy IPCAgent dials, e.g. flare_site_url)
driver-specific connection properties
secure_mode / connection_security (network drivers only)
CA or trust-material reference when the selected driver uses TLS
protocol_version
optional job_wait_timeout
```

`site_name` is the authoritative site identity for the attach rendezvous and is used to
derive the trainer FQCN. A generic URL host or filesystem path is not treated as an
NVFlare site name. If a future driver exposes an authoritative peer/site identity,
the profile must validate it against `site_name`; Attach does not guess that identity
from `connect_url`.

Job-specific values such as job ID, CJ FQCN, session ID, heartbeat policy,
task-exchange settings, `memory_gc_rounds`, and `cuda_empty_cache` arrive in the
CJ's session-open request (the trainer learns them at rendezvous, not from the
pre-job profile).

### Cell Topology

The trainer forms its Cell identity from data it already has, before the job
exists:

```text
<site>.-client_api_<attach_id>
```

This is the same shape legacy `IPCAgent` uses (`agent_site_fqcn` =
`<site>.-<agent_id>`, `nvflare/client/ipc/defs.py`): a child of the site, sibling
of the CJ. The `-` leaf prefix follows the established ad-hoc child convention. The
FQCN is derivable from the pre-provisioned `site_name` and `attach_id` with no
knowledge of the job, so it is known at Cell construction — the trainer never has
to learn or adopt a new identity after connecting.

The trainer connects to the site's Cell, exactly as `IPCAgent` connects with
`parent_url=flare_site_url` and `create_internal_listener=False`. It is therefore a
routable participant in the overlay: the site holds its routing agent. That gives
two properties for free, from the existing fabric:

- **The CJ can reach it.** `AttachBackend` derives the identical
  `<site>.-client_api_<attach_id>` from its configured `attach_id` and site, and
  routes control requests to it up through the shared site parent — the same way
  `IPCExchanger` reaches `agent_site_fqcn(client_name, agent_id)`.
- **A downstream receiver can pull from it (no-filter pass-through case only).**
  When the result is passed through — no CJ filter needs the content, see the
  Filters note below — a server/workflow receiver resolves the lazy `ViaDownloader`
  reference to the trainer's FQCN by longest-prefix overlay routing (server → CP →
  trainer, one hop shorter than external-process's server → CP → CJ → trainer). The
  trainer is a routable cell, so no route injection is needed. State the provenance
  precisely: the routing is proven by IPCAgent, and server-originated lazy pulls
  *from the trainer source* are proven by #4906 external-process — but from a CJ
  *descendant*. A #4906 lazy source served from a CP-child (CJ sibling) is
  first-of-its-kind and must be validated by the gating integration test, not
  assumed because "IPCAgent data flows this way" (it did not — IPCAgent used the
  Pipe full-payload transport over the CJ↔agent link, and the server never pulled
  from the agent).

The trainer is **not confined** to talking only to the CJ; like the IPCAgent
agent, it is a routable site peer. That is the accepted V1 posture (IPCAgent
parity). A confined variant (a CJ-rooted identity behind a dedicated CJ listener)
is stronger but requires the job id at trainer startup and a per-listener admission
change; it is deliberately deferred (see Non-Goals).

**Filters work as in external-process: pass-through is conditional, and a wired CJ
filter causes the CJ to materialize and filter.** Direct trainer → receiver
pass-through (lazy references the ultimate receiver pulls itself) is a #4906
*optimization*, used only when no CJ filter needs the content. When a task-result
filter is installed, the result is **not** passed through: the CJ materializes the
whole payload, the filter transforms it, and the CJ sends the *filtered* result to
the server. Attach preserves this unchanged. The CJ materializes by obtaining the
payload from the trainer's `<site>.-client_api_<attach_id>`, routed CJ → CP →
trainer — the same path the CJ already uses to reach the trainer for `SESSION_OPEN`,
`TASK_READY`, and heartbeat — then filters and re-serves. This is identical to
external-process except that the CJ's pull traverses the CP relay (a different
process) rather than a direct CJ↔child link, which works the same. The direct
server → trainer pull (server → CP → trainer, bypassing the CJ) therefore occurs
**only** in the no-filter pass-through case, where there is nothing for a filter to
miss. The one attach-specific difference is that this no-filter pass-through pull is
served from a CJ *sibling* rather than a CJ *descendant* — the first-of-its-kind
path covered by the gating integration test.

The complete trainer FQCN is non-secret and, like `attach_id`, may appear in
routing headers, diagnostics, and lazy references forwarded to the server. Learning
it authorizes nothing; it is a routing name.

### Transport and Security

Attach runs the same Cell and Client API protocol above a swappable F3 driver, over
the **same connection legacy IPCAgent already uses** to reach the site. The URL
scheme and exported site connection properties select the driver; the attach
implementation must not assume that every connection is a TCP/TLS socket.

V1 adopts legacy IPCAgent's rendezvous trust model: protection comes from the
transport plus the operator keeping the trainer's connection to the site inside
the site's trust boundary. There is no application-layer secret. Secure network
attach additionally requires mTLS so F3 authenticates and FQCN-binds the trainer
before it is routable. Cleartext remains limited to loopback unless the operator
explicitly enables a trusted development network. `attach_id`, like `agent_id`, is
distributable through the job and is not a credential. A hardened mode that adds a
shared secret and mutual proof is future work for deployments that cannot make the
trust-boundary assumption.

V1 needs **no new F3 capability API** to make its transport decisions. F3 does not
expose per-driver security properties today (`driver.py` capabilities do not report
configured boundary/authentication), and adding that would contradict the
zero-F3-change scope. Instead the backend derives policy from what it already has:
the parsed `connect_url` (network vs loopback vs `file://`) and the profile's
`connection_security` (`mtls` / clear; `tls` is rejected). A generalized "normalized security
description" from Cell/F3 is deferred to the work that introduces the custom/`file://`
drivers. For the V1 driver (TLS network + loopback) the policy is:

- For a TLS network driver, V1 uses **IPCAgent's provisioned credential handling** — same
  code path, no attach-specific transport story. The trainer runs from a provisioned
  workspace containing `rootCA.pem`, constructs its Cell with
  `credentials = {CA_CERT: rootCA.pem}`, and the Cell ctor's `enhance_credential_info`
  (`nvflare/fuel/f3/drivers/net_utils.py`) auto-discovers `client.crt`/`client.key`
  from that same folder. Attach then fails before Cell construction unless the CA,
  client certificate, and client key paths all name readable regular files; the
  `mtls` policy label alone is not sufficient. The trainer presents the project-CA
  client cert over the site's normal mTLS backbone; no dedicated listener is needed.
  F3 binds the certificate CN to the claimed FQCN at registration
  (`ConnManager.update_endpoint`, gated on the mTLS connection), so the peer is
  authenticated as a project member and FQCN-bound before it becomes routable.
  `SESSION_OPEN`'s `attach_id` then selects which trainer. A profile specifying
  `connection_security=tls` is rejected: supporting a cert-less client would require
  a separately gated listener/admission design and is deferred.
- For the future `file://` driver, there is no TLS handshake and no CA.
  Confidentiality and integrity depend on the configured directory's ownership and
  ACLs. That driver, not Attach, validates that the path is absolute, provisioned
  outside the submitted job, available to exactly the intended principals, and not
  world-accessible or world-writable. Symlinks and unsafe ownership/permission
  changes fail closed. It then reports a protected-local-boundary result to
  Cell/F3. Rendezvous for `file://` is wait-until-present.
- A cleartext non-loopback network driver is rejected by default. The attach-only
  `allow_insecure_attach` argument (default `False`) must be set `True` as an
  explicit development/POC opt-in to permit such a route; enabling it emits a
  prominent warning. This flag does not apply to `file://`.
- Custom and `file://` drivers are out of V1 scope; the generalized security-property
  contract (and any F3 capability API for it) is defined together with those drivers,
  not in V1. V1 supports the TLS network driver and loopback only.

Loopback classification uses the parsed URL host and `ipaddress.is_loopback`; it
does not trust arbitrary DNS names that temporarily resolve to loopback. Wildcard
addresses such as `0.0.0.0` and `::` are not loopback.

Because attach reuses the existing site connection path and routing (the same one
IPCAgent dials), V1 needs **no change to the shared connection manager, overlay
admission, or routing** — the transport-core "zero change" thesis holds — and it
needs **no per-job and no dedicated attach listener**: a provisioned trainer rides
the site's **existing (mTLS) backbone** exactly as a provisioned IPCAgent does.
Bare-CA one-way TLS is not a supported Attach deployment in V1.

## Session Establishment

The CJ initiates the session because it knows the trainer's derived FQCN, while the
trainer does not know the CJ's FQCN in advance (it contains the job id). This
borrows IPCAgent's rendezvous *direction* (the CJ reaches out to a passively
waiting, id-named peer), but it is **not** a reuse of any existing handshake:
`SESSION_OPEN`/`SESSION_ACCEPTED` are new topics, and this **inverts** #4906's
existing session setup, where `HELLO` is *trainer*-initiated to a CJ FQCN the
launch bootstrap already provided. (`IPCExchanger` itself has no session handshake
at all — it drives the agent by requests and heartbeats.) So `CellClientAPI.init()`
must be refactored for attach: instead of actively sending `HELLO` to a known
`_cj_fqcn`, it registers a `SESSION_OPEN` request callback and waits passively; the
`session_id` and heartbeat policy it currently reads from `HELLO_ACCEPTED` now
arrive *in* `SESSION_OPEN`; and its control-message validation binds the CJ origin
*learned* from that request rather than a pre-known FQCN. This is a real (bounded)
`init()` change, listed under Implementation Changes, not free rendezvous reuse.

```text
CJ -> trainer : SESSION_OPEN(session_id, job_id, site_name, protocol_version,
                             heartbeat_policy, task_exchange, runtime_settings)
trainer -> CJ : SESSION_ACCEPTED(session_id, trainer_protocol_version)
                or SESSION_REJECTED(session_id, reason)
```

`AttachBackend` derives `<site>.-client_api_<attach_id>` and sends `SESSION_OPEN`
to it, retrying (bounded by `attach_timeout`, abort, and finalization) until the
trainer is reachable. While the trainer cell has not yet registered in the site
overlay, `SESSION_OPEN` fails with a Cell no-route/delivery error; the CJ retries on
that error class and treats the first `SESSION_ACCEPTED` as proof the trainer has
appeared. The CJ generates a fresh CSPRNG `session_id` for the logical open and
reuses it on retries. The trainer's handler records the CJ's FQCN from the request
origin as its bound peer, applies the job-specific settings, and returns
`SESSION_ACCEPTED`. Runtime validation, including framework decomposer registration,
must complete before any peer/session binding state is committed. A failure returns
`SESSION_REJECTED` while leaving the trainer unbound and able to accept a later valid
open; binding and `_opened` publication are one atomic success transition. Every
subsequent message in either direction must match both the bound peer FQCN and the
current `session_id`; attach protocol fields are removed before a result is forwarded
beyond the CJ.

After `SESSION_ACCEPTED`, the session runs the unchanged #4906 loop: the CJ
**pushes** `TASK_READY`, the trainer's `receive()` returns it from its handler queue
(the Client API is push-fed — `receive()` does not poll the CJ), and `send()`
publishes `RESULT_READY`. The CJ-initiated handshake only establishes the binding;
task/result flow is identical to external-process.

Establishment is retry tolerant:

- repeating an identical `SESSION_OPEN` returns the same acceptance;
- a second, different CJ origin for an already-bound trainer is rejected;
- a mismatched `attach_id`/site, wrong rank, or incompatible protocol version is
  rejected without disturbing a bound or waiting state;
- unreachable delivery is retried by the CJ only until `attach_timeout`, abort, or
  finalization; and
- stale/duplicate messages that do not match the current binding are dropped.

In attach mode, `flare.init()` constructs the trainer Cell from
`(site_name, attach_id)`, connects to the site, and waits for a valid
`SESSION_OPEN`. `job_wait_timeout=None` intentionally permits an externally started
trainer to wait for a future job; deployments may set a finite timeout. Interruption
and explicit shutdown always stop the wait. Semantic rejection and incompatible
protocol/runtime settings fail immediately and clean up the partial session without
falling back to another Client API engine.

`CellClientAPI.init(rank=...)` uses the explicitly supplied rank when present,
otherwise the same global `RANK` environment variable used by external-process
mode, defaulting to `"0"` only for a genuinely single-process trainer. It never uses
`LOCAL_RANK` for control ownership. Nonzero ranks return before constructing a Cell.
A platform sharing one profile across several processes must assign global ranks
correctly; otherwise multiple processes will try to bind the same trainer FQCN and
all but one will fail.

## Protocol Additions

The existing channel and task/result message shapes remain. Add these topics:

```text
SESSION_OPEN / SESSION_ACCEPTED / SESSION_REJECTED
TASK_STATUS
RESULT_STATUS
```

`HELLO`/`HELLO_ACCEPTED`/`HELLO_REJECTED` remain the external-process
(trainer-initiated) session handshake and are unchanged. `SESSION_OPEN` and its
replies are the attach-only, CJ-initiated handshake; attach does not use `HELLO`.
Both plug into the base backend's mode-specific "create/wait-for-session" hook and
coexist without conflict.

Add stable message keys for:

```text
ATTACH_ID
RESULT_ID
ATTEMPT_ID
ACCEPTED_ATTEMPT_ID
TASK_STATE
RESULT_STATE
```

Task and result `Shareable` values remain directly inside `TASK_READY` and
`RESULT_READY`. The status messages are small control-only requests and carry no
model payload.

## Paired Retry and Deduplication

PR #4906 deliberately did not add receiver-only deduplication while there was no
sender retry. Attach adds both sides together.

### Task Delivery

Each call to `AttachBackend.execute()` creates one stable logical `task_id`. Each
transmission uses a fresh `attempt_id`.

The initial request is:

```text
TASK_READY(session_id, task_id, attempt_id, task_name, Shareable)
```

If the outcome is uncertain because the reply was lost or the request reports a
retriable transport failure, the CJ first sends:

```text
TASK_STATUS(session_id, task_id)
```

Possible states are `UNKNOWN`, `QUEUED`, `DELIVERED`, `RESULT_PUBLISHING`, and
`COMPLETE`.

- If the trainer reports a known state, the CJ treats the original delivery as
  accepted and does not resend the model.
- If it reports `UNKNOWN`, the CJ may resend `TASK_READY` with the same task ID and
  a new attempt ID.
- Semantic rejection, abort, attach timeout, or session loss is not retried.
- Retries remain bounded by `task_wait_timeout`, abort, and session liveness.

`CellClientAPI` keeps a task ledger. A duplicate task ID returns the current state
and is never queued to user code twice. A different task ID is rejected while
another task is nonterminal. The same rule covers reordering: if an earlier delayed
attempt arrives after a later attempt was accepted, its task ID is recognized and it
receives the already-current state; it cannot replace the accepted model or enqueue
a second copy.

FOBS materializes a request before the task handler runs. Consequently, blindly
resending a large task could repeat its download even though handler-level
deduplication prevents double execution. `TASK_STATUS` is required specifically to
avoid that cost in the normal lost-ACK case.

### Result Publication

Every `flare.send()` creates one stable logical `result_id`. Each transmission uses
a fresh `attempt_id`:

```text
RESULT_READY(session_id, task_id, result_id, attempt_id, Shareable)
```

The CJ atomically accepts the first valid attempt for `(session_id, task_id,
result_id)` and records that attempt as canonical. Its reply includes:

```text
RESULT_ACCEPTED(result_id, accepted_attempt_id)
```

Duplicates behave as follows:

- the same result ID returns `RESULT_ACCEPTED` with the already chosen canonical
  attempt ID;
- a different result ID after a result was accepted for the task is rejected; and
- no duplicate creates a second task result or replaces the canonical `Shareable`.

After an uncertain reply, the trainer first sends:

```text
RESULT_STATUS(session_id, task_id, result_id)
```

The response is `UNKNOWN`, `ACCEPTED(accepted_attempt_id)`, or `REJECTED(reason)`.

This result protocol must preserve lazy-transfer ownership across uncertain control
replies:

1. The trainer records the `DownloadService` transfer waiters created by every
   result attempt.
2. A request timeout does not immediately delete that attempt's transactions.
3. Once the CJ identifies the canonical attempt, the trainer keeps that attempt's
   transactions alive and cancels noncanonical attempts.
4. The trainer waits for strict receiver-confirmed terminal success of the canonical
   attempt before `flare.send()` releases its source resources.
5. Abort, semantic rejection, session failure, or retry exhaustion cancels all
   still-owned attempts.

This handles the critical race where attempt 1 was accepted by the CJ, its
`RESULT_ACCEPTED` reply was lost, and attempt 1 contains lazy references already
being forwarded to the server. Deleting attempt 1 before discovering the CJ's
canonical choice would corrupt the accepted result.

After a lost or uncertain result reply, the trainer probes `RESULT_STATUS` before
honoring a concurrent orderly SHUTDOWN. Routine SHUTDOWN prevents new work but does
not revoke an already-admitted send: the trainer keeps its Cell and attempt transfers
alive until it resolves the CJ's canonical choice and the canonical publication
barrier settles. ABORT and terminal transport/session failure remain allowed to end
that recovery.

`RESULT_ACCEPTED` canonicalizes the result envelope; it is not durable delivery of
unresolved lazy references. Terminal delivery requires the externally owned trainer
to remain alive and attached until receiver-confirmed transfer success. An orderly
main-thread exit or `flare.shutdown()` defers Cell cleanup while the canonical source
is being served. If the process crashes, is killed, or exits without allowing that
barrier to settle, the transfer and task fail even though the CJ had recorded the
attempt as canonical. NVFlare cannot prevent an external owner from forcibly
terminating the source.

The CJ keeps bounded scalar deduplication tombstones containing IDs, state, canonical
attempt, and rejection reason. It does not retain completed model payloads merely for
deduplication.

The per-task canonical-acceptance record is authoritative and is not a normal
capacity-evictable tombstone. It remains pinned while the task is active, while the
CJ can still return or forward its result, while any canonical result source is live,
and throughout the result-wait plus heartbeat/retry uncertainty window. Capacity
pressure may evict only fully terminal records that are safely past that window. TTL
cleanup also applies only after those conditions are false. Consequently, eviction
can never make a live task appear result-free or allow a different `result_id` to
become canonical. A request for an evicted historical task is rejected as stale; it
is never treated as new work.

LOG remains fire-and-forget. HEARTBEAT requests are naturally idempotent and do not
need an attempt ledger.

## Attach Lifecycle

The backend state machine is:

```text
CREATED
  -> WAITING_ATTACH
  -> ACTIVE_IDLE
  -> ACTIVE_TASK
  -> ACTIVE_IDLE
  -> DISCONNECTED
       -> WAITING_ATTACH(new session, same attach ID)  when allow_reconnect=True
       -> CLOSED                                      otherwise
```

`initialize()` sets up callbacks, derives the trainer FQCN, and starts bounded
`SESSION_OPEN` attempts without blocking START_RUN indefinitely. `execute()` waits
for the session. The wait is interruptible by the task abort signal, backend
finalization, `attach_timeout`, and successful session binding.

`attach_timeout` is measured from backend initialization for the current job, not
from trainer startup or profile creation. `None` means the CJ has no attach deadline,
but execution remains abort/finalize interruptible. Timeout stops the CJ's open
attempts and wakes task waiters; it does not invalidate or rewrite the
pre-provisioned attach profile.

Once `SESSION_OPEN` has been accepted, the attach deadline no longer participates in
liveness and cannot revoke a long-running task. An active session is governed by its
session ID, bound peer FQCN, heartbeat/transfer precedence, abort, and job lifetime.

### Heartbeat and Session Loss

For attach, session-bound message activity and the heartbeat lease are the only
trainer liveness signals. There is no PID or process-group probe.

The existing transfer precedence rules remain:

- a progressing result transfer prevents control-heartbeat timeout from revoking the
  source session;
- task materialization is governed by the Cell/FOBS progress-aware request wait; and
- after materialization, ordinary heartbeat timeout remains authoritative during user
  training.

Attach adds an absolute task-materialization ceiling because the trainer is an
externally owned, lower-trust peer. `task_wait_timeout` is measured from the first
TASK_READY attempt to terminal TASK_ACCEPTED/TASK_FAILED and is not reset by bytes,
FOBS progress, status requests, or retransmission attempts. Progress continues to
suppress the lower idle/no-progress timeout, but it cannot extend this absolute
deadline. On expiry the CJ cancels the request and its owned task transfers, sends
ABORT when possible, and fails the task. For attach mode, an omitted/`None`
`task_wait_timeout` is normalized to a finite 600-second default; V1 does not permit
an unbounded attach materialization wait.

If the session is lost before a result is accepted, the active task fails. The
backend does not silently redeliver it to a replacement trainer because the old
trainer may have executed arbitrary user code.

If `allow_reconnect=False`, session loss closes the backend for the rest of the run.
If `allow_reconnect=True`, the backend returns to reaching out on the same
`attach_id` and opens a fresh session ID after the active task has failed or settled.
The original trainer may reconnect, or an externally managed replacement may connect
under the same attach_id after the old session is gone. A replacement may receive
subsequent tasks, but active-task continuation is not part of V1.

Reconnect requires no CJ-to-platform callback, new bootstrap, remount, or attach-ID
rotation. Stale traffic from the prior connection is rejected by its old session ID.
Rotation to a different attach ID is a provisioning action for a future trainer/job
association, not automatic reconnect behavior.

### Trainer-Side Failure Behavior

`flare.init()` has an observable wait contract:

- with `job_wait_timeout=None`, waiting for a future job is intentionally unbounded
  and remains interruptible by shutdown or process signals;
- with a finite `job_wait_timeout`, expiry raises `TrainerSessionError` after stopping
  the partial Cell;
- malformed profiles, incompatible transport/protocol settings, wrong rank, and
  session rejection fail immediately;
- no failure path falls back to another Client API engine; and
- exception text contains a safe reason but never the complete profile.

Attach also adds an idempotent trainer-side lifecycle guard to `CellClientAPI`. It
registers `atexit` cleanup and a main-thread/session watcher. Heartbeat loss, ABORT,
or terminal session failure triggers the task/result abort signals, wakes blocked
Client API calls, stops the session Cell, and marks `is_running()` false. Orderly
SHUTDOWN stops admission and wakes `receive()`, but it does not abort a result send
that was already admitted. When the main thread returns without an explicit
`flare.shutdown()`, it requests the same cleanup. Orderly cleanup must defer Cell
shutdown while a canonical accepted result source is still being served; it waits for
that transfer's receiver-confirmed terminal outcome or failure first. Because the
process is externally owned, the guard does not call `os._exit()` or kill the process
and does not retire attach-shared process-global F3 services. User code executing
outside a Client API call cannot be forcibly interrupted; the external platform
remains responsible for detecting a nonresponsive trainer process after the NVFlare
session has terminated.

### Abort, Shutdown, and Finalization

ABORT and SHUTDOWN remain peer-bound CJ-to-trainer controls carrying the session ID.

On ABORT:

- the trainer unblocks Client API calls with `TrainerSessionError`;
- active task/result operations are cancelled; and
- the CJ revokes the session without touching the external process.

On orderly END_RUN:

- the backend stops accepting new tasks and sessions;
- if a result source is accepted and still live, the backend preserves the CJ Cell/F3
  environment until the trainer's publication barrier settles or fails;
- it sends SHUTDOWN and closes the session; and
- it never calls `kill`, `taskkill`, `terminate`, `waitpid`, or another
  process-management operation.

`CellClientAPI.shutdown()` also needs mode-specific runtime ownership. The dedicated
process launched by `ExternalProcessBackend` may retire process-global F3 services so
worker pools cannot keep it alive. An attached process is externally owned and may be
resident or later create a new Client API context; attach shutdown stops only the
session Cell and session-owned transfers. It does not irreversibly stop process-global
F3 services.

## Configuration Surface

The public configuration remains centered on `ClientAPIExecutor`, with the
pre-provisioned rendezvous ID and one attach-only insecure-route opt-in:

```python
ClientAPIExecutor(
    execution_mode="attach",
    attach_id="<pre-provisioned rendezvous id>",
    attach_timeout=300.0,
    allow_reconnect=False,
    allow_insecure_attach=False,
    heartbeat_interval=5.0,
    heartbeat_timeout=30.0,
    task_wait_timeout=600.0,
    result_wait_timeout=None,
    params_exchange_format=...,
    server_expected_format=...,
    params_transfer_type=...,
    train_task_name="train",
    evaluate_task_name="validate",
    submit_model_task_name="submit_model",
    memory_gc_rounds=0,
    cuda_empty_cache=False,
)
```

`attach_id` is a required, non-secret executor argument (or, as with `IPCExchanger`,
readable from job meta). It is never generated by the CJ and, being non-secret, may
live in the submitted job configuration. The backend validates its format and derives
`<site>.-client_api_<attach_id>` from it and the site name.

The trainer's connection target (`connect_url`) is the site's existing Cell endpoint —
the same connection legacy IPCAgent dials — provisioned in the trainer profile. There
is no new per-job listener to bind and no per-slot address to allocate; attach reuses
the site connection the client already runs. A provisioned trainer authenticates with
an auto-discovered project-CA client cert over the existing mTLS backbone (the same
`{CA_CERT}` + `enhance_credential_info` path IPCAgent uses). A bare-CA trainer and
`connection_security=tls` are rejected in V1.

That endpoint is an operator-provisioned deployment value, not the federation server
port and not a value inferred from a submitted job. A stock local POC client's
internal listener may use a dynamically selected port and does not automatically
export an attach profile; such a POC is not attach-ready until the site platform
publishes a reachable child endpoint to the trainer.

`attach_id`, `attach_timeout`, and `allow_reconnect` must be represented in
`ClientAPIBackendContext`. #4906 correctly omitted active attach behavior while no
attach backend existed.

`allow_insecure_attach` is a new attach-only argument and defaults to `False`. It is
rejected in other execution modes. It does not make cleartext attach secure; it only
records an explicit operator decision to permit a cleartext, non-loopback network
route. Local loopback and policy-compliant `file://` attach do not require the flag.

For attach only, `task_wait_timeout=None` is normalized to the finite 600-second
absolute materialization ceiling shown above. External-process mode retains its #4906
timeout semantics.

Attach must always have a result-liveness bound. `heartbeat_timeout=0` is valid only
when `result_wait_timeout` is finite; otherwise a dead externally owned trainer could
leave the CJ waiting forever.

The following remain invalid in attach mode because they express process ownership:

- `command`
- `launch_once`
- `launch_timeout`
- `shutdown_timeout`
- `stop_grace_period`

Control retry uses the existing task/result bounds and heartbeat/session state. V1
should not add Pipe-style `resend_interval` or `max_resends` to the public executor
surface unless integration evidence shows those policies must be operator-tunable.

## Implementation Changes

Expected production changes:

1. `nvflare/app_common/executors/client_api/cell_backend.py`
   - add the private common Cell backend/session machinery;
   - add task/result deduplication records and status handlers.
2. `nvflare/app_common/executors/client_api/external_process_backend.py`
   - delegate common protocol behavior to the base;
   - retain only owned-process behavior and launch-token authentication.
3. `nvflare/app_common/executors/client_api/attach_backend.py`
   - derive the trainer FQCN from `(site, attach_id)` and reach out with
     `SESSION_OPEN` (IPCExchanger-style), retrying until reachable or `attach_timeout`;
   - implement attach wait, reconnect on the same attach_id, absolute
     task-materialization timing, the transport-security policy derived from
     `connect_url` + `connection_security` (mTLS secure path, bare-CA TLS rejected;
     no new F3 API), and non-owning teardown.
4. `nvflare/app_common/executors/client_api/backend_spec.py`
   - add the public attach ID, restore attach settings, and add the insecure-route
     policy to `ClientAPIBackendContext`;
   - document the non-owning attach lifecycle.
5. `nvflare/app_common/executors/client_api_executor.py`
   - construct `AttachBackend`;
   - include the attach settings in the backend context.
6. `nvflare/client/cell/bootstrap.py`
   - accept and validate the pre-provisioned attach connection-profile schema;
   - pass the URL and driver connection properties through without a network-only
     schema or dependency on FilePipe;
   - keep external-process runtime-bootstrap behavior unchanged.
7. `nvflare/client/cell/defs.py`
   - add attach session-open, status, result, and attempt constants.
8. `nvflare/client/cell/api.py`
   - construct the trainer Cell from `(site, attach_id)` and connect to the site
     (`parent_url=connect_url`), mirroring `IPCAgent`;
   - accept the CJ-initiated `SESSION_OPEN`, bind the CJ origin/session, and apply
     runtime settings only after decomposer/runtime validation succeeds;
   - implement task deduplication and TASK_STATUS;
   - implement result attempts, RESULT_STATUS, and canonical transfer cleanup;
   - add bounded failure cleanup plus the attach lifecycle guard;
   - make process-global F3 shutdown conditional on runtime ownership.
9. Documentation and provisioning
   - document attach-ID generation and independent provisioning to the job and trainer;
   - document that secure network attach uses IPCAgent's credential path
     (`{CA_CERT}` + `enhance_credential_info` auto-discovery from the trainer's
     workspace): a provisioned trainer rides the existing mTLS backbone only after
     the CA, client certificate, and client key are validated, while bare-CA one-way
     TLS is rejected;
   - document the V1 (IPCAgent-equivalent) trust boundary the operator must maintain;
   - add attach examples using the standard Client API, without exposing legacy IPC
     classes.

There is intentionally no shared-F3 / connection-manager change: attach reuses the
existing site connection and routing that IPCAgent already uses, so the transport-core
blast radius is zero. (A future confined mode would add a dedicated CJ listener and
CJ-rooted identity; that is out of scope here.)

## Verification Plan

### Unit Tests

- Connection-profile schema accepts both supported Cell modes and rejects unknown
  modes. Driver-provided authoritative peer identity, when available, must agree
  with `site_name`; no identity is guessed from a generic URL.
- The backend derives transport policy from the parsed `connect_url` and the profile's
  `connection_security` (no new F3 capability API): a network route with `mtls` is the
  secure path; `tls` (bare-CA) is rejected; a cleartext non-loopback route is rejected
  unless explicitly opted in.
- Attach ID format validation accepts a canonical rendezvous ID, rejects
  malformed/empty IDs, and never generates a fallback in the CJ.
- The CJ derives the same trainer FQCN, `<site>.-client_api_<attach_id>`, that the
  independently configured trainer constructs (rendezvous agreement), matching the
  legacy `agent_site_fqcn` shape.
- A cleartext non-loopback network route is rejected unless `allow_insecure_attach=True`;
  loopback classification rejects wildcard and arbitrary DNS hosts. This option is not
  consulted for `file://`.
- Secure attach builds trainer credentials as `{CA_CERT: rootCA.pem}`, uses
  `enhance_credential_info` auto-discovery, and rejects the profile before Cell
  construction unless the CA, client certificate, and client key are all present and
  readable. A provisioned startup-kit workspace presents the project-CA client cert
  and completes **mTLS** on the normal backbone; a bare-CA profile is rejected.
- When the replacement file driver is available, a `file://` profile requires no CA or
  TLS fields, waits until the site listener artifact is present, accepts a protected
  shared directory, and is rejected by the driver for relative, symlinked,
  world-accessible, world-writable, or ownership/ACL-incompatible paths.
- A CJ `SESSION_OPEN` with a matching derived FQCN binds a session; a second, different
  CJ origin for an already-bound trainer is rejected; an identical `SESSION_OPEN` is
  idempotent.
- Wrong site/attach ID, wrong rank, or incompatible protocol version is rejected without
  disturbing the bound/waiting state.
- A framework decomposer/runtime setup failure rejects SESSION_OPEN without committing
  peer/session state; a later valid open can bind and release `flare.init()`.
- Attach mode makes no change to the shared connection-manager admission path for
  existing cells (no regression to non-attach connections).
- Nonzero global ranks do not construct a Cell; `LOCAL_RANK=0` does not elect a nonzero
  global rank.
- Messages from stale peers and session IDs cannot mutate state.
- Attach timeout and abort wake blocked execution.
- A trainer with `job_wait_timeout=None` can wait before the job starts; a finite wait
  expires cleanly and releases partial Cell resources.
- `allow_reconnect=False` closes after session loss.
- `allow_reconnect=True` reaches out again on the same attach ID, creates a fresh
  session ID, rejects stale traffic, and does not replay an active task.
- A duplicate TASK_READY is never queued twice.
- A delayed earlier TASK_READY attempt cannot replace or repeat a later accepted attempt.
- A lost TASK_ACCEPTED reply is recovered through TASK_STATUS without model redelivery.
- Continuous task-transfer progress cannot extend the absolute attach `task_wait_timeout`.
- Duplicate RESULT_READY messages select exactly one canonical attempt.
- A lost RESULT_ACCEPTED reply preserves the canonical attempt's lazy transfer.
- Noncanonical result transactions are cancelled only after the canonical attempt is
  known.
- RESULT_STATUS remains useful after `execute()` returns without retaining the old
  result payload.
- Capacity/TTL cleanup cannot evict the canonical result authority of a live or
  still-uncertain task.
- END_RUN sends SHUTDOWN and revokes the session but invokes no process-control
  operation.
- Attach-mode trainer shutdown does not retire process-global F3 services.
- Session rejection and finite wait expiry make `flare.init()` raise cleanly after
  releasing partial Cell resources; no failure path falls back.
- The attach lifecycle guard wakes blocked API calls and cleans up after main thread
  exit without terminating the externally owned process, while deferring orderly Cell
  shutdown for a live canonical result source.
- Existing external-process tests remain unchanged and green after extraction.

### Integration Tests

- A trainer started before its job connects to the site, waits, is reached by the CJ,
  receives, trains, sends, logs, and exits its Client API loop on SHUTDOWN.
- A job started before its trainer reaches out and attaches successfully within
  `attach_timeout` once the trainer appears.
- Multiple waiting trainers with different attach IDs bind only to their corresponding
  jobs.
- **(Gating, first-of-its-kind)** A large lazy result *with no CJ filter* (pass-through)
  completes through receiver-confirmed streaming — the server-side receiver pulls
  #4906 `ViaDownloader` bytes directly from the trainer, which is a CP-child (site
  sibling of the CJ, so server → CP → trainer, bypassing the CJ) — before the
  attached source is released. This exercises a source topology neither IPCAgent (no
  server-originated pull) nor #4906 external-process (CJ-descendant source) covered,
  so it must pass before attach ships, not be assumed from either precedent.
- A large lazy result *with a CJ task-result filter* is materialized at the CJ (the
  CJ pulls the payload from the trainer over the CP relay, CJ → CP → trainer),
  filtered, and re-served to the server — the filtered data reaches the server, and
  the server does not pull the unfiltered original from the trainer. Confirms the
  materialize-then-filter path works over the attach (CJ-sibling) topology.
- Killing the trainer after RESULT_ACCEPTED but before lazy-transfer completion fails
  the canonical transfer/task rather than reporting durable success.
- Injected loss of TASK_ACCEPTED does not execute the task twice.
- Injected loss of RESULT_ACCEPTED does not invalidate the result accepted by the CJ.
- Injected loss of RESULT_ACCEPTED followed by orderly SHUTDOWN still recovers the
  canonical attempt through RESULT_STATUS, preserves its lazy transfers through
  receiver confirmation, and only then stops the trainer Cell.
- Killing the trainer before RESULT_READY fails the task without hanging.
- Heartbeat loss does not revoke a progressing transfer but does revoke a stalled/idle
  session.
- A replacement trainer can attach between tasks when reconnect is enabled.
- `torchrun --nproc_per_node=2` establishes one session from global rank 0.
- Secure-mode attach with a provisioned trainer workspace completes **shared-CA mTLS
  over the existing site backbone** (auto-discovered project-CA client cert, exactly as
  IPCAgent does); a bare-CA trainer is rejected. Regular mTLS connections remain
  unchanged.
- When the replacement driver lands, `file://` attach runs the same Cell/session
  protocol without a CA or TLS and passes only when the shared path satisfies the
  driver's local filesystem policy. This acceptance test may live in that driver's PR
  if it lands after Attach.
- Non-TLS network attach fails by default and succeeds only with the explicit insecure
  opt-in in a marked development test.

Run targeted unit tests first, then the external-process affected suite to prove the
common-code extraction did not regress #4906, followed by the attach integration tests
and project style checks.

## Compatibility and Rollout

The first attach implementation is additive:

- `execution_mode="attach"` changes from a clear not-implemented error to the new
  backend.
- In-process and external-process configuration and wire behavior remain compatible.
- Legacy `IPCAgent`, `IPCExchanger`, `FlareAgent`, and Pipe classes remain importable
  during migration. Attach gives their self-started-trainer use case a standard Client
  API path: the same training script that runs in-process or launched now also runs
  attached, and the CJ swaps `IPCExchanger` for `ClientAPIExecutor(execution_mode="attach")`.
- No compatibility bridge translates between the old `flare_agent` protocol and the new
  `client_api` protocol.
- Attach may land before the replacement file driver; until that driver is registered, a
  `file://` profile fails as an unsupported scheme rather than falling back to FilePipe
  or another transport.

V1 attach matches legacy IPCAgent's trust model (transport plus the site trust boundary;
`attach_id` is a rendezvous name, not a credential). A hardened mode — a site-local
shared secret with mutual proof, and/or a confined CJ-rooted trainer identity — is
deliberately future work and is additive over this design's rendezvous and task/result
protocol.

After attach and external-process coverage is green across supported examples, the
legacy path can be deprecated in a separate change. FilePipe removal also requires the
replacement file driver and its end-to-end Attach coverage to be green. Neither removal
nor the replacement driver itself is part of this implementation.
