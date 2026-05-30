# Progress-Aware Streaming For Congested Transfers

## Goal

Make streamed task/result transfer resilient to slow networks, large payloads, and many concurrent receivers without requiring users to tune several timeout settings in lockstep.

The default behavior should be:

> If a streamed transfer is making monotonic progress, do not fail or resend solely because wall-clock time is long. If progress stops for the idle timeout, fail clearly.

## Current Problem

The parent client job (CJ) could hit `peer_read_timeout` while the subprocess is still downloading/materializing the task payload. The CJ then resends the task even though the previous task payload is still active. Under congestion this creates duplicate work, stale state, and eventually corrupt or premature receive behavior.

Existing streaming/download code already has partial progress or idle-timeout concepts:

- `DownloadService` transaction timeout is based on time since last download activity.
- `ViaDownloader` has `min_download_timeout`, described as an inactivity timeout.
- `ByteStreamer` has ACK progress timeout.
- `ByteReceiver` has read timeout while waiting for chunks.

The gap is cross-process visibility. The CJ does not know that the subprocess is actively materializing the task payload.

## Non-Goals

- Do not add a user-facing `large_model_transfer` mode.
- Do not require users to set `peer_read_timeout`, `heartbeat_timeout`, `*_min_download_timeout`, and `*_streaming_per_request_timeout` manually.
- Do not make heartbeat/liveness trust an unbounded activity flag.
- Do not replace the 2.8 safety guards now present in `main`; build on them.

## User-Facing Configuration

Normal users should set nothing.

If an advanced override is needed, use generic streaming terms:

```json
{
  "streaming_idle_timeout": 600,
  "streaming_max_peer_silence": 900
}
```

Definitions:

- `streaming_idle_timeout`: fail an active transfer only after this many seconds with no monotonic progress.
- `streaming_max_peer_silence`: upper bound for treating a peer as alive because of transfer progress. This prevents a stuck progress signal from hiding dead peers indefinitely.

If `streaming_idle_timeout` is raised above the default, the default `streaming_max_peer_silence` should be derived as `max(900, 1.5 * streaming_idle_timeout)` unless explicitly configured.

## Design

### 1. Keep The 2.8 Safety Guards

The 2.8 correctness fixes have already been ported to `main` and are the baseline for this design:

- `receive(timeout=...)` forwards timeout.
- timed-out `receive()` returns `None`.
- timed-out `receive()` does not arm `send()`.
- missing FOBS downloaded item raises immediately.
- explicit `None` downloaded item raises immediately.
- intentional `FLModel.params` entries with `None` remain allowed when they are not failed download artifacts.

These guards prevent silent corruption. They are necessary even after progress-aware streaming is added.

### 2. Track Monotonic Transfer Progress

Progress must be based on monotonic counters, not "something happened."

A transfer progress record should include:

```python
transfer_id: str
job_id: str
task_id: str
direction: str
receiver_id: str | None
sequence: int
bytes_done: int
items_done: int | None
last_progress_time: float
started_time: float
completed: bool
failed: bool
```

Valid progress means at least one of these counters advanced:

- bytes received/sent
- ACK offset advanced
- item count advanced, when item semantics are defined
- EOF/completion observed

Repeated activity with unchanged counters is not progress.

`items_done` is optional and only applies to FOBS object batches where an "item" is a decomposer item ID such as `T0`, `T1`, etc. that has fully materialized. Raw byte streams and file streams should set `items_done` to `null` or omit it and rely on `bytes_done`.

Each progress event should carry a per-transfer monotonic `sequence`. Consumers must ignore stale sequence numbers and must update counters using monotonic max semantics rather than blindly overwriting state. Out-of-order progress events must not move `bytes_done`, `items_done`, or `last_progress_time` backwards.

### 3. Use Per-Transfer Scope With Roll-Up

Primary scope:

```text
(job_id, task_id, transfer_id/ref_id, direction)
```

CJ-side decisions may roll up to task or channel level, but state must remain per-transfer so that one progressing transfer does not mask a stalled sibling transfer.

For source-side `result_upload`, scope must include the downstream receiver/requester identity when available:

```text
(job_id, task_id, transfer_id/ref_id, direction, receiver_id)
```

This covers swarm and peer-to-peer paths where more than one downstream client parent or server-side receiver may pull the same source ref. A progressing receiver must not mask a stalled receiver for the same ref.

Reverse-path decisions may roll up across refs at the transaction level, but state must remain per `(transfer_id, receiver_id)` so a progressing receiver does not mask a stalled receiver.

### 4. Add Progress Signalling

The component that can observe monotonically advancing transfer counters must report progress to the component making the wait decision.

For task payload download:

- materializing side: subprocess
- waiting side: CJ
- direction: `task_payload_download`

For result upload:

- source/serving side: subprocess `DownloadService`
- materializing side: downstream receiver path, such as the FL server, a swarm aggregation client parent, or another peer client-side receiver
- waiting side: subprocess waiting for result transfer completion
- direction: `result_upload`

The known 5GB x 16-client failure first appears on `task_payload_download`, because the CJ resends while the subprocess is still materializing the task payload. The same event schema and tracker must be direction-neutral so `result_upload` uses the same progress contract rather than a separate fixed-timeout model.

Preferred mechanism: callback or pipe event, not shared memory or filesystem state. When progress must cross the subprocess/CJ pipe, the public event concept is `STREAM_PROGRESS`; the internal `Pipe` topic uses the reserved sentinel `_STREAM_PROGRESS_` to match other internal pipe control topics such as `_HEARTBEAT_`, `_ABORT_`, and `_END_`. When the progress source and waiter are in the same process, as in the reverse result path, use the same schema through a local callback rather than forcing a pipe round trip.

Example event:

```json
{
  "topic": "_STREAM_PROGRESS_",
  "job_id": "job-id",
  "task_id": "task-id",
  "transfer_id": "ref-id-or-stream-id",
  "direction": "task_payload_download",
  "sequence": 12,
  "bytes_done": 123456789,
  "items_done": 430,
  "timestamp": 1790000000.0,
  "state": "active"
}
```

Canonical progress-source rules:

- FOBS object transfers that use `DownloadService` should use `DownloadService` as the canonical progress source at the ref/transaction level. It owns the ref ID, receiver identity, transaction status, and served byte/item counts.
- Raw byte-stream transfers that bypass FOBS object refs should use `ByteReceiver` for download/materialization progress and `ByteStreamer` for upload/send progress.

Only one component may emit byte counters for a given transfer. If lower-level `ByteReceiver` or `ByteStreamer` progress is later wired under a FOBS object transfer, `DownloadService` should aggregate or forward that lower-level progress with ref/transaction metadata rather than emitting a second competing `bytes_done` counter.

TODO for lower-level byte-stream progress integration: keep this single-authority rule when wiring `ByteReceiver` or `ByteStreamer` progress under existing FOBS object transfers.

Phase 1 implementation note: the task-payload path uses the module-level `download_object()` helper in `nvflare/fuel/f3/streaming/download_service.py` as the FOBS materialization progress emitter, with `ViaDownloader` attaching `job_id`, `task_id`, `transfer_id`, item counts, and pipe callbacks. `ByteReceiver` and `ByteStreamer` remain the lower-level canonical emitters for a later pass that covers raw byte-stream progress directly.

Reverse-path implementation note: the result-upload path should use `DownloadService` on the subprocess side as the source-side progress emitter because that is where downstream receiver pull requests are observed. It should update a subprocess-local tracker through the same event schema with `direction: "result_upload"` and `receiver_id` set to the requester FQCN or equivalent identity when known.

For `task_payload_download`, `receiver_id` is optional and identifies the subprocess materializing the payload when present; forward aggregation does not depend on it. For `result_upload`, missing `receiver_id` is treated as `receiver_id = null`, which is acceptable for a single downstream receiver. Multi-receiver reverse transfers require the source to populate `receiver_id` consistently for every expected `(ref_id, receiver_id)` pair.

Emit progress events:

- on transfer start
- every 30 seconds while active, if counters advanced
- on completion
- on failure

Do not emit per chunk.

Progress events may travel on the same pipe as other control messages in the first implementation. This is acceptable only because the emit interval is much smaller than the default idle timeout (`30s << 600s`). If tests show progress events can be queued behind the large message they are meant to protect, move `_STREAM_PROGRESS_` to a priority channel before enabling heartbeat integration.

CJ-side receivers should log one INFO-level progress line for accepted start, periodic active, completion, and failure events. The event rate limit already keeps this to roughly one line per transfer every 30 seconds while active.

### 5. Suppress Resend While Progress Continues

When CJ sends a task to the subprocess, the current failure path is based on fixed wall-clock `peer_read_timeout`.

New behavior:

```text
if peer has ACKed/read the task:
    success
elif task payload transfer has monotonic progress within streaming_idle_timeout:
    continue waiting; do not resend
elif no progress for streaming_idle_timeout:
    fail or retry according to existing resend policy
```

This prevents duplicate resend while the previous payload is still materializing.

### 6. Make Reverse Result Upload Progress-Aware

The reverse path should be symmetric in contract, but not forced into the same code location as the forward path.

Forward task payload:

```text
CJ -> subprocess agent
subprocess downloads task/model payload from server
```

Reverse result payload:

```text
subprocess agent -> CJ/client parent -> downstream receiver
downstream receiver downloads result tensors directly from subprocess agent
```

The downstream receiver may be the FL server, a swarm aggregation client parent, or another peer client-side receiver. In the reverse pass-through path, the subprocess owns the source objects and creates a local `DownloadService` transaction. CJ/client parent receives lightweight refs and ACKs the subprocess before the downstream receiver has necessarily finished pulling the result tensors. Therefore the reverse-path waiter is the subprocess, not CJ/client parent.

`FlareAgent` refers to the subprocess-side client API agent that submits task results through the configured task pipe.

Current reverse-path protection is completion-based:

- register `DOWNLOAD_COMPLETE_CB`
- send the result refs to CJ
- wait up to `download_complete_timeout` for the server download to finish

That is safer than exiting immediately, but it is still a fixed total-duration wait. A slow 5GB upload over a congested network should not fail merely because wall-clock time is long when the server is still pulling chunks.

Concrete fast-path rule:

- if no `DownloadService` transaction is created for the submitted result, do not enter reverse progress-aware waiting;
- if a `DownloadService` transaction is created, use reverse progress-aware waiting even before the first server pull arrives;
- if reverse progress tracking cannot be installed for a created transaction, fall back to the existing `download_complete_timeout` behavior.

Add source-side progress for `result_upload`:

```text
downstream receiver pulls chunks from subprocess DownloadService
    -> DownloadService records bytes/items served
    -> result_upload progress updates a local TransferProgressTracker
    -> FlareAgent waits while progress remains recent
```

The reverse wait policy should become:

```text
if no DownloadService transaction was created:
    proceed immediately
elif DOWNLOAD_COMPLETE_CB fires:
    success or terminal failure according to transaction status
elif any expected ref has terminal failure, timeout, deletion, or abort:
    fail or warn clearly according to existing result-submit behavior
elif all expected refs are terminal-success:
    wait briefly for DOWNLOAD_COMPLETE_CB, then finish according to the observed transaction state
elif every started non-terminal ref has advanced within streaming_idle_timeout
     and no unstarted ref has exceeded its idle no-start budget:
    continue waiting
elif no result_upload progress for streaming_idle_timeout:
    fail or warn clearly according to existing result-submit behavior
```

This is an idle-based policy, not a total-duration policy. A transfer may exceed `download_complete_timeout`-style historical values if bytes/items continue advancing. If no reverse progress callback is available, retain the current `download_complete_timeout` behavior for compatibility.

Multi-ref aggregation rule:

- transaction state is active while any expected `(ref_id, receiver_id)` pair is non-terminal;
- transaction success requires every expected `(ref_id, receiver_id)` pair to reach terminal success;
- any terminal failure, timeout, deletion, or abort for an expected `(ref_id, receiver_id)` pair makes the transaction failed for wait-policy purposes;
- a `(ref_id, receiver_id)` pair is "started" once it reports any progress event with `bytes_done > 0`, `items_done > 0`, or a terminal state;
- a progressing pair must not mask a started sibling ref or sibling receiver that has stopped advancing beyond `streaming_idle_timeout`;
- an unstarted pair uses `streaming_idle_timeout` as its no-start budget;
- an unstarted pair is allowed only until its no-start budget expires; once any expected pair remains unstarted beyond `streaming_idle_timeout`, the transaction is treated as stalled regardless of sibling progress;
- an expected pair missing from the tracker counts as terminal success only if its prior completed state is still retained in the tombstone/late-ACK window; otherwise it is treated as deleted and failed for wait-policy purposes.

Implementation pseudocode:

```text
if no DownloadService transaction was created:
    return success
if DOWNLOAD_COMPLETE_CB fired:
    return success_or_failure_from_transaction_status
if any expected ref/receiver pair is terminal-failure/timeout/deleted/aborted:
    return failure
if all expected ref/receiver pairs are terminal-success:
    return continue_until_post_completion_grace_or_callback
if any started non-terminal ref/receiver pair has no progress within streaming_idle_timeout:
    return failure
if any unstarted ref/receiver pair has exceeded streaming_idle_timeout since transaction creation:
    return failure
return continue_waiting
```

If `state: "completed"` is observed for every expected `(ref_id, receiver_id)` pair before `DOWNLOAD_COMPLETE_CB` fires, wait for the same post-completion grace used by the forward path (`STREAM_PROGRESS_COMPLETION_ACK_GRACE`, 30 seconds) rather than returning immediately. If the callback does not arrive within that grace window, return success based on the observed all-terminal-success state. This covers callback ordering races while keeping completion bounded.

Reverse progress events must use the shared schema:

```json
{
  "direction": "result_upload",
  "transfer_id": "ref-id",
  "receiver_id": "receiver-fqcn-or-site",
  "sequence": 12,
  "bytes_done": 734003200,
  "items_done": 700,
  "timestamp": 1790000000.0,
  "state": "active"
}
```

Terminal states are the same as the forward path: `completed`, `failed`, and `aborted`.

Small-result and metrics-only results should not wait. If no large-object download transaction is created, the subprocess should proceed immediately as it does today.

Reverse-path abandonment triggers:

- user cancel, job kill, or abort signal;
- CJ pipe close, peer-gone, or subprocess stop;
- subprocess process crash or restart, with stale in-memory transfer state discarded on startup;
- downstream receiver download failure, timeout, or transaction deletion;
- fixed `download_complete_timeout` expiry on fallback paths where progress tracking is unavailable.

Abandoned reverse transfers should be marked terminal `aborted` or `failed`, release their source objects through normal transaction cleanup, and stop influencing later task/result submissions.

Reverse-path terminal progress records should be retained through the same late-ACK/tombstone window used by `DownloadService` for completed refs, then pruned. This keeps delayed completion/EOF handling consistent with the forward path without retaining stale transfer state indefinitely. The tombstone/late-ACK retention window must stay longer than the longest plausible wait-policy iteration so a legitimately completed ref is not pruned mid-wait and misclassified as deleted.

### 7. Treat Progress-Aware Waiting As Backpressure

Progress-aware waiting must not simply "wait longer." It should also avoid piling up duplicate large payload work.

Rules:

- Do not resend the same task while its previous streamed payload is still active.
- Do not enqueue a new same-task payload behind an active one for the same subprocess.
- Do not start a new `result_upload` for the same task/result while a prior reverse result upload remains active.
- Ensure transaction cleanup still runs after success, failure, timeout, or deletion.
- Ensure PASS_THROUGH and lazy-ref state remain scoped to the correct task/transfer.
- Clear active-transfer state on job kill, user cancel, `abort_task`, peer-gone, or pipe close. Abandoned transfers should be marked with a terminal `aborted` or `failed` state so they cannot block a later legitimate task with the same task name.
- When a transfer reaches `completed`, retain enough per-transfer state through the DownloadService tombstone/late-ACK window so delayed EOF or completion acknowledgements do not hit a missing progress entry.

### 8. Keep Contract Symmetry Without Code-Placement Symmetry

The forward and reverse paths should share the same progress contract and tracker semantics, but their wait policies belong at different control points.

| Area | Forward Task Payload | Reverse Result Payload |
| --- | --- | --- |
| Direction | `task_payload_download` | `result_upload` |
| Progress schema | `sequence`, `bytes_done`, `items_done`, `state`; `receiver_id` optional | same, with `receiver_id` required when multiple downstream receivers may pull the same source ref |
| Timeout model | idle timeout | idle timeout |
| Wait owner | CJ / `TaskExchanger` | subprocess / `FlareAgent` |
| Progress source | subprocess materializing task payload | subprocess `DownloadService` serving result chunks to server or peer receivers |
| Purpose | avoid task resend during materialization | keep result source alive during downstream receiver pull |

This avoids an artificial implementation symmetry. The contract is shared; the code path is owned by the component that is actually waiting.

### 9. Heartbeat Integration Is Staged

Changing heartbeat semantics is higher risk than suppressing task resend.

Phase 1:

- Use progress awareness to suppress task resend while materialization continues.
- Use reverse `result_upload` progress only for subprocess result-submit waiting, not for heartbeat/liveness.
- Keep heartbeat behavior mostly unchanged.

Phase 2:

- Include transfer progress in peer-liveness decisions.
- Compute liveness from:

```text
last_peer_active = max(
    last heartbeat received,
    last pipe read/write activity,
    last monotonic streaming progress
)
```

- Enforce `streaming_max_peer_silence` so activity cannot keep a dead peer alive forever.

This can be guarded for one release if needed.

Heartbeat progress-awareness initially applies to forward CJ/subprocess peer liveness. Reverse `result_upload` progress should not mark the downstream receiver alive from the subprocess point of view unless a later design explicitly extends liveness semantics for that path.

### 10. Backward Compatibility With Existing Timeout Knobs

Keep existing timeout knobs for compatibility:

- `peer_read_timeout`
- `heartbeat_timeout`
- `*_min_download_timeout`
- `*_streaming_per_request_timeout`
- `download_complete_timeout`

In `main`, they should become advanced overrides, not the normal solution.

`streaming_idle_timeout` is the shared idle-timeout knob for both `task_payload_download` and `result_upload`. V1 should not introduce separate forward/reverse idle settings unless operational evidence shows the two directions need different policies.

Where fixed values are not explicitly configured by the user, derive effective defaults from the generic idle policy:

```text
effective_peer_read_timeout = max(default_peer_read_timeout, streaming_idle_timeout)
effective_heartbeat_timeout = max(default_heartbeat_timeout, streaming_idle_timeout)
effective_min_download_timeout = max(default_min_download_timeout, streaming_idle_timeout)
```

Explicit user-configured values should be honored to preserve fast-fail semantics. If a user explicitly sets `peer_read_timeout=120`, NVFLARE should not silently raise it to 600. Instead, log that the explicit fast-fail setting can still interrupt slow streamed transfers.

That warning should be a startup WARNING-level log when an explicit `peer_read_timeout` or `heartbeat_timeout` is lower than `streaming_idle_timeout`.

`_MIN_DOWNLOAD_TIMEOUT=300` should become a fallback only when no generic `streaming_idle_timeout` is configured. It should not remain a separate hard floor once the generic idle policy is active.

## Relationship To Existing Fixes

- 2.8 receive/decomposer safety guards: already in `main`; complementary and required.
- DownloadService tombstones: complementary; still needed for delayed/lost EOF replies after clean completion.
- PASS_THROUGH and lazy-ref memory/scoping fixes: complementary; progress-aware waiting assumes these are correct.
- Timeout warning PRs: already in `main`; warnings should eventually be downgraded or rewritten once progress-aware streaming becomes default.
- Static timeout alignment guidance: superseded for normal users, retained only for advanced troubleshooting.

## Testing Requirements

Use deterministic fake-clock tests for timeout behavior.

Required tests:

1. External-process task-send positive:
   - `peer_read_timeout` would expire under old behavior.
   - subprocess emits monotonic progress events.
   - CJ does not resend.
   - task eventually completes.

2. External-process task-send negative:
   - no ACK and no monotonic progress.
   - fails/retries after `streaming_idle_timeout`.

3. Cross-process progress test:
   - subprocess-side simulated download emits progress through pipe.
   - parent-side CJ receives and records progress.
   - parent timeout decision uses the received progress.

4. Sibling transfer contamination test:
   - transfer A progresses.
   - transfer B stalls.
   - B still times out; A must not mask B.

5. Stuck activity test:
   - repeated progress events with unchanged counters.
   - not considered progress.
   - transfer times out.

6. Heartbeat upper-bound test:
   - buggy/stuck activity signal cannot keep peer alive beyond `streaming_max_peer_silence`.

7. Emitter-death test:
   - progress emitter stops mid-transfer.
   - materialization no longer reports monotonic progress.
   - transfer idles out cleanly with no false success.

8. Abort/abandon cleanup test:
   - active transfer is abandoned because of job cancel, abort, peer-gone, or pipe close.
   - active-transfer state is cleared or marked terminal.
   - a later legitimate task with the same task name is not blocked by stale state.

9. Out-of-order progress test:
   - progress events arrive out of order.
   - stale sequence/counter values are ignored.
   - counters and `last_progress_time` never move backwards.

10. Explicit fast-fail override honors the user-configured timeout:
   - user explicitly configures `peer_read_timeout` lower than `streaming_idle_timeout`.
   - NVFLARE uses the lower configured timeout rather than silently applying `max(configured, streaming_idle_timeout)`.
   - startup WARNING-level log explains that the explicit fast-fail setting can still interrupt slow streamed transfers.

11. Receive timeout safety preserves request/response state:
   - timed-out `receive()` returns `None`.
   - `send()` after timed-out receive raises instead of writing to stale request state.
   - later successful receive/send still works after the timeout path.
   - both external-process and in-process APIs cover the same behavior.

12. FOBS download failure safety rejects failed artifacts without rejecting valid metadata:
   - missing downloaded item raises.
   - explicit `None` downloaded item raises.
   - valid model with intentional `None` metadata remains allowed outside the download-failure path.

13. Scale simulation completes while many slow receivers keep making progress:
   - many synthetic receivers pull concurrently.
   - chunks follow a slow schedule with variable delay.
   - progress continues past the old 300-second-equivalent threshold.
   - transfer completes without fixed-duration timeout failure.

14. Reverse result-upload keeps the subprocess alive while the server pulls a large result:
   - subprocess submits a large pass-through result.
   - server-side simulated pulls continue beyond the historical fixed wait threshold.
   - subprocess keeps waiting while `result_upload` progress is recent.
   - completion callback ends the wait successfully.

15. Reverse result-upload no-start path fails clearly when the receiver never pulls:
   - subprocess creates a downloadable result transaction.
   - server never starts pulling chunks.
   - no-start budget is measured with `streaming_idle_timeout`.
   - subprocess exits the wait path with a clear timeout or warning.

16. Reverse result-upload stall path fails after progress goes idle:
   - server pulls some chunks.
   - `result_upload` progress then stops.
   - subprocess does not wait indefinitely.
   - failure or warning identifies an idle result-upload stall.

17. Reverse result-upload skips progress waiting for non-streamed results:
   - result contains no streamed tensors.
   - no download transaction is created.
   - subprocess proceeds immediately without waiting for progress.

18. Reverse result-upload completes only after every downloadable ref succeeds:
   - result contains multiple downloadable refs.
   - progress and completion remain per-transfer.
   - one completed ref does not mask an active, failed, or stalled sibling ref.
   - final completion is reported only after all refs reach terminal success.

19. Reverse result-upload fallback keeps legacy timeout behavior when progress tracking is unavailable:
   - reverse progress tracking cannot be installed for a created download transaction because progress callback registration
     fails or the transaction/ref type exposes no progress hook.
   - existing `download_complete_timeout` behavior is used.
   - no new progress wait is required for that fallback path.

20. Reverse result-upload post-completion grace handles callback ordering races:
   - every expected ref reports `state: "completed"` before `DOWNLOAD_COMPLETE_CB` fires.
   - subprocess continues waiting for the short post-completion grace period.
   - callback arrival during the grace period completes the wait successfully.
   - missing callback after the grace period returns success based on all observed terminal-success states.

21. Swarm/peer result-upload isolates progress by downstream receiver:
   - one subprocess result ref is pulled by multiple downstream receivers, such as a server and swarm aggregation client
     parent or two peer client parents.
   - receiver A reports recent progress.
   - receiver B never starts or stalls beyond `streaming_idle_timeout`.
   - receiver A's progress does not mask receiver B; the expected `(ref_id, receiver_id)` pair for B fails or stalls
     clearly.

22. Swarm task-payload progress suppresses resend at the client parent:
   - client agent downloads a task/model payload from its client parent rather than from the FL server.
   - "client parent" here refers to the CJ/`TaskExchanger` of the peer client serving as the parent in swarm topology.
   - `_STREAM_PROGRESS_` reaches the waiting client parent.
   - the client parent does not resend while client-agent materialization progress remains recent.

## Implementation Order

Each implementation PR should land with the matching deterministic tests from the test plan.

1. Preserve existing safety guards before changing wait behavior:
   - keep receive timeout safety and FOBS downloaded-item safety tests green.
   - cover both external-process and in-process receive/send paths.
   - confirm intentional `None` model metadata is still allowed outside failed-download artifacts.

2. Add the shared progress model and fake-clock tracker tests:
   - implement the direction-neutral progress record and monotonic `TransferProgressTracker`.
   - cover stale sequence handling, unchanged counters, sibling-transfer contamination, and emitter death.
   - include receiver-scoped keys so later `result_upload` and swarm tests do not require tracker redesign.

3. Wire forward task-payload progress emission and pipe delivery:
   - emit monotonic `task_payload_download` progress from the canonical materialization path.
   - send `_STREAM_PROGRESS_` through the subprocess-to-CJ pipe.
   - record the event on the CJ side and prove parent timeout decisions use the received progress.

4. Add forward task-send wait policy and backpressure:
   - suppress resend while per-transfer monotonic progress remains recent.
   - fail or retry clearly after `streaming_idle_timeout` when there is no progress.
   - clear or mark active-transfer state on abort, peer-gone, job kill, or pipe close.
   - honor explicit fast-fail overrides and log the startup WARNING when they can interrupt slow streamed transfers.

5. Add scale coverage for the forward path:
   - simulate many receivers with slow chunks and variable delay.
   - prove progress can continue past the old 300-second-equivalent threshold.
   - keep the real 12/16-client x 5GB K8s run as nightly or release-validation coverage, not a standard PR test.

6. Wire reverse result-upload progress as the second consumer of the shared contract:
   - emit source-side `result_upload` progress from subprocess `DownloadService`.
   - track progress in subprocess result-submit waiting through the same schema and tracker.
   - include downstream `receiver_id` for every expected pull when a result ref can be consumed by multiple receivers.
   - skip progress waiting when no large-object download transaction is created.
   - fall back to existing `download_complete_timeout` behavior when progress tracking cannot be installed.

7. Add reverse result-upload wait-policy tests in the same PR as the reverse implementation:
   - positive large-result path continues while server pull progress remains recent.
   - no-start path exits clearly after `streaming_idle_timeout`.
   - stall path exits clearly when progress stops after some chunks.
   - multi-ref completion reports final success only after every expected ref reaches terminal success.
   - post-completion grace handles `state: "completed"` arriving before `DOWNLOAD_COMPLETE_CB`.

8. Add swarm and peer topology coverage:
   - result-upload receiver isolation proves one receiver's progress cannot mask another receiver's no-start or stall.
   - task-payload progress from a client agent to its client parent reaches that parent through `_STREAM_PROGRESS_`.
   - the client parent does not resend while client-agent materialization progress remains recent.

9. Add heartbeat integration after task resend and result-submit waits are stable:
   - include monotonic transfer progress in peer-liveness decisions.
   - enforce `streaming_max_peer_silence` so activity cannot keep a dead peer alive indefinitely.
   - keep reverse `result_upload` progress out of downstream receiver liveness unless a later design explicitly extends it.

10. Revisit timeout warnings and public docs:
    - rewrite or downgrade warnings that are no longer actionable once progress-aware streaming is the default.
    - document `streaming_idle_timeout`, `streaming_max_peer_silence`, and explicit fast-fail override behavior.
    - keep legacy timeout knobs documented as advanced compatibility overrides.

## Decisions From Review

1. Transfer ID: use `DownloadService` ref ID when available. It is already stable across materialization and known on both sides. If a stream has no ref ID, use stream ID as a fallback and record the ID kind.
2. Progress event location: use a dedicated `_STREAM_PROGRESS_` internal pipe topic. Do not piggyback on heartbeat or task-control messages.
3. Heartbeat progress-awareness: guard for one release. The task-resend suppression can be enabled first; heartbeat integration is the riskier phase.
4. Default `streaming_idle_timeout`: 600 seconds.
5. `_MIN_DOWNLOAD_TIMEOUT=300`: fallback only when no generic idle timeout is configured, not an additional hard floor.
