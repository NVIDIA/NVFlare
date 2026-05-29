# Progress-Aware Streaming For Congested Transfers

## Goal

Make streamed task/result transfer resilient to slow networks, large payloads, and many concurrent receivers without requiring users to tune several timeout settings in lockstep.

The default behavior should be:

> If a streamed transfer is making monotonic progress, do not fail or resend solely because wall-clock time is long. If progress stops for the idle timeout, fail clearly.

## Current Problem

In the 5GB x 16-client failure, the parent client job (CJ) can hit `peer_read_timeout` while the subprocess is still downloading/materializing the task payload. The CJ then resends the task even though the previous task payload is still active. Under congestion this creates duplicate work, stale state, and eventually corrupt or premature receive behavior.

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

### 4. Add Cross-Process Progress Signalling

The side materializing streamed bytes must report progress to the side waiting on that materialization.

For task payload download:

- materializing side: subprocess
- waiting side: CJ
- direction: `task_payload_download`

For result upload:

- materializing side: CJ/server-side receiver path
- waiting side: subprocess waiting for result transfer completion
- direction: `result_upload`

The first implementation must cover `task_payload_download`, because that is the known 5GB x 16-client failure. The same event schema and tracker must be direction-neutral so `result_upload` can use the same mechanism rather than a separate timeout model.

Preferred mechanism: pipe event, not shared memory or filesystem state.

Example event:

```json
{
  "topic": "STREAM_PROGRESS",
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

Canonical byte-progress emitters:

- download/materialization side: `ByteReceiver`
- upload/send side: `ByteStreamer`

Higher layers such as `ViaDownloader` and `DownloadService` may attach transfer IDs, item counts, state transitions, and callbacks, but they should not independently emit competing byte counters for the same transfer. This avoids double-counting and conflicting `bytes_done` values.

Emit progress events:

- on transfer start
- every 30 seconds while active, if counters advanced
- on completion
- on failure

Do not emit per chunk.

Progress events may travel on the same pipe as other control messages in the first implementation. This is acceptable only because the emit interval is much smaller than the default idle timeout (`30s << 600s`). If tests show progress events can be queued behind the large message they are meant to protect, move `STREAM_PROGRESS` to a priority channel before enabling heartbeat integration.

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

### 6. Treat Progress-Aware Waiting As Backpressure

Progress-aware waiting must not simply "wait longer." It should also avoid piling up duplicate large payload work.

Rules:

- Do not resend the same task while its previous streamed payload is still active.
- Do not enqueue a new same-task payload behind an active one for the same subprocess.
- Ensure transaction cleanup still runs after success, failure, timeout, or deletion.
- Ensure PASS_THROUGH and lazy-ref state remain scoped to the correct task/transfer.
- Clear active-transfer state on job kill, user cancel, `abort_task`, peer-gone, or pipe close. Abandoned transfers should be marked with a terminal `aborted` or `failed` state so they cannot block a later legitimate task with the same task name.
- When a transfer reaches `completed`, retain enough per-transfer state through the DownloadService tombstone/late-ACK window so delayed EOF or completion acknowledgements do not hit a missing progress entry.

### 7. Heartbeat Integration Is Staged

Changing heartbeat semantics is higher risk than suppressing task resend.

Phase 1:

- Use progress awareness to suppress task resend while materialization continues.
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

### 8. Backward Compatibility With Existing Timeout Knobs

Keep existing timeout knobs for compatibility:

- `peer_read_timeout`
- `heartbeat_timeout`
- `*_min_download_timeout`
- `*_streaming_per_request_timeout`
- `download_complete_timeout`

In `main`, they should become advanced overrides, not the normal solution.

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

10. Explicit fast-fail override test:
   - user explicitly configures a lower `peer_read_timeout`.
   - NVFLARE honors it rather than silently applying `max(configured, streaming_idle_timeout)`.
   - startup WARNING-level log explains the fast-fail consequence.

11. Receive safety tests:
   - timed-out `receive()` returns `None`.
   - `send()` after timed-out receive raises.
   - later successful receive/send works.
   - cover both external-process and in-process APIs.

12. FOBS safety tests:
   - missing downloaded item raises.
   - explicit `None` downloaded item raises.
   - valid model with intentional `None` metadata remains allowed outside the download-failure path.

13. Scale simulation:
   - many synthetic receivers.
   - slow chunk schedule and variable delay.
   - progress continues past the old 300-second-equivalent threshold.
   - transfer completes.

The real 12/16-client x 5GB K8s run should remain a nightly or release-validation test, not a standard PR test.

## Implementation Order

1. Verify the existing `main` safety guards and keep their regression tests green.
2. Add progress record and tracker with fake-clock unit tests.
3. Emit monotonic progress from streaming/download materialization paths.
4. Add subprocess-to-CJ progress pipe event.
5. Add CJ-side progress receiver and per-transfer state.
6. Suppress task resend while per-transfer monotonic progress continues, in the same PR as no-progress failure and sibling-contamination tests.
7. Wire `result_upload` as the second consumer of the same direction-neutral progress contract. If it cannot land in Phase 1, it should ship in the immediately following release milestone, not remain on legacy fixed timeout behavior indefinitely.
8. Add heartbeat integration with `streaming_max_peer_silence`.
9. Revisit timeout warnings and docs.

## Decisions From Review

1. Transfer ID: use `DownloadService` ref ID when available. It is already stable across materialization and known on both sides. If a stream has no ref ID, use stream ID as a fallback and record the ID kind.
2. Progress event location: use a dedicated `STREAM_PROGRESS` pipe topic. Do not piggyback on heartbeat or task-control messages.
3. Heartbeat progress-awareness: guard for one release. The task-resend suppression can be enabled first; heartbeat integration is the riskier phase.
4. Default `streaming_idle_timeout`: 600 seconds.
5. `_MIN_DOWNLOAD_TIMEOUT=300`: fallback only when no generic idle timeout is configured, not an additional hard floor.
