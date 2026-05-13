# Live Job Log Streaming Design

## 1. Overview

This document describes the design of NVFlare's live job log streaming
mechanism. The feature provides near-real-time delivery of job log data from
clients to the server while a federated job is still running.

The design provides a runtime-oriented log streaming path for the primary
live-streaming use case. It streams active log output while a federated job is
still running and works across launcher environments where the job subprocess
owns the relevant log files.

## 2. Problem Statement

The legacy log-transfer path sends a completed file only after the job finishes.
That behavior is insufficient when:

- operators need to observe logs while a job is still executing
- a job runs in a separate runtime boundary such as Docker or Kubernetes
- the parent client process cannot directly access the job subprocess log files

In those environments, a client-side site-level component running in
`CLIENT_PARENT` cannot reliably read the job's log file. The streaming logic
must execute inside the job subprocess that owns the filesystem view of the
active log.

## 3. Goals and Non-Goals

### Goals

- Stream job log data from clients to the server while the job is running.
- Support launchers where the job subprocess runs in a different container or
  pod than the client parent process.
- Preserve log data written during shutdown as much as the framework lifecycle
  allows.
- Detect dead or disconnected senders during long quiet periods.
- Keep server-side handling compatible with job artifact storage.

### Non-Goals

- Capture framework log lines emitted after the `END_RUN` handler has already
  returned.
- Replace every historical log-transfer path at once.
- Enforce cross-site timeout consistency at runtime. Deployment still needs to
  configure compatible heartbeat and timeout values.

## 4. Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `JobLogStreamer` | Job subprocess | Tails a log file, sends data chunks, emits liveness heartbeats, drains remaining bytes on shutdown |
| `JobLogReceiver` | Server | Receives stream chunks, writes them to disk immediately, finalizes the result into job-managed storage |
| `SiteLogStreamer` | Client `resources.json` / `CLIENT_PARENT` | Injects `JobLogStreamer` into jobs that do not already declare one |

## 5. High-Level Architecture

The design splits the feature into three responsibilities:

1. Site-level injection on the client.
2. Runtime streaming in the job subprocess.
3. Stream reception and persistence on the server.

`SiteLogStreamer` handles policy and configuration. It modifies the deployed
job configuration just before launch so that the job process loads a
`JobLogStreamer` component even when the job author did not explicitly add one.

`JobLogStreamer` handles runtime data movement. It discovers the active log
directory from the process logging configuration, tails the target file, and
streams new bytes to the server over the streaming infrastructure.

`JobLogReceiver` handles server-side persistence. It writes incoming chunks
directly to disk so the file can be observed while the job is running, and then
hands the completed file to the job manager when the stream ends successfully.

## 6. Why the Streamer Runs Inside the Job

The key architectural choice is that log streaming runs in the job subprocess,
not in `CLIENT_PARENT`.

With local launchers, the parent process may be able to read the job's log
files. With Docker or Kubernetes launchers, that assumption does not hold. The
job may execute in a different container or pod, and the parent process may
have no filesystem access to the log path at all.

Running `JobLogStreamer` inside the job guarantees that the streamer executes in
the same environment as the log-producing process. `SiteLogStreamer` exists
to make that placement automatic.

## 7. Configuration Injection Flow

`SiteLogStreamer` is declared in the client's `resources.json`. On
`BEFORE_JOB_LAUNCH`, after the job configuration has been deployed to disk but
before the subprocess starts, it performs the following steps:

1. Read the deployed `config_fed_client.json`.
2. Inspect the configured components.
3. Determine whether a `JobLogStreamer` is already present.
4. If absent, append a new `JobLogStreamer` entry with the configured
   parameters.
5. Write the updated configuration back to disk.

The job subprocess then starts with the modified configuration and loads the
streamer as a normal job component.

## 8. Runtime Lifecycle

### 8.1 Startup

`JobLogStreamer` registers for:

| Event | Behavior |
|-------|----------|
| `START_RUN` | Create a background streaming thread and begin tailing the log |
| `ABOUT_TO_END_RUN` | Signal the thread to stop draining, but do not join |
| `END_RUN` | Join the thread and wait for EOF delivery |

At startup, `JobLogStreamer` locates the log directory from active Python file
handlers and constructs the target file path from the configured base name.

### 8.2 Steady-State Streaming

The streamer tails the file rather than treating it as a static snapshot. It:

- reads appended bytes in chunks
- sends those chunks to the server
- tolerates log rotation by reopening the file when inode or size behavior
  indicates a rotation or truncation event
- emits heartbeats when the log is quiet

### 8.3 Shutdown

Shutdown is intentionally split across `ABOUT_TO_END_RUN` and `END_RUN`.

If `ABOUT_TO_END_RUN` blocked on `join()`, the event path would not return until
streaming finished, and framework log lines written immediately after that event
would be missed. Instead, `ABOUT_TO_END_RUN` only sets the stop signal.

`END_RUN` performs the join. This keeps `client_run()` alive until the stream
has drained and EOF has been sent, reducing the chance that later shutdown logic
aborts the stream before the final bytes reach the server.

## 9. Drain Behavior

When the streamer sees a stop condition and no new data is immediately
available, it does not send EOF on the first empty read. Instead, it sleeps for
one additional `poll_interval` and retries the read once more.

This extra drain window is designed to capture shutdown-era log writes that land
just after the stop signal is raised. If data appears during the retry, the
drain state resets so another final retry can occur later if needed.

## 10. Abort-Signal Handling

The job's original `run_abort_signal` may already be triggered before
`ABOUT_TO_END_RUN` is processed. If the streaming path reused that original
signal directly, the sender loop could terminate immediately and drop buffered
log bytes.

To prevent that behavior, `JobLogStreamer` creates a fresh `FLContext` with a
new `Signal` for the streaming thread. The implementation uses `put()` rather
than `set_prop()` because the inherited abort signal is pre-populated with a
sticky mask, and `set_prop()` will not change the mask of an existing property.

This design isolates graceful streaming shutdown from the job's broader abort
path.

## 11. Liveness and Idle Detection

Log streams can be idle for long periods without indicating failure. To
distinguish a quiet log from a dead sender, the design uses heartbeats and an
idle watchdog.

- `JobLogStreamer` sends a heartbeat every `liveness_interval` seconds when no
  new data has been sent.
- `JobLogReceiver` tracks the time of the last received message, including
  heartbeats.
- If no message arrives within `idle_timeout`, the receiver closes the stream
  and marks it as timed out.

The required relationship is:

```text
liveness_interval < idle_timeout
```

With the default values of `10s` and `30s`, a healthy sender refreshes the
receiver's idle timer well before it expires.

## 12. Server-Side Persistence Model

`JobLogReceiver` writes incoming chunks directly to a file as they arrive. This
allows operators to inspect the evolving file during job execution.

When the stream finishes successfully:

- if a job manager is available, the receiver passes the file into
  `job_manager.set_client_data(...)`
- if a job manager is not available, such as some simulator scenarios, the file
  is retained in a workspace-accessible location

The receiver uses trusted peer context from the server-side `FLContext` when
associating a stream with a client and job. This prevents the server from
treating sender-supplied stream metadata as authoritative for storage identity.

## 13. Security Considerations

The design treats stream payload metadata and storage identity differently.

- The streamed file name is used as descriptive metadata and is sanitized before
  being used in file paths.
- Client identity and job identity used for persistence must come from the
  trusted peer context maintained by the server runtime, not directly from
  sender-provided stream context.

This avoids path-manipulation and misassociation risks where a modified client
attempts to store data under another client or job.

## 14. Known Limitations

Some late framework log lines are still unavoidably absent from the streamed
output. Specifically, log messages emitted after the `END_RUN` handler returns
cannot be captured because the stream has already been closed at that point.

Examples include late teardown messages such as:

- `END_RUN fired`
- simulator shutdown messages
- final cleanup messages emitted after the join completes

Capturing those lines would require additional framework lifecycle hooks after
the current shutdown sequence.

## 15. Operational Guidance

- Prefer `SiteLogStreamer` in client `resources.json` when jobs should receive
  log streaming automatically.
- Keep `liveness_interval` strictly smaller than `idle_timeout`.
- Use one `JobLogStreamer` per log file when multiple files must be streamed.
- Expect the streamed file to be highly useful for live diagnostics, but not to
  contain every final teardown log line.

## 16. Summary

The live job log streaming design moves log transport into the job subprocess,
adds site-level injection for broad coverage, uses heartbeat-based liveness
detection, and preserves compatibility with NVFlare's server-side job artifact
storage model.

The result is a runtime-oriented logging path that works across launcher models
and provides materially better observability than a post-run file handoff.
