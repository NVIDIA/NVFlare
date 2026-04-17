# Live Job Log Streaming

This package provides real-time log streaming from clients to the server
during federated jobs.  It replaces the deprecated `ErrorLogSender` /
`LogReceiver` pair, which could only send a static snapshot of the error log
after a job finished.

## Components

| Component | Runs in | Purpose |
|-----------|---------|---------|
| `JobLogStreamer` | Job subprocess | Tails a log file and streams bytes to the server in real time |
| `JobLogReceiver` | Server | Receives streamed bytes, writes them to disk, hands the file to the job manager |
| `SystemLogStreamer` | `CLIENT_PARENT` (resources.json) | Injects a `JobLogStreamer` into any job that doesn't already declare one |

### Deprecated (still functional)

| Component | Replacement |
|-----------|-------------|
| `ErrorLogSender` | `SystemLogStreamer` + `JobLogStreamer` |
| `LogReceiver` | `JobLogReceiver` |

## Why streaming runs inside the job subprocess

The streamer must run inside the job subprocess, not in `CLIENT_PARENT`.
With Docker or Kubernetes job launchers the job may execute in a different
container or pod.  The parent process has no filesystem access to the job's
log files in that case.  By injecting `JobLogStreamer` into the job config
(via `SystemLogStreamer`), the streamer always runs where the log file
lives -- regardless of the launch mechanism.

## How SystemLogStreamer works

`SystemLogStreamer` lives in the client's `resources.json`.  When the
`BEFORE_JOB_LAUNCH` event fires (after the job config is deployed to disk
but before the subprocess starts) it:

1. Reads the deployed `config_fed_client.json`.
2. Checks whether a `JobLogStreamer` component is already declared.
3. If not, appends one with the configured parameters (log file name,
   poll interval, liveness interval).
4. Writes the modified config back to disk.

The job subprocess then loads the config and `JobLogStreamer` runs inside it
as if the user had declared it explicitly.

## Two-phase stop (JobLogStreamer)

`JobLogStreamer` registers for three events:

| Event | Action |
|-------|--------|
| `START_RUN` | Starts a background thread that tails the log file |
| `ABOUT_TO_END_RUN` | Sets `stop_event` but does **not** join -- returns immediately |
| `END_RUN` | Joins the streaming thread (blocks until EOF is sent) |

Splitting the stop across two events is critical:

- **`ABOUT_TO_END_RUN`** fires inside `fire_event()`.  If the handler
  blocked on `join()` here, `fire_event()` would not return until the
  stream finished.  Any log lines written by the framework *after*
  `fire_event()` returns (e.g. "ABOUT\_TO\_END\_RUN fired") would be
  missed because the stream would already be closed.  By only setting
  `stop_event`, the handler returns immediately, letting the framework
  write those lines while the streaming thread is still draining.

- **`END_RUN`** joins the thread.  This keeps `client_run()` alive until
  the streaming thread has sent EOF and the server has acknowledged it.
  Without this, `executor.shutdown()` could return before the stream
  finishes, causing `server.abort_run()` to fire and reject the
  in-flight stream with `TASK_ABORTED`.

### Drain retry

When the streaming thread sees `stop_event` and no new data, it does not
send EOF immediately.  Instead it sleeps one more `poll_interval` and
retries the read.  This captures bytes written by cleanup code that runs
just after the stop signal fires (e.g. between `ABOUT_TO_END_RUN` and
`END_RUN`).  If data arrives during the retry, the drain state resets so
another retry can occur -- this handles the case where log lines trickle
in over multiple poll intervals during shutdown.

### Caveat: last few log lines are always missing

The streamed log file will always be missing the last few lines of the
original log.  These lines (e.g. "END\_RUN fired", "End the Simulator
run", "Clean up ClientRunner") are written by the framework *after* the
`END_RUN` event handler returns -- which is after `join()` completes and
the stream is already closed.  There is no way to capture them without
modifying the core framework to emit additional events after those log
calls.  In practice these are job-teardown housekeeping messages and do
not affect the usefulness of the streamed log.

## Fresh abort signal (JobLogStreamer)

The job's `run_abort_signal` is triggered before `ABOUT_TO_END_RUN` fires.
The sender loop in `stream_runner.py` checks this signal at the top of
every iteration and would abort immediately -- dropping buffered log bytes.

To prevent this, `JobLogStreamer` creates a fresh `FLContext` with a new,
untriggered `Signal`:

```python
stream_fl_ctx = fl_ctx.get_engine().new_context()
stream_fl_ctx.put(key=ReservedKey.RUN_ABORT_SIGNAL, value=Signal(),
                  private=True, sticky=False)
```

`put()` is used instead of `set_prop()` because `new_context()`
pre-populates the abort signal as `private+sticky`.  `set_prop()` refuses
to change the sticky mask on an existing key (silent failure); `put()`
bypasses this check.  Setting `sticky=False` ensures `_get_prop()` returns
the local fresh signal immediately without consulting the context manager,
so the original triggered signal never leaks back in.

## Liveness heartbeats and idle timeout

When the log file is quiet (no new bytes), `JobLogStreamer` sends periodic
heartbeat messages (no payload) every `liveness_interval` seconds.  On the
receiver side, `JobLogReceiver` runs a watchdog thread that closes the
stream if no message (data or heartbeat) arrives within `idle_timeout`
seconds.

```
liveness_interval < idle_timeout
```

With the defaults (10 s / 30 s), a healthy sender heartbeats every 10 s so
the 30 s watchdog always resets before it fires.  The timeout only triggers
when the sender is genuinely unreachable (crash, network partition).
