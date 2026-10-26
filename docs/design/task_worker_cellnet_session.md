# Task worker cellnet session

Status: implemented, on top of the task worker lifecycle (`task_worker_lifecycle.md`).
The disk handoff of that document is unchanged; this one describes the one
live channel that now exists next to it.

## 1. Why a channel at all

The #5405 worker exchanged everything with its Client Job through files in the
attempt directory. That is right for what must survive a process: the input a
worker starts from, its saved result, `completion.json`, the sealed checkpoint
and `failure.json` are all read back after the worker is gone, and a file with
a digest is the cheapest durable record. It is wrong for what two live
processes say to each other while both are running:

- Metrics. Log calls were spooled to disk and replayed by the Client Job after
  settlement, so dashboards lagged by one attempt and a worker killed mid-task
  lost every metric it had logged.
- Outbound calls. A worker that needs the federation (Collab calls, model
  download) needs a request/reply channel to the resident process; a file
  cannot be one.

Both need the same thing: a Cell. The Client Job already is one, and the Client
API external trainer already connects to it as a child cell. The task worker now
does the same.

## 2. What was built

### 2.1 The split

| Exchange | Carrier |
| --- | --- |
| Input, staged before launch | disk, as before |
| Result, `completion.json`, checkpoint, `failure.json` | disk, written by the worker as before |
| Metrics | cell session, one request per log call |

The cell is additive: `TaskSupervisor` still launches the worker, polls the
process and reads the result from disk exactly as before. It does not know the
session exists. There is no attempt handoff, no group, no end signal and no
heartbeat: the worker exits after its one attempt, and the process poll remains
the Client Job's authority on whether the worker is alive.

### 2.2 The session

`nvflare/private/fed/task_worker/cell_session.py` holds both halves.

`listen(cell)` runs once per executor at `START_RUN`, as the external-process
backend does: it makes the Client Job cell's clear TCP loopback internal
listener and routes the two topics below. `CellWorkerSession` is the Client Job
half. `TaskWorkerExecutor.execute` creates one per attempt with a fresh worker
FQCN (`<cj>.task_worker_<id>`) and a launch token, which registers it, and
closes it as soon as the supervisor reports the worker settled, so a record
can only ever fire while its attempt is owned. `bootstrap_fields` is what
goes into the bootstrap's `cell_session`: `connect_url`, `worker_fqcn`, `token`.
The Client Job FQCN is the worker FQCN's parent.

`CellWorkerTransport` is the worker half. `run_worker` opens it once every codec
is registered, right before reading the input, and closes it on exit. `open`
creates a Cell under the Client Job cell, waits for the connection and says
HELLO. `log(record)` is the only thing it does afterwards. Records go as one
framed message on the parent connection, not through the blob-streaming layer
the Cell uses for application channels, and are FOBS-encoded natively by the
worker so array values (images, histograms) travel inline instead of as
download references. The Client Job fires one record at a time per session,
since they all share the attempt's `FLContext`.

| Direction | Topic | Payload | Reply |
| --- | --- | --- | --- |
| worker to CJ | `task_worker.hello` | `token` | ok |
| worker to CJ | `task_worker.log` | `token`, the metric's DXO Shareable | ok once fired into the job |

Several `TaskWorkerExecutor`s share one Client Job cell, whose callback registry
holds one handler per topic, so requests are routed to their session by the
sender's FQCN. The token on every request is the one credential: a request from
an unknown FQCN or with another session's token is rejected. HELLO exists only
so a worker that cannot prove its launch fails before the script runs.

### 2.3 Metrics

`bind_client_task_context` hands each Client API log call to `log` as the
metric's DXO Shareable. The Client Job answers the request after
`send_analytic_dxo` on the executing attempt's `FLContext`, on a cell thread,
while `execute` is still blocked in the supervisor; the resident
`ConvertToFedEvent` widget forwards it to the server as for any resident
executor. A metric the Client Job rejects (not a DXO, or a receiver that raises) fails the worker's attempt,
as a rejected spool write did before. The analytics spool, its artifact kind
and its replay are gone. `cell_session` stays optional in the bootstrap for
executors that never log, but a Client API worker refuses to start without one.

## 3. Consequences

- The Client Job remains the only federation endpoint; the worker talks only to
  its Client Job. Hop count is unchanged.
- Launchers need a network path from the worker to the Client Job listener:
  loopback for the Process launcher. Nothing supported today lacks it.
- Outbound calls from a worker (Collab) are the next thing to carry over this
  session: the worker has a Cell and a proven identity, so a call is one more
  request the Client Job authorises and dispatches, with no second mechanism.
  `TaskRuntime.get_cell` still refuses until that exists.
