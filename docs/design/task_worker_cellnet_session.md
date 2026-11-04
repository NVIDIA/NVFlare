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

A worker joins its Client Job exactly as the Client Job joins its CP:

| | Client Job joins CP | Worker joins Client Job |
| --- | --- | --- |
| Listener | CP cell's internal listener, site internal settings | Client Job cell's internal listener: the site's scheme and security, the launcher's host, its own port |
| Told how to connect | `--parent_url`, `--parent_conn_sec` | bootstrap `connect_url`, `connection_security` |
| Credential (secure mode) | per-job certificate, pinned in both roles | the same per-job certificate, pinned in both roles |
| Names it may claim | job cell and descendants (certificate scope) | `<job cell>.task_worker_<id>`, inside that scope |

`listen(cell, host)` runs once per executor at `START_RUN`. It takes the
scheme and connection security from the site's internal settings, mTLS
included when the site configures it, but not their address: those settings
name the CP's own host and port. The host comes from the task launcher's
`get_worker_listener_host()` (`localhost` for the Process launcher; a launcher
that places workers elsewhere returns a Client Job address they can reach), and
the port is picked freshly, so each Client Job has its own listener. The
Client Job cell is created with its own FQCN mapped to its own identity in
`auth_identity_map`, so a cell under it must present the job's identity;
together with the certificate scope, only holders of this job's credential can
join, and only under this job's names. In a clear deployment the listener is
clear, the same as the CP's.

`CellWorkerSession` is the Client Job half. `TaskWorkerExecutor.execute`
creates one per attempt with a fresh worker FQCN, which registers it, and
closes it as soon as the supervisor reports the worker
settled. Closing never waits on a stalled receiver longer than the settlement
timeout, and a record that arrives afterwards is rejected, so a metric only
fires while its attempt is owned. `bootstrap_fields` is what goes into the
bootstrap's `cell_session`: `connect_url`, `worker_fqcn`, and the
Client Job's `connection_security`, `identity` and credential paths where it
has them: the CA and both of its pairs, which in secure mode are the job
credential. A cell also holds a server role, which message crypto prefers, and
fills a missing pair from the CA's folder, i.e. the site kit; passing both
pairs pins both of the worker's roles to the job credential, exactly as
`FederatedClientBase` pins the Client Job's. The site's private key is never
needed by a worker.

`CellWorkerTransport` is the worker half. `run_worker` opens it once every codec
is registered, right before reading the input, and closes it on exit; a close
that fails is logged and never replaces the attempt's outcome. `open` creates
a cell under the Client Job cell with those settings, waits for the connection
and says HELLO. `log(record)` is the only thing it does afterwards. Records go as one
framed message on the parent connection, not through the blob-streaming layer
the Cell uses for application channels, and are FOBS-encoded natively by the
worker so array values (images, histograms) travel inline instead of as
download references. The Client Job fires one record at a time per session,
since they all share the attempt's `FLContext`.

| Direction | Topic | Payload | Reply |
| --- | --- | --- | --- |
| worker to CJ | `task_worker.hello` | none | ok |
| worker to CJ | `task_worker.log` | the metric's DXO Shareable | ok once fired into the job |

Several `TaskWorkerExecutor`s share one Client Job cell, whose callback registry
holds one handler per topic, so requests are routed to their session by the
sender's FQCN. The connection credential is the only authentication, as for
the Client Job's own link to the CP: a request from a worker name with no open
session, including a settled attempt's, is rejected. HELLO exists only so a
worker that cannot reach its Client Job fails before the script runs.

### 2.3 Metrics

`bind_client_task_context` hands each Client API log call to `log` as the
metric's DXO Shareable. The Client Job answers the request after
`send_analytic_dxo` on the executing attempt's `FLContext`, on a cell thread,
while `execute` is still blocked in the supervisor; the resident
`ConvertToFedEvent` widget forwards it to the server as for any resident
executor. A metric the Client Job rejects (not a DXO, undecodable, or arriving
after the attempt settled) fails the worker's attempt. A receiver that raises
does not: event dispatch logs the exception and records it on the context, the
same as for a resident executor in job lifetime, so a failing tracking backend
never fails training; the replayed spool behaved the same. The worker's `log`
therefore means delivered and fired, not processed by every receiver. The
analytics spool, its artifact kind
and its replay are gone. `cell_session` stays optional in the bootstrap for
executors that never log, but a Client API worker refuses to start without one.

## 3. Consequences

- The Client Job remains the only federation endpoint; the worker talks only to
  its Client Job. Hop count is unchanged.
- The security boundary is the one the CP already draws for Client Jobs: the
  job credential. A worker holding it can reach the Client Job cell's other
  routes, as code running inside the Client Job can in job lifetime; a process
  without it cannot connect in a secure deployment. In a clear deployment
  nothing on the internal listeners is authenticated, for the CP's listener
  as for this one.
- Launchers need a network path from the worker to the Client Job listener and
  the job credential: the Process launcher shares the host and the files. A
  container launcher returns the Client Job's reachable address from
  `get_worker_listener_host()` and mounts the job credential where the
  bootstrap's paths point, as it already does for the Client Job.
- Outbound calls from a worker (Collab) are the next thing to carry over this
  session: the worker has a Cell and a proven identity, so a call is one more
  request the Client Job authorises and dispatches, with no second mechanism.
  `TaskRuntime.get_cell` still refuses until that exists.
