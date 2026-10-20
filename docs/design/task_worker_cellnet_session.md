# Task worker cellnet session

Status: proposal, after task groups (`task_groups.md`). Independent of the
group semantics: groups ship on the disk session protocol of
`task_groups.md` section 5; this document replaces that transport.

## 1. What the disk handoff costs

Inside a task group the Client Job and the worker exchange, per attempt,
one input payload, one result payload, an analytics spool and two small
records. Each payload is serialized to a file, fsynced, digested on write
and again on read. That is correct and adequate for CPU workers on one
host, which is the #5405 profile. It is not adequate for:

- Multi-GB models inside a tight group loop, where the file round trip is a
  measurable share of each attempt.
- Live analytics: log calls are spooled and replayed after the attempt, so
  dashboards lag by one attempt.
- Launchers without a filesystem shared between the Client Job and the
  worker (pods, Slurm steps on another node).
- Collab: a worker that makes outbound calls needs a channel to the
  resident process; files are not one.

Disk still has two jobs that no transport replaces:

| What disk brings | Keep | Why |
| --- | --- | --- |
| State that outlives a worker: the per-assignment checkpoint | yes | the point of task lifetime is that no memory survives between workers |
| Durable records: `completion.json`, `failure.json`, the saved result with its digest | yes | idempotent resend without re-execution, crash forensics, settlement evidence |
| Transport of inputs and results between Client Job and worker | no, once a session exists | a file is a slow, polled message; cellnet already streams large payloads, carries lazy tensor refs, and is the channel the Client API external trainer uses today |

## 2. Design

The worker opens a Cell as a child of the Client Job cell, exactly as
`ExternalProcessBackend` trainers do: bootstrap file with `connect_url` and
a launch token, HELLO, delegated job-scoped credentials, heartbeat, abort,
shutdown, Client Job PID watchdog, process-group termination. Over that
session:

- next-attempt delivery with the input streamed;
- per-attempt result streamed back and materialised in the Client Job before
  the worker may continue;
- analytics as live messages, no spool;
- the end signal.

On disk, per attempt: the checkpoint candidate, completion and failure
records, and the saved result the Client Job retains for resend. The Client
Job writes the saved result; the worker no longer needs the artifact root.

The group semantics in `task_groups.md` are written against a small
`WorkerSession` interface (`open`, `deliver(attempt)`,
`wait_completion(attempt)`, `end`, `alive()`); the disk session directory is
one implementation and the cellnet session is the other. `TaskSupervisor`,
`TaskWorkerExecutor`, the controller API and the headers do not change.

## 3. Consequences

- The worker is the executor-hosting twin of the external-process trainer:
  one session protocol serves Client API scripts and ordinary executors.
  `ExternalProcessBackend` is where the session is factored from, and that
  factoring is most of the work.
- The Client Job remains the only federation endpoint. The worker's cell
  talks to the Client Job at the application level; nothing is routed
  through it. Hop count is unchanged.
- Launchers need a network path from the worker to the Client Job listener:
  loopback for Process, in-cluster for pods and Slurm steps, the same class
  of requirement as the CP-to-CJ link. The artifact root needs to be
  reachable from the Client Job only.
- GPU task workers are a policy and launcher change, not a transport change:
  #5405 disables GPU visibility in the forwarded environment by policy.
  Lifting that is a separate decision that neither transport blocks.
- Future outbound calls from a worker (Collab) are session messages the
  Client Job authorises and dispatches; no second mechanism.

## 4. Open questions

1. Keep the disk session as a fallback for launchers without a network path,
   or make cellnet the only session transport once it lands? Recommendation:
   keep disk while any supported launcher lacks the path; drop it when none
   does.
2. Does the Client Job stream the result straight to the saved-result file,
   or materialise it in memory first? Recommendation: stream to file with
   the digest computed on the way, as the worker does today; the in-memory
   result is decoded from the verified file as today.
3. Should the worker reuse `nvflare.client.cell` (the Client API cell
   runtime) or a thinner session client? Recommendation: reuse; the HELLO,
   token, heartbeat and shutdown handling are already hardened there.
