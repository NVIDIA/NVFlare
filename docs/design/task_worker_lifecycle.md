# Task worker lifecycle and integration contract

P3 provides one stateless Process worker for an ordinary Executor. P4 adds a plain,
role-neutral TaskSupervisor that owns its execution and artifact lifetime, plus
the CPU Process client integration. The resident Client Job owns filters, analytics
publication, result submission and receipts; its TaskWorkerExecutor launches a fresh
compute worker for each assignment. Local completion, physical process settlement, and server receipt are
separate facts.

## Client configuration and script lifetime

Client configuration accepts `execution_lifetime: "job"` (the default) or `"task"`.
Job lifetime keeps the existing executors and Client API backends. Task lifetime
authorizes inert executor and component specifications in the Client Job and again
in the worker before importing application code. Compute instances are constructed
only in the worker. Task data and result filters always stay in the Client Job.

In task mode, each worker receives the executor and its referenced compute
dependencies, including transitive component IDs passed under arbitrary constructor
argument names or listed in `component_dependencies`. Other components, including
tracking receivers and federation widgets, remain resident in the Client Job
automatically. Ordinary dictionaries are preserved and nested compute helpers are
created with their owning executor. A component cannot be shared between the compute
graph and a resident service or task filter. There are no per-component lifetime
flags. Jobs cannot configure the framework's internal adapter, launcher,
environment policy or declared task state.

`FedJob(execution_lifetime="task")`, `job.set_execution_lifetime(...)`, and
`recipe.set_execution_lifetime(...)` expose these controls. The unified and PyTorch
FedAvg and PyTorch FedProx recipes also accept `execution_lifetime="task"`. Export
validates the receiving sites' GPU resource/launcher settings; unrelated job-lifetime
GPU apps in the same deployment remain valid.

The task ClientAPIExecutor supports `execution_mode="in_process"`. Its script
runs on the worker's main thread, receives one assigned model, and hands off one
eager result through the existing Client API model, format, FULL/DIFF, metadata and
rank semantics. The same receive/send loop exits after that send. Script arguments
retain their boundaries, and supported secret placeholders resolve in the worker.
Sending only hands off a local snapshot: script failure or finalizer failure still
prevents completion. Analytics are captured synchronously and replayed by the Client
Job after settlement; local worker handlers also receive those events.

Script process memory, optimizer state and executor instances do not survive the
assignment. Applications needing previous local model state must use a job-lifetime
script or manage their own explicit persistence; this profile supplies no state
declaration, promotion or aggregation admission mechanism. The hello-pt task example
supports training and final global evaluation. Its cross-site local-model submission
uses job-lifetime memory and therefore remains a job-lifetime option.

The initial profile supports CPU Process workers on Linux and on macOS with Python
3.13+ providing waitid/WNOWAIT. External/attach Client API sessions, GPU task workers,
remote task launchers, and moving task orchestration to the Client Process are not
part of this profile. No server capability negotiation is added.
The Collab API's resident RPC service continues to use job lifetime.

## Site policy

Only the site's canonical `local/resources.json` or its default fallback supplies
`task_execution`. Job application and job resource overrides cannot select it.
The default launcher is ProcessTaskLauncher, environment forwarding is explicit,
and artifact cleanup happens at job end. An optional site configuration is:

```json
{
  "task_execution": {
    "environment_variables": ["TRAINING_SECRET"],
    "worker_timeout": 600,
    "settlement_timeout": 10,
    "poll_interval": 0.05,
    "artifact_cleanup": "job"
  }
}
```

`artifact_cleanup: "retain"` keeps every attempt. `max_payload_bytes` optionally
limits input and output payloads with the same policy. The job's executor definition
can shorten `worker_timeout`, and ClientAPIExecutor's `result_wait_timeout` bounds
the wait from script start until its local send. A timely send does not waive the
worker deadline or physical settlement requirement.

Workers receive only the framework's standard runtime environment allow-list plus
site-approved variable names. Federation credentials, Client API bootstrap and GPU
visibility cannot be forwarded; GPU visibility is disabled. Process workers share
the site UID and filesystem, so this is process lifetime isolation, not a security
sandbox. The site policy is excluded from client registration metadata.

## Execution and diagnostics

The worker checks the original compute graph against site component policy before
application logging, custom imports, decomposers, or input decoding. It initializes
the job audit sink first, preserving allow-all and warn-mode authorization records.
The current builder repeats authorization during graph construction; these checks
are safe but can be consolidated in a later optimization.

START_RUN and END_RUN apply only to the selected attempt compute graph, including
nested FLComponents constructed for executor/component arguments. Each instance
receives lifecycle and task events once, with children registered before parents.
On a Python exception, both cleanup events are attempted and the first exception is
retained in failure.json. Later cleanup exceptions cannot overwrite it.
Wall-clock start/end timestamps are diagnostic and may move backward. Elapsed
time is measured with time.monotonic(). The worker has no SIGTERM handler: launcher cancellation can terminate it without
END_RUN. Executors must support this stateless one-shot cancellation contract;
END_RUN is not a guaranteed cleanup mechanism after process termination.

The guardian distinguishes worker exit from Client Job loss. While the Client Job
lives, the launcher owns descendant settling and shutdown deadlines. Client Job
loss triggers immediate cleanup of the owned process group.

## Payload storage

There is no fixed 4 GiB payload ceiling. FileTaskArtifactStore and WorkerBootstrap
accept an optional positive max_payload_bytes supplied by the trusted site/client
adapter. The adapter must apply the same policy to input staging, the bootstrap,
and result reading. None leaves capacity to the filesystem and its quota. The
separate bootstrap and metadata record limits remain in place.

After confirming writer and reader quiescence, release_payloads also removes
interrupted publication temporaries for its payload/metadata names. Full retention
cleanup can remove an incomplete attempt directory whose identity.json was never
published. An existing identity record must still be valid and match the caller.

FOBS serialization hashes each byte as it is written. The worker still verifies
the stored result before publishing completion, and the reader verifies it before
decoding. These verification passes detect changed or corrupt payloads; staging
no longer rereads the complete file merely to calculate its initial digest.

Trusted bootstrap directory aliases, including /tmp on macOS, are resolved before
opening. Bootstrap files themselves and attempt artifacts cannot be symlinks.
Directory ancestors require search permission; directories receiving durable
artifact writes also require read/write permission for the durability barrier.

## Completion and supervisor integration

The worker serializes completion publication with fatal, abort, and event-failure
reporting. A failure observed before the publication decision prevents completion.
An application callback still in progress also prevents it. Successful publication
closes the runtime to subsequent application events and abort requests.

A valid completion.json can coexist with a later nonzero process exit, such as
SIGKILL after publication, or with failure.json if the directory fsync fails after
the completion link is installed. The worker preserves both records for diagnosis.
A directory durability error means persistence across a host crash is uncertain.

P4 must validate completion identity, worker PID, payload size and digest, and
confirm physical settlement before consuming a result. A fully verified completion
is evidence of finished local computation even when a later process error or
post-publication failure diagnostic exists. Those later facts do not erase local
completion. TaskSupervisor returns the verified result and the observed execution
status independently, retaining payloads when execution failed or failure.json
exists. Missing or invalid completion still fails validation after settlement.

The supervisor has one job lifetime and admits one active physical attempt at a
time. Unconfirmed settlement permanently prevents further attempts on that
instance and retains its handle and artifacts. end_run closes admission and
cancels known execution. If launch is still acquiring a handle, the caller must
join its run call before discarding the supervisor; that call owns and cancels
the late handle. There is no reset or automatic physical retry.

For publication, the caller validates the reply's workflow, task, and attempt
before calling record_receipt. RETRY or unknown receipt retains the saved result.
The default job-end cleanup requires confirmed settlement, a validated successful
result, RECEIVED or TASK_CLOSED, and no outstanding transfer_reader leases. Each
actual transport reader must hold a lease until its final source read, including
after cancellation or receipt. Cleanup excludes new readers once release starts;
failed cleanup retains that exclusion while allowing cleanup retries. Explicit
retention and failed attempts keep their payloads. Completion and diagnostic
records remain after successful payload release. Closure says nothing about an
earlier upload's receipt or aggregation use.

A local completion never overrides Client Job/task cancellation or authorizes
network publication by itself. The adapter must drain pending launches during job
shutdown, recheck cancellation before sending, retain the saved result through
receipt retries, and keep artifact files until every actual transport reader has
finished. Sim/POC/Prod E2E tests must exercise these boundaries, including process
termination and filesystem failure after completion publication.
