# Task worker lifecycle and integration contract

P3 provides one stateless Process worker for an ordinary Executor. P4 connects
that worker to the Client Job supervisor and the existing result submission path.
Local completion, physical process settlement, and server receipt are separate
facts.

## Execution and diagnostics

The worker checks the original compute graph against site component policy before
application logging, custom imports, decomposers, or input decoding. It initializes
the job audit sink first, preserving allow-all and warn-mode authorization records.
The current builder repeats authorization during graph construction; these checks
are safe but can be consolidated in a later optimization.

START_RUN and END_RUN apply only to the selected attempt compute graph. On a
Python exception, both cleanup events are attempted and the first exception is
retained in failure.json. Later cleanup exceptions cannot overwrite it. The
worker has no SIGTERM handler: launcher cancellation can terminate it without
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
completion. The current standalone supervisor requires successful process status;
the integration PR must implement and test the intended precedence explicitly.

A local completion never overrides Client Job/task cancellation or authorizes
network publication by itself. The adapter must drain pending launches during job
shutdown, recheck cancellation before sending, retain the saved result through
receipt retries, and keep artifact files until every actual transport reader has
finished. Sim/POC/Prod E2E tests must exercise these boundaries, including process
termination and filesystem failure after completion publication.
