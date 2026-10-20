# Communication accounting: reviewer guide

This local branch provides optional byte-accounting hooks for NVFlare application
messages and their tensors. It does **not** turn accounting on. A deployment
without the reporting integration follows its normal communication and shutdown
paths without creating counters or transfer-accounting ledgers.

## Branch boundaries

| Branch | Responsibility |
| --- | --- |
| `codex/resource-stats-communication-review` | Optional counters, F3 hooks, application-send classification, and external-trainer adapters. |
| `codex/resource-stats-final-integration` | Starts counters, collects resource measurements, drains counters before reporting, publishes reports, and implements storage and CLI queries. |

The communication branch is based on locally cached `origin/main`,
`94c5b0aebd3de9a6733926d5fc285144bfd335da`. Its worker startup, server runner,
JobRunner, configuration loader, and shared process-cleanup code match that base.
There is no new launcher command, Python startup mode, package dependency,
privilege, operator setting, monitoring service, or shutdown wait.

## Suggested review order

| Section | Read first | Review question |
| --- | --- | --- |
| Counter | [f3_counter.py](../../nvflare/private/fed/resource_stats/f3_counter.py) | Does an accepted logical send contribute once, and does closing fix its cutoff? |
| F3 hooks | [send_accounting.py](../../nvflare/fuel/f3/send_accounting.py), [core_cell.py](../../nvflare/fuel/f3/cellnet/core_cell.py), [cell.py](../../nvflare/fuel/f3/cellnet/cell.py) | Are observations optional, bounded, and independent of transport success? |
| Tensor downloads | [download_service.py](../../nvflare/fuel/f3/streaming/download_service.py), [file_downloader.py](../../nvflare/fuel/f3/streaming/file_downloader.py), [cacheable.py](../../nvflare/fuel/f3/streaming/cacheable.py) | Are actual chunks identified without retaining arbitrary request data or changing download lifetime? |
| Serialization and streams | [via_downloader.py](../../nvflare/fuel/utils/fobs/decomposers/via_downloader.py), [byte_streamer.py](../../nvflare/fuel/f3/streaming/byte_streamer.py) | Do source settlement, cancellation, and transport acceptance retain their existing meanings? |
| Application send sites | [server_command_agent.py](../../nvflare/private/fed/server/server_command_agent.py), [communicator.py](../../nvflare/private/fed/client/communicator.py), [admin.py](../../nvflare/private/fed/server/admin.py) | Are only the intended originating operations classified? |
| External trainers | [external_transfer.py](../../nvflare/private/fed/resource_stats/external_transfer.py), [bridge protocol](client_api_resource_transfer_bridge.md) | Are trainer bytes attributed to the known client and delegated credentials checked without changing delivery? |
| Dormant default | [passive_accounting_test.py](../../tests/unit_test/private/fed/resource_stats/passive_accounting_test.py) | Does normal startup and cleanup remain free of counter activation and statistics-specific waits? |

For a focused diff:

```shell
git diff 94c5b0aeb...HEAD -- nvflare/fuel/f3 nvflare/fuel/utils/fobs/decomposers/via_downloader.py
git diff 94c5b0aeb...HEAD -- nvflare/private/fed/resource_stats nvflare/client/cell
```

## What is counted when reporting enables accounting

| Operation | Origin | Destination |
| --- | --- | --- |
| Job application deployment | Server parent | Each client site |
| Task response carrying application data | Server job process | Task's client site |
| Task result carrying application data | Client job process | Server |

Bytes include serialized application payload and accepted tensor chunks belonging
to that operation. Tensor chunks add bytes, not logical messages. Retries and
repeated completion callbacks do not add the same chunk twice. Relays do not
create a second originating send.

These are sender-side observations after local transport acceptance, not proof
of receipt. Reporting derives each participant's inbound number from the other
participants' accepted sends. Protocol framing, encryption overhead, heartbeats,
admin control messages, workspace-result uploads, the resource report itself,
and same-process direct delivery are excluded.

Traffic classification is a process-local message attribute, not a remotely
supplied header. F3 does not choose NVFlare's traffic classes or participants;
the application send sites do.

## Bounded chunk accounting

A producer supplies a small, stable numeric chunk identity. Built-in file
downloads use the actual byte range; cacheable tensor downloads use the actual
item range. Arbitrary receiver-provided state is not copied into accounting keys.
A repeated identity with a different byte length marks an accounting gap rather
than silently discarding the difference.

Each logical operation retains at most 16,384 accounting entries across chunk
identities and delegated-consumer mappings. Source-only trainer accounting uses
the same limit. Once the budget is exhausted, accounting stops retaining new
metadata and marks the existing counter gap. A custom producer without a stable
identity also leaves accounting incomplete. **Downloads continue normally in
both cases.** When accounting is absent, the identity hook is not called.

Task-response accounting can reserve an observation before encoding. Encoding
or send-setup failure, or a response with no reply route, releases that unused
reservation. An already-started transport remains responsible for its own
completion. Accounting cleanup preserves the original transport exception.

## External trainers and authentication

A trainer can serve tensors directly to the server while the client job forwards
only references. The optional bridge joins those tensor bytes to the client
result through the existing source-settlement exchange.

For server task downloads, the server can associate a managed trainer with the
known client, job, and task after validating its existing delegated site token
and signature. Reporting integration supplies the parent site's existing public
certificate for this check. This foundation does not alter certificate setup.

Those credentials authenticate the site, not a particular trusted program within
it. Missing or invalid evidence leaves accounting incomplete. It does not reject
an otherwise valid download, admit an otherwise invalid training request,
change receiver-count requirements, or extend source lifetime. Reports contain
no credentials.

Without an active reporting counter, the trainer bridge is not negotiated and
does not create a source accumulator or validate accounting-specific proof.

## Lifecycle work belongs to reporting integration

The reporting branch starts child counters after normal workspace preparation
and before application execution. It also starts the server parent's per-job
deployment counter. The foundation alone does neither.

Before a child publishes a report, reporting integration waits for admitted
callbacks while transport is available, then closes the counter with a bounded
drain. The counter drain is at most five seconds and returns immediately when
settled. Callback waits use the existing five-second bound; if the pre-publication
wait expires, cleanup still stops transport and performs its bounded post-stop
wait. All three waits expiring can therefore take up to fifteen seconds.

Failed-client publication keeps its existing owner and timing. Its later deferred
cleanup has no report to publish: it stops transport before waiting for callbacks
and does not add a statistics-specific pre-stop wait. Unsettled observations are
marked incomplete. The server parent's final counter cutoff waits zero seconds.

A failed restore removes a newly created parent counter only if the job has not
entered the running-job table. Existing counters and counters already owned by
a running job are preserved. Final publication, report schemas, archive access,
and CLI rendering are reviewed with the reporting branch, not this foundation.

Shared [communication_limits.py](../../nvflare/private/fed/resource_stats/communication_limits.py)
holds the participant-name pattern and numeric bound. The reporting contract
imports these same constants; the foundation does not import that contract.

## Verification and remaining validation

Regression coverage includes passive startup and cleanup, bounded chunk and
consumer metadata, real file delivery after accounting saturation, unsupported
custom producers, conflicting retries, encoding/setup failures, and reporting
cleanup and restore ownership. The generic accounting implementation and
trainer bridge are kept identical in both branches.

Local verification on October 20, 2026:

| Check | Result |
| --- | --- |
| Full F3 and FOBS suites in the foundation | 1,125 passed, including 122 focused accounting regressions. |
| Foundation counter, trainer, application-send, and ordinary lifecycle tests | 646 passed; one Linux-only test skipped on macOS. |
| Full reporting resource, CLI, lifecycle, query, and API tests | 1,016 passed, including 92 reporting shutdown/restore tests. |
| Shared accounting regressions in reporting integration | 122 passed. |
| Formatting, lint, and whitespace | Scoped Black, isort, Flake8, and diff checks pass. |
| Reporting examples | All 24 golden artifacts reconcile; 15 JSON examples validate. |
| Documentation | 49 local Markdown links and 567 public-page link targets resolve. User-guide build adds no warnings against the previous full user-guide build. |

Earlier Colossus runs do not validate these latest changes. Live managed-trainer tests
with large tensors and concurrent-job timing still need qualification, as do
Docker, Kubernetes, Slurm, multi-node, and newer task-worker launch paths.
No new hardware-validation claim is made by this branch.
