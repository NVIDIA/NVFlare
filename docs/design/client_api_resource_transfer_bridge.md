# Optional Client API resource-transfer bridge

The managed external trainer and its client job negotiate a separately
versioned accounting extension. This does not change the required Client API
v1 handshake, topic values, existing payload-key values, result acceptance,
source lifetime, or shutdown policy. `PROTOCOL_VERSION` remains 1. Existing
v1 handlers accept additional dictionary keys; the earlier optional delegated
security fields use the same additive compatibility policy.

## Negotiation

The trainer includes `resource_transfer_version: 1` in its existing `HELLO`.
The managed backend includes the same field in `HELLO_ACCEPTED` only when
job message accounting is active and the trainer offered the supported
extension. The value must be an integer, not a Boolean or string. It is fixed
for the accepted session, including duplicate handshakes.

An older peer, disabled accounting, or an unsupported extension version
leaves the bridge disabled without rejecting the normal trainer handshake.
Attach/IPC and unmanaged trainers are not covered by this extension. They
require separate adapters before their tensor contributions can be claimed
complete. An inline result has no external tensor contribution to import.

For a negotiated session, `TASK_READY` also carries
`resource_transfer_recipients`: the final destination identities already
resolved by the managed backend for this task. This is accounting metadata,
not a new download-lifetime rule. The trainer validates a nonempty list of at
most 256 unique ASCII Cell identities, each at most 512 characters. Missing or
malformed metadata is ignored for training. If the result contains external
tensors, missing destination evidence leaves its accounting incomplete; a
later task cannot reuse an earlier task's destinations.

The trainer passes these identities only to its source accumulator. It does
not add a `RECEIVER_IDS` model header or change the existing `receiver_ids` and
`num_receivers` transfer arguments. A count-based transfer can keep
`receiver_ids=None` while its accounting accumulator knows the final server
destination. Old peers ignore the optional field without changing behavior.

## One summary at the existing source boundary

For a negotiated managed session, the trainer creates one source accumulator
for each result. Serialization registers real source reference IDs and final
receivers. Existing transport hooks observe accepted tensor chunks; they do
not increment public message counts for internal trainer-to-client forwarding.

After the existing transfer waiters succeed, the trainer takes a frozen
snapshot. Its existing `RESULT_SOURCE_SETTLED` request carries an optional
`resource_transfer` block:

```json
{
  "version": 1,
  "groups": [
    {
      "source_ref": "ref-a",
      "recipient_fqcn": "server.job-1",
      "accepted_payload_bytes": "1048576",
      "complete": true
    }
  ]
}
```

These reference and process identities are private correlation data, not
new participant keys or persisted report fields. The client job determines
the public participant and traffic class from its runtime state. The block
is bounded to 64 KiB of canonical compact JSON and at most 256 groups. Missing,
oversized, unsupported, or malformed data is an accounting gap, not zero.
Repeated valid contributions do not add bytes again; conflicting contributions
remain incomplete. The source summary is a site self-report, not signed proof.

## Chunk identities and memory bounds

Byte accounting identifies chunks using the producer's actual progress:
files use the starting and ending byte offsets; cached objects use the first
item's index and the number of produced items. Extra request fields are not
copied into accounting keys. A custom `Downloadable` can supply its own stable
key through `accounting_chunk_key(current_state, next_state)`: one to four
unsigned 64-bit integers. Without a valid key, its data is still served, but
its accounting remains partial. Retrying a key does not add bytes twice; a
changed byte length for the same key makes accounting partial.

Each logical send keeps at most 16,384 accounting entries across chunk keys
and delegated-consumer mappings. Each trainer-result accumulator has the same
chunk-key limit across its source references. Consumer names cached for
accounting are limited to 512 characters. If more entries are needed,
accounting becomes partial and stops retaining new chunk and consumer
metadata; downloading and source release continue normally. These in-memory
limits are separate from the 64 KiB,
256-group summary-block limits above. No chunk-key hook runs when accounting
is disabled.

## Settlement stays independent

The backend first performs its existing outer origin/session/task checks.
It then attempts optional accounting before retiring the known source
references. Whether that attempt succeeds or fails, a valid settlement still
performs the existing source-liveness transition and receives the existing
acknowledgement. A stale outer task/session remains rejected under the existing
protocol policy; accounting does not reopen it.

There is no new request, wait, timer, retry, filesystem operation, or process
cleanup rule. The existing settlement request retains its original retry
budget. The summary is snapped once and reused for those attempts. If the
existing transfer barrier fails, no complete contribution is claimed. Late
summaries cannot reopen a frozen job counter or rewrite an archive.

## Implementation and validation boundary

- [Protocol constants](../../nvflare/client/cell/defs.py): existing v1 values
  plus the separate optional extension and fixed bounds.
- [Trainer API](../../nvflare/client/cell/api.py): capability offer, per-result
  source accumulator, post-barrier snapshot, and existing settlement exchange.
- [Shared backend callbacks](../../nvflare/app_common/executors/client_api/cell_backend.py):
  unchanged outer validation and best-effort optional settlement hook.
- [Managed backend](../../nvflare/app_common/executors/client_api/external_process_backend.py):
  negotiated session, known accepted result references, and private manager calls.
- [Private contribution join](../../nvflare/private/fed/resource_stats/external_transfer.py):
  bounded validation, source/recipient association, deduplication, and cutoff.

Controlled tests cover compatibility, disabled and unsupported capabilities,
barrier ordering, per-result isolation, inline results, size/snapshot failures,
outer-identity rejection, and acknowledgement despite optional accounting
failure. They also cover resolved destinations without a lifetime header,
accepted tensor bytes on a count-based transfer, and task-scoped reset after
missing or malformed optional metadata. Live large-tensor qualification is
tracked separately; controlled tests alone do not prove every topology.
For server-side tasks, the secure managed trainer carries the site's existing
delegated name, token, and signature through the relay. At `GET_TASK`, the
private resolver verifies that tuple using the provisioned parent server's
public certificate, checks the registered client, and binds its current job.
The server job's different TLS certificate is not the token-signing identity.
The configured parent public certificate is retained before job TLS replacement;
no parent private key or new configuration is required.

While accounting is active, every descendant pull, retry, and EOF request must
carry that exact tuple and match the current client registration and known
job/source context. Signature verification happens once at task binding, not
per chunk. A cached consumer mapping cannot replace this check. The first
failed check stops accounting for that logical send; subsequent downloads
continue without trying to restore it. Direct authenticated TLS remains an
independent alternative. Failure leaves accounting partial and does not alter
download authorization, receiver settings, completion, or source release.

This uses the existing site trust boundary: a bearer tuple proves site
delegation, not independent trainer identity or malicious-site-proof statistics.
Credentials and private source/process identities stay out of public reports.
Forwarded downloads without either accepted proof remain deliberately partial.
The [binding tests](../../tests/unit_test/private/fed/resource_stats/external_transfer_test.py)
cover valid, missing, invalid, and changed proof; the
[integration tests](../../tests/unit_test/private/fed/resource_stats/external_process_accounting_test.py)
exercise the managed source contribution. Live large-tensor jobs and fault
recovery on the final branch remain separate qualification work.
