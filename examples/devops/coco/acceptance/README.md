# NVFlare 2.9 CPU-only TDX acceptance

This harness prepares both topologies, provides an image-baked finite application,
observes the trusted federation, and keeps a private evidence ledger. Real TDX
appraisal, key release and guest-policy enforcement require hardware evidence.
Offline tests cannot satisfy those gates. No NVFlare public API changes are needed.

Select two protected TDX clients and trusted infrastructure for provisioning,
administration, and topology A's ordinary server. Use an independently operated
secure-services host (for example, `trustee.example.com`) for Trustee and an
HTTPS registry (for example, `registry.example.com`). Build and publish images
from trusted provisioning infrastructure directly to the registry; protected
guests contact secure services directly. Transfer private service handoffs from
the trusted provisioning host directly to the secure-services host over
authenticated transport. A separate operator workstation carries control traffic
only; keep image, credential, and key-release payloads off that workstation.
Record each endpoint's independently authenticated TLS trust anchor and the AS
signing public key separately. Changing endpoints or signing authorities requires
fresh provisioning and final release review; retagging an old encrypted image
does not update its key URLs or signed InitData.
The observer and audit hooks are optional acceptance-test instrumentation; they
are absent from the normal provisioning and federation workflow. This harness
requires the observer for its fresh cross-validation evidence.

Run A before B. Each acceptance topology has `site-1`, `site-2`, and an ordinary
`site-observer`; only the first two receive job tasks. B also protects logical
identity `server`, independently of the server's TLS hostname.
This acceptance run covers CPU-only TDX. It does not require a GPU or a subsequent
CPU+GPU phase.

## Trusted inputs and preparation

Pin the tested 2.9 commit and record any local patch hash. The reviewed base image
must contain that NVFlare version, Python 3.11+, Bash and all dependencies. Supply
an independently authenticated P-256 **AS signing public key** and its file SHA256;
the KBS TLS certificate is a separate trust anchor. Supply an approved MRTD and a
prepared CPU-only platform.env/launch contract. Do not create a baseline from an
untrusted cluster's current evidence.

Also prepare a unified `cc_project.yml` containing exactly one Trustee service,
one authenticated registry, and the trusted CoCo build command. Its credential
and certificate paths resolve from that file. The harness copies the common
settings into each private topology, pins the authenticated AS key supplied
below, and adds the topology's complete CPU/MRTD constraints.

Run on trusted provisioning infrastructure, with NVFlare installed from the pinned
checkout. Replace the shell variables with the actual reviewed inputs:

```bash
python examples/devops/coco/acceptance/prepare.py \
  --output "$PRIVATE_RUN" --run-id "$UNIQUE_RUN_ID" \
  --base-image "$REVIEWED_BASE_AT_SHA256" \
  --as-key "$AUTHENTICATED_AS_PUBLIC_KEY" --as-key-sha256 "$AS_KEY_FILE_SHA256" \
  --mrtd "$APPROVED_MRTD" --platform "$APPROVED_PLATFORM_ENV" \
  --server "$SERVER_DNS" --cc-project "$REVIEWED_CC_PROJECT_YAML"
```

The output directory is private and must be new. Separate project audiences,
releases and repositories are generated for A/B and every protected participant.
Every protected participant gets `cpu_tee: intel_tdx`, explicit `gpu_tee: none`, common
signed CPU/MRTD constraints, 120-second checks, 300-second proof lifetime and
registration budget, 30-second refresh budget and 45-second peer request timeout.
There is no circular InitData pin in the image; enforce final InitData in the
release-specific KBS policy. The common packager selects `kata-qemu-tdx`.

Provision each project on the trusted machine. Include the acceptance directory
on Python's import path so the observer builder runs **before SignatureBuilder**:

```bash
export PYTHONPATH="$CHECKOUT/examples/devops/coco/acceptance:$CHECKOUT"
nvflare provision -p "$PRIVATE_RUN/a/project.yaml"
# Run B only after A's positive chain and disruptive cases are restored/passed.
nvflare provision -p "$PRIVATE_RUN/b/project.yaml"
```

Existing CoCo publishing requires review/approval of its concrete release
artifacts. Preserve plaintext kits privately; verify protected-kit signatures before
startup. The pinned SignatureBuilder does not sign ordinary, non-CC kits. For the
trusted server, observer and admin, authenticate the provisioning snapshot and
verify its hashes and the provisioned TLS certificate chain before startup.
Publish only approved Pod YAML through the handoff mechanism. Inspect
each image for its own participant credentials and no other kit. The application
module is baked at `/local/custom/tdx_acceptance.py`. Install the same reviewed
module on topology A's ordinary server's trusted custom-code path. Do not change
signed kits after provisioning. The common CC builder installs verifier-only
settings, complete protected-site mapping, and peer binding on the ordinary
observer; the observer obtains no guest tokens. The observer builder only grants
the reviewed baked application classes on the ordinary server.
The startup patch gives well-formed discovery of missing required participants
a single 600-second registration window before periodic validation shuts down
the federation. The runner still enforces its 600-second deadline from the first
Pod apply. Malformed discovery, failed services and invalid proofs remain fatal.
Job scheduling requires successful validation of the complete required mapping,
and a participant lost after that validation causes shutdown without another
startup window.
The builder also adds only `tdx_acceptance.AcceptanceController` and
`tdx_acceptance.AcceptanceExecutor` to the server's enforced component allow-list
before signing. An ordinary server otherwise rejects the baked controller even
when participant attestation and registration pass. Wildcard authorization and
warning-only enforcement are rejected.

## Hardware and service gates

Before launch, verify BIOS, TDX-enabled kernel/KVM, Intel platform provisioning,
QGS, collateral retrieval, Kubernetes Ready nodes and `kata-qemu-tdx`. Neither
`/dev/kvm` nor RuntimeClass presence proves attestation. Verify authenticated TLS,
the AS signing-key provenance, pinned runtime/Trustee/verifier images, and
default-deny resource authorization on the secure-services host. A running Docker container alone
does not satisfy these checks.

Follow [TDX reference approval](../service/TDX-REFERENCE-VALUES.md) and
[the TDX launch profile](../trusted_system/TDX-LAUNCH-PROFILE.md). During trusted
rehearsal verify two independently challenged quotes: DCAP signatures,
unexpired collateral, `UpToDate` TCB, debug disabled, event-log replay, and the
complete approved MRTD/RTMR0–3/XFAM/kernel/kernel-parameter tuple. Install and
read back the complete tuple from RVPS. Reject mixed fields from separately
approved tuples. Rehearse the actual image and measured runtime's guest-local
`127.0.0.1:8006/aa/token?token_type=kbs` API; never export its private-key-bearing
response.

Use the existing approved release workflow and running-workload verification
scripts (see [runtime variants](../RUNTIME-VARIANTS.md) and
[running federation verification](../provision/VERIFY-RUNNING-FEDERATION.md)).
Verify encrypted digest, runtime, command, final InitData, UID/GID 65532, approved
writable guest rootfs, absence of host volumes, Ready status and zero unexpected
restarts in every fresh guest. For K1, run the service's
`13-verify-workload-release.sh RELEASE SINCE RELEASE_AUTHORIZATION.json` with the
CPU-only release authorization; omitting that argument can select SNP/GPU
defaults. Require attested access to the exact release's `security-policy`,
`sig-public-key`, and `image-key`. Trustee log matches are diagnostic evidence,
not cryptographic correlation across separate requests.

## Finite job and fresh observer evidence

Generate a new config-only job and nonce for each initial launch, cold relaunch,
restart and baseline-restoration run:

```bash
python examples/devops/coco/acceptance/application/generate_job.py \
  --output "$NEW_JOB_DIR" --nonce "$FRESH_NONCE"
python examples/devops/coco/acceptance/verify_federation.py \
  --admin-kit "$ADMIN_KIT" --username admin@example.com \
  --observer-local "$OBSERVER_KIT/local" --observer-log "$TRUSTED_OBSERVER_LOG" \
  --topology A --job "$NEW_JOB_DIR" --nonce "$FRESH_NONCE" \
  --download-dir "$PRIVATE_DOWNLOAD_DIR" --receipt "$NEW_PRIVATE_RECEIPT"
```

Use topology B for the second project. Start the ordinary observer with its
authenticated provisioning kit on trusted infrastructure. An admin console alone is insufficient.
The runner verifies its issuer-free configuration and exact registered membership,
observes two **new** periodic CCManager successes after opening the log at its
end, submits only the reviewed config-only job, and verifies the downloaded
nonce/client/value/aggregate artifact. It rejects stale/replaced/truncated logs,
unexpected members, executable BYOC content and unsuccessful jobs.
Immediately before submission, the runner rechecks the original reviewed JSON
source and its digest. Submission may add NVFlare job metadata; post-submission
checks use the recorded source digest and downloaded result rather than treating
those expected additions as a new submitted application.
Registration, initial two-round observation and job completion each have a
600-second bound. The default soak is 900 seconds and requires continuing new
validation rounds within 600-second windows. `--soak 0` explicitly yields partial
evidence. Proof renewal may reuse a cached EAR; periodic validation does not prove
a new hardware quote. For F4, enable `trusted_proof_audit.py` only in the ordinary,
issuer-free observer process before it starts. The hook calls the original
`CoCoAuthorizer.verify_for_site` unchanged and records sanitized metadata only
when it returns `True`: site, observation time, proof/EAR issue and expiry times,
and SHA256 fingerprints. It saves no JWT, JTI, key, or annotated evidence. Audit
IO/schema failures latch an unhealthy session while preserving verifier results;
a missing or unhealthy completion footer cannot qualify renewal.

For failure diagnosis, `trusted_channel_audit.py` records bounded metadata from
`Cell._send_request` in a fresh ordinary server or observer process. CC requests
use this streaming method, so a recorder on `CoreCell.send_request` misses them.
Pin the helper and Cell source before installation. The recorder preserves the
request, response and exception behavior and emits only known return codes,
timings, fixed error categories and reply-shape checks. It exports no proof or
error text. These observations do not establish appraisal or authorization, and
a healthy recorder footer does not establish a successful 900-second run.

On trusted infrastructure, make a new account-owned mode-0700 directory for each
observer launch, with a mode-0600 `sitecustomize.py` containing this reviewed
process-scoped hook (substitute the absolute local checkout and private output
paths; add `server` to the list for topology B):

```python
import os
import sys

path = os.environ.pop("NVFLARE_TRUSTED_PROOF_AUDIT", None)
if path:
    sys.path.insert(0, "/absolute/NVFlare/examples/devops/coco/acceptance")
    from trusted_proof_audit import install
    session = install(path, ["site-1", "site-2"])
```

Set `PYTHONPATH` to this private directory and `NVFLARE_TRUSTED_PROOF_AUDIT` to a
new JSONL file inside it **only for the observer Python entrypoint**. Do not apply
this environment to provisioning, admin, protected participants, or an entire
shell session. Confirm the `audit_start` header exists before starting the
900-second observation. The environment flag is consumed once so subprocesses
do not install another audit. Use the unchanged authenticated kit and stop the observer
gracefully after the complete observation so `atexit` writes its footer; forced
termination yields incomplete evidence. Do not edit provisioned kits to install it.

```bash
python examples/devops/coco/acceptance/trusted_proof_audit.py \
  --audit /private/observer-run/proofs.jsonl --expected-sites site-1 site-2
```

The helper requires a completed session of at least 900 seconds, distinct proof
issue times/fingerprints spanning each identity's original 300-second expiry,
and at least one distinct valid EAR with a later authenticated issuance time for
every required protected identity. Different signatures or encodings with the
same issuance time, and cycles of older EARs, do not establish refresh.
EAR refresh timing relative to the previous expiry is diagnostic: a cached EAR
can be replaced after its expiry, provided the original authorizer successfully
verifies the new evidence. Keep this report together with the runner's continuing
CCManager-round and successful-job receipts; metadata alone does not complete
F4. Proof or EAR refresh never establishes a fresh hardware quote.

Cold relaunch uses new Pod/guest identities and a fresh nonce/job. Repeat all
appraisal/resource/registration/job gates. For R1, record approved reference and
policy hashes before and after a controlled RVPS/KBS restart, read them back, then
repeat a fresh guest launch and job. C1 requires zero protected application-log
bytes and exec/attach denial at the guest-policy boundary; RBAC or transport
failure is not a pass. A successful attach that returns zero bytes does not prove
denial. Do not download protected logs to the host.

## Isolated denial cases and restoration

Keep baseline references/policies unchanged. Use new isolated releases and
reviewed separately provisioned fixtures when launch-time policy or signed
evidence must change. Never approve hostile evidence just to start a case.

| Group | Inputs | Required observation |
|---|---|---|
| platform | Corrupt quote/log, wrong challenge, debug, bad TCB, unapproved or mixed tuple | DCAP verification or appraisal denial |
| kbs | Wrong InitData/MRCONFIGID, CPU type, resource path, cross-release key | Denial at the relevant exact resource boundary |
| guest | Unsigned/corrupt image, command/context/policy changes, host mount, prohibited process | Actual guest/runtime enforcement; launcher-only rejection is insufficient |
| authorizer | Wrong audience/site/signature/key, expiry/replay, missing CPU appraisal, trust vectors, namespace omissions/extras/duplicates | Cryptographic/unit coverage plus reviewed hardware fixtures where needed |
| federation | Missing server proof, omitted client, required participant lost after registration | Registration denial or implemented shutdown/client exit |
| availability | Temporary and persistent token-service failure | Bounded recovery or termination; measure cached EAR expiry and timeout delays |
| image-binding | Other accepted-key signed image in same repository, unchanged InitData | Characterize actual behavior; exact-effective-image binding remains open |

Prepare reviewed Pod candidates on the trusted host with Python 3.11+ and
PyYAML, after the signed participant handoff and its Pod digest have been
authenticated:

```bash
python examples/devops/coco/acceptance/negative_fixtures.py \
  --pod "$REVIEWED_RELEASE/pod.yaml" \
  --baseline-sha256 "$AUTHENTICATED_POD_SHA256" \
  --output "$PRIVATE_FIXTURES" --run-id negative-run-a \
  --topology A --participant site-1 --namespace tdx-negative-run-a \
  --authority-note "Digest verified from signed participant provisioning handoff"
```

The new private directory contains a control Pod and four candidates: changed
command, UID/GID zero with non-root enforcement disabled, a read-only `/proc`
host mount, and changed measured InitData. The InitData variant appends a valid
TOML comment, changing the raw release-bound digest while preserving embedded
policies. Images and baseline authorization remain unchanged. The manifest
records baseline/candidate hashes, changed field paths, source Pod UID and all
fixture UID/GID contexts; actual fixture Pod UIDs are recorded after an operator
launches them. This tool checks the supplied digest, not the signature's trust
chain; authenticate the digest through the signed provisioning handoff first.

No candidate is applied, no namespace is created, and no policy or approval is
written. Operators review each candidate and the isolated namespace before
launching it under the authorized untrusted role. First establish the control
guest and a successful fresh-nonce job so a namespace or identity change cannot
be mistaken for a targeted denial. Record the actual Pod UID, timestamps,
sanitized events, and enforced layer for each candidate. A launcher, RBAC,
network, or syntax rejection does not establish guest-policy or KBS denial;
leave the intended hardware case unqualified until its enforcement layer is
observed. The preparer provides no launcher bypass or broader authorization.

Protocol proof faults reuse the authorizer/CCManager cryptographic unit suites
for offline coverage. Hardware audience/site/signature/key/expiry/namespace
faults still require separately reviewed baked participant fixtures. Unsigned
or corrupted image cases and same-repository accepted-key image substitution
also require separately reviewed images; Pod candidate preparation does not
create those images or close the exact-image-binding gap.

Restore the baseline and prove another successful fresh-nonce job after each
disruptive case. Keep exact inputs, enforced layer, timestamps, outcome and
restoration receipt privately. Failures and unexecuted fixtures stay visible in
the ledger. Prepared candidates remain `NOT_RUN` until actual evidence is
recorded. Delete only the fixture Pods created by the operator, and preserve the
approved baseline and all existing evidence.

## Evidence and regression prerequisites

```bash
python examples/devops/coco/acceptance/evidence.py init \
  --manifest "$PRIVATE_MANIFEST" --run-id "$UNIQUE_RUN_ID" --git-commit "$TESTED_COMMIT" \
  --revisions-file "$SANITIZED_REVISIONS_JSON" --hosts-file "$SANITIZED_HOSTS_JSON"
python examples/devops/coco/acceptance/evidence.py record \
  --manifest "$PRIVATE_MANIFEST" --topology A --scope hardware \
  --case F3 --status PASS --evidence "$NEW_PRIVATE_RECEIPT"
python examples/devops/coco/acceptance/evidence.py assess --manifest "$PRIVATE_MANIFEST"
```

Record P1/P2/H1/H2/K1/F1/F2/F3/F4/C1/R1 and all denial groups for both topologies.
Include hardware identity, reference provenance, revisions, release/Pod/policy
hashes, timestamped quote-verification results, sanitized KBS observations,
observer events, job IDs/results and restoration runs. The private ledger hashes
referenced files and invalidates modified evidence. It records operator assertions;
it cannot cryptographically validate arbitrary attached claim content. `offline`
records never satisfy hardware gates. Assessment exits 2 unless full security
qualification passes. Exact-image-binding starts `OPEN`; characterization alone
cannot close the documented security gap.

Run regression prerequisites on Linux with the scripts' GNU tools/Bash requirements:

```bash
python -m pytest -q tests/unit_test/app_opt/confidential_computing/coco_authorizer_test.py \
  tests/unit_test/app_opt/confidential_computing/cc_manager_test.py \
  tests/unit_test/lighter/cc_provision/impl
```

Report separate TDX appraisal, resource release, peer authorization, application
execution and confidentiality outcomes. Functional acceptance requires both full
positive chains; full security qualification additionally requires every denial
gate and closure of exact-image binding. GPU coverage is outside this run.
