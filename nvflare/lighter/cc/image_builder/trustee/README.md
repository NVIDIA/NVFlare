# Trustee deployment boundary

For the complete installation procedure, sample configuration, certificates,
bundle enablement, and key operations, see [TRUSTEE_GUIDE.md](../TRUSTEE_GUIDE.md).

The systemd units are deployment templates, not automatically installed by either
builder. They run the patched compatibility baseline from the main README with a
dedicated `cvm-trustee` account. Install Python modules under `/opt/cvm-builder`
and the compiled KBS under `/opt/cvm-trustee/bin/kbs`.
Include `scripts/trustee_preflight.py` under `/opt/cvm-builder`; the KBS unit uses
it to reject host swap and piped core collectors before loading keys. Configure
the host's crash-collection policy alongside its swap policy.

Create the directories named in the units, including
`/var/lib/cvm-trustee-resources/default`, owned by `cvm-trustee` with mode 0700.
Keep `/var/lib/cvm-trustee-revocations` outside restorable resource backups.
Both processes read the same 0600 resource files; KBS's mount namespace makes the
resource repository read-only. Only the key service and reconciliation unit can
write it. Reconciliation runs before KBS starts. Stop KBS before restoring any
resource backup; restoring revocation state to an earlier point is forbidden.

Supply `/etc/cvm-trustee/key-service.json` with the schema described in the main
README. Use `/var/lib/cvm-trustee-resources` for `resources` and
`/var/lib/cvm-trustee-revocations` for `state`. Initialize `approved-bundles.json`
there with an empty `build_ids` array. Set `CVM_AS_POLICY_ID` in
`/etc/cvm-trustee/environment` to the approved, versioned AS policy ID.
The KBS unit creates a writable `/var/cache/cvm-trustee` and sets
`XDG_CACHE_HOME` for Intel DCAP collateral caching. Configure the deployment's
collateral services and network access explicitly; a read-only home directory
must not prevent the verifier from maintaining its cache. Cached collateral still
undergoes normal signature, expiry and TCB validation.
The service account also needs read access to its protected DCAP/QCNL
configuration, including any collateral-service credential. Keep that file
read-only and do not put credentials into images or logs.

In KBS configuration, use HTTPS, authenticated administrative APIs, and trusted
AS signing certificates from a dedicated AS root; transport/client roots must
never be trusted as attestation signers. Point the LocalFs resource plugin at the resource
repository. Point the EAR broker's `policy_dir` at
`/etc/cvm-trustee/as-policies`; preinstall both its deny-by-default
`opa/default_cpu.rego` and the reviewed `opa/<policy-id>_cpu.rego`. Set the resource
policy path under `/var/lib/cvm-trustee/policy/`, RVPS LocalJson storage under
`/var/lib/cvm-trustee/rvps/`, and AS working state under
`/var/lib/cvm-trustee/as/`. The AS policy files remain read-only in the KBS process.

These templates establish the filesystem boundary. Validate them with the exact
deployment's authentication, policy-selection, mutation-denial, durable-upload,
revocation, restore and retirement tests before writing a deployment receipt.
The lab helper deliberately uses separate disposable paths and does not install
these units or modify another KBS service.

GPU profiles also install the immutable `opa/<policy-id>_gpu.rego`, generated from
the measured profile policy, and record both policy digests. Configure
`CVM_NVIDIA_CONFIG` for backend NRAS verification with a pinned JWKS snapshot;
see the main Trustee guide. Each backend instance serves one exact security
profile and imports TCB and GPU driver/VBIOS approvals without unioning sets or
renewing existing expiry dates.
