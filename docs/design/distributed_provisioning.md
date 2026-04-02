# FLARE Distributed Provisioning — Manual Workflow

Created Date: 2026-03-27
Updated Date: 2026-04-01

---

## Assumptions and Non-Goals

### Trust Model (Non-CC Deployments)

This document applies to standard (non-Confidential Computing) deployments. The
following assumptions define the security model:

1. **Trusted local site admin** — the site admin who runs `nvflare package` and
   `start.sh` is trusted not to modify their own startup kit maliciously. Startup kit
   integrity (`signature.json`) is a CC and HE concern, not a general-purpose tamper check
   for local config edits.

2. **PKI/mTLS as the non-CC trust anchor** — `rootCA.pem` + the X.509 cert chain is
   the trust anchor for all non-CC deployments. A site is "authorized" if and only if
   its mTLS cert chains to the project root CA. Replacing `rootCA.pem` locally would
   cause mTLS to fail immediately.

3. **Local configuration customization is allowed** — site admins may adjust local
   config (e.g. resource limits, log levels) without triggering an integrity failure.
   This is intentional: `signature.json` enforcement in non-CC would prevent any local
   customization.

4. **Attestation-chain integrity remains CC-only** — the CVM boot-time attestation
   sequence (`signature.json` → CVM measurement → remote attestation) is not in scope
   here and remains unchanged.

### Non-Goals

- **Confidential Computing** — CC deployments are centralized by design; see §CC Deployments.
- **HE with distributed provisioning** — HE requires centralized provisioning (Option C);
  incremental site onboarding for HE federations (Option B, asymmetric HE) is a future
  release item; see §HE Deployments.
- **Hierarchical FL (relay nodes)** — relay node ownership is ambiguous in a distributed
  model (unclear which site admin provisions the relay, and which root CA it belongs to).
  `nvflare package` does not support relay participant types. Use `nvflare provision` for
  relay topologies.
- **Multi-root-CA federations** — all participants share one root CA per project.
- **Key rotation** — out of scope for this iteration.
- **Automated CSR exchange** — the Manual Workflow is intentionally out-of-band
  (email, USB, secure file transfer). Automating the exchange channel is a future concern.

---

## Problem Statement

Current FLARE provisioning (`nvflare provision`) is centralized: the Project Admin must
collect all participant information upfront, generate startup kits containing each site's
**private key**, and distribute full kits over a secure channel. Adding a new site after
initial provisioning requires dynamic provisioning process.

Three specific problems:

1. **Private keys in transit** — keys are generated centrally and must be sent to each site
2. **Centralized gathering** — Project Admin must collect all participant details before any
   kit can be generated
3. **Static roster** — adding one new site requires re-provisioning or manual dynamic-provision

---

## Manual Workflow (Distributed Provisioning)

The Manual Workflow eliminates all three problems without requiring any new infrastructure.
Private keys are always generated locally. The only things exchanged are:

- **Site → Project Admin**: CSR (Certificate Signing Request) — public key only, no private key
- **Project Admin → Site**: signed certificate + `rootCA.pem` + server URI

The Project Admin holds the root CA key on their own machine. CSR signing is a local
`nvflare cert sign` command. Exchange is out-of-band (email, USB, secure file transfer).

### Step-by-Step

| Step | Who | Action |
|------|-----|--------|
| 1 | Site Admin | `nvflare cert csr -n hospital-1 -t client -o ./csr` |
| 2 | Site Admin | Send `hospital-1.csr` to Project Admin via email/secure channel |
| 3 | Project Admin | `nvflare cert init -n "Project" -o ./ca` (one-time — skip if CA already exists) |
| 4 | Project Admin | `nvflare cert sign -r ./hospital-1.csr -c ./ca -o ./signed -t client` |
| 5 | Project Admin | Send back: `hospital-1.crt` + `rootCA.pem` + server URI |
| 6 | Site Admin | `nvflare package -n hospital-1 -e grpc://server:8002 -t client --cert ./signed/hospital-1.crt --key ./csr/hospital-1.key --rootca ./signed/rootCA.pem` |
| 7 | Site Admin | `cd hospital-1 && ./startup/start.sh` |

**What makes this different from current provisioning:**

- No infrastructure to deploy
- No tokens, no service
- Private key never leaves the site
- Project Admin does not need all participants upfront — each CSR is signed on demand
- Startup kit is assembled locally by the site admin, not received as a black box

### Variant: Multiple Participants and Custom Builders

When a site needs kits for more than one participant (e.g. a client process plus several
admin users), or when the federation uses custom provisioning builders, the site admin uses
a site-scoped project YAML:

```bash
# Same CSR step for each participant
nvflare cert csr -n hospital-1 -t client -o ./certs
nvflare cert csr -n admin@hospital.com -t lead -o ./certs

# After receiving signed certs + rootCA.pem from Project Admin,
# drop all into ./certs/ (named by CN: hospital-1.crt, admin@hospital.com.crt, rootCA.pem)

# Assemble all site kits in one command
nvflare package -e grpc://fl-server:8002 -p site.yaml --dir ./certs
```

The `site.yaml` lists participants and any custom builders (schema-compatible with
`nvflare provision` project.yaml). The `-t` flag optionally filters by participant type.
See `impl_plan/04_package.md §2b` for full details.

---

## Signature Validation Analysis

The v2 design document omitted analysis of FLARE's existing signature validation system.
This section provides that analysis and determines what needs to change.

### Two Distinct Signature Systems in FLARE

FLARE currently has two separate signature mechanisms that serve different purposes:

#### System 1: Startup Kit Integrity Signature (`signature.json`)

**Where:** `startup/signature.json` and `local/signature.json` in each startup kit

**Who signs:** Root CA private key, during `nvflare provision`

**What is signed:** All files in the `startup/` and `local/` directories

**How it works:**
- `SignatureBuilder` (`lighter/impl/signature.py`) runs during provisioning and calls
  `sign_folders()` with the root CA private key
- At runtime, `SecurityContentManager` (`fuel/sec/security_content_service.py`) loads
  `signature.json` and verifies each config file it reads against the root CA public key
- The `AdminAPI` (`fuel/hci/client/config.py`) uses `SecurityContentManager` to load
  `fed_client.json`, `fed_server.json`, authorization config, etc.

**Purpose:** Detect tampering with startup kit configuration files after distribution.
If an attacker modifies `fed_client.json` post-distribution, the signature check fails.

---

#### System 2: Job Submission Signature (`__nvfl_sig.json` + `.__nvfl_submitter.crt`)

**Where:** Inside each folder of the submitted job directory

**Who signs:** The **admin user's private key** (from their startup kit), at submission time

**What is signed:** All job files in each subfolder of the job

**How it works:**
- `push_folder` command (`fuel/hci/client/file_transfer.py:310`) calls `sign_folders(full_path, private_key, api.client_cert)` before zipping and uploading the job
- `sign_folders()` with a `crt_path` embeds `.__nvfl_submitter.crt` (the user's certificate) in each folder
- Server verifies via `verify_folder_signature(app_path, root_ca_path)` in `job_runner.py:170`
- Client verifies the same in `training_cmds.py:113` when it receives the deployed app
- Verification checks: (a) submitter cert chains to root CA, (b) all files match signatures

**Purpose:** Ensure only authorized users (cert signed by project root CA) can submit jobs,
and that job files are not tampered in transit from admin to server/clients.

---

### Analysis: Startup Kit Signature in the Manual Workflow

**Finding: Meaningful only in CC mode (attestation chain) and HE mode (shared context integrity). Redundant in all plain non-CC, non-HE deployments.**

The startup kit in the Manual Workflow contains:

| File | Origin |
|------|--------|
| `client.key` | Generated **locally** by site admin |
| `client.crt` | Received from Project Admin (signed cert) |
| `rootCA.pem` | Received from Project Admin |
| `fed_client.json` | Generated **locally** by `nvflare package` (contains server URI + client name) |
| `start.sh` | Generated **locally** by `nvflare package` |

Only `rootCA.pem` and the server URI (in `fed_client.json`) come from outside. Everything
else is locally generated. There is no centrally assembled kit distributed over an
untrusted channel — the premise that made `signature.json` meaningful in the old flow.

Tampering with the two externally-received values:

| Tampered | Consequence | Caught without signatures? |
|----------|-------------|---------------------------|
| `rootCA.pem` replaced | Client trusts wrong CA | Yes — mTLS to legitimate server fails |
| `fed_client.json` server URI | Client connects to wrong host | Yes — mTLS cert mismatch |

Both are caught by mTLS before any FL operation proceeds. This holds whether the kit
was centrally provisioned or assembled via the Manual Workflow — mTLS is the trust
anchor in both cases. Startup kit signature verification adds no meaningful security
for any plain non-CC, non-HE deployment.

The existing code in `lighter/impl/signature.py` already branches on `CC_ENABLED`, but
still generates `signature.json` for **both** CC and non-CC kits — just with different
scope:

```python
if p.get_prop(PropKey.CC_ENABLED):
    # CC mode: sign from the root — full startup kit verified before CVM launch
    sign_folders(ctx.get_ws_dir(p), root_pri_key, ...)
else:
    # Non-CC mode: sign only startup and local (redundant — mTLS is the trust anchor)
    sign_folders(ctx.get_kit_dir(p), root_pri_key, ...)
```

Because `signature.json` is generated for non-CC kits, `SecurityContentManager.valid_config=True`
for all centrally provisioned kits regardless of CC mode. This causes the startup integrity
check in `fed_utils.py` to run for non-CC deployments (redundant) and to abort for the
Manual Workflow (false positive — nothing was tampered, there is simply no centrally
assembled kit to sign).

In CC (Confidential Computing) mode, the signature is part of the **attestation chain**:
the CVM verifies the startup kit before launch, before any network connection is possible.
mTLS is not yet available at that point — the signature is the only integrity check.
This is where `signature.json` is genuinely load-bearing.

**Decision:**
- **All plain non-CC, non-HE deployments** (centralized or distributed): `signature.json`
  must not be generated. The provisioner fix (see §Step 1) stops generating it for non-CC,
  non-HE kits. For FL runtime sites (server and client kits), `signature.json` presence
  reliably indicates CC or HE mode. (Admin user kits may vary by implementation — the
  signal applies to server/client kits specifically.) mTLS is the trust anchor.
- **CC deployments**: Distributed provisioning does not apply. The full attestation chain
  (including `signature.json`) continues unchanged. See §CC Deployments.
- **HE deployments**: Centralized provisioning required (Option C). `signature.json` is
  generated to protect the shared TenSEAL context. See §HE Deployments.

---

### Analysis: Job Submission Signature in the Manual Workflow

**Finding: Must be preserved. No changes needed.**

The job submission signature (System 2) works independently of how startup kits were
provisioned. It depends only on:

1. The submitting admin has a **private key** — generated locally in the Manual Workflow
2. The submitting admin has a **certificate signed by the root CA** — received from the
   Project Admin in Step 4 of the workflow

Both are present after the Manual Workflow completes. The `nvflare job submit` /
`push_folder` path calls `sign_folders()` with the admin's private key, producing
`__nvfl_sig.json` + `.__nvfl_submitter.crt` in each job folder — exactly the same as
today. The server and clients verify the submitter cert chains to `rootCA.pem` — which
it does, because it was signed by the same root CA.

**The Security Policy (SP) enforcement is fully preserved:**
- Server: `job_runner.py:167-173` — `verify_folder_signature()` runs for every job app
- Clients: `training_cmds.py:113` — same check before executing a task
- The check verifies: (a) submitter holds a key whose cert chains to root CA, (b) files
  were not modified after signing

**The only requirement for the Manual Workflow:** the admin user must be provisioned with
a certificate of type `admin` (or `admin`) signed by the project root CA, and they must
have their private key at the path `AdminAPI` expects (`client.key` in startup kit).
This is satisfied by the Manual Workflow — the user's CSR is signed by the Project Admin
with the same root CA used for all participants.

---

### Summary

| Signature System | Scope | Reason |
|-----------------|-------|--------|
| Startup kit `signature.json` | **CC and HE mode** | Meaningful in CC (attestation chain before mTLS exists) and HE (shared TenSEAL context must be integrity-verified). In all other standard deployments mTLS is the trust anchor — `signature.json` is redundant regardless of provisioning method. |
| Job submission `__nvfl_sig.json` | **All production deployments** | Authenticates the job *submitter* (admin-type cert), proves job content unmodified in transit. Independent of provisioning method and transport driver. Must not be gated on `secure_train`. |

---

## Implementation Notes

### What `nvflare package` Must Produce (Manual Workflow)

Minimum startup kit contents assembled by `nvflare package`:

```
hospital-1/
└── startup/
    ├── client.crt          # copied from received cert (renamed from <name>.crt by PrebuiltCertBuilder)
    ├── client.key          # generated locally (never transmitted)
    ├── rootCA.pem          # received from Project Admin
    ├── fed_client.json     # generated by nvflare package (server URI + client name)
    └── start.sh            # generated by nvflare package
```

No `signature.json` is generated. The startup kit integrity check applies only to CC and HE kits.

`local/` is also produced with the usual runtime-compatible layout (authorization policy, log config, etc.) so the site is indistinguishable from a centrally provisioned kit at runtime.

---

### `secure_train` Is a Historical Proxy, Not a Clean Design

`secure_train=true` was introduced as a single mode flag meaning "run with PKI." Over
time it became the gate for several unrelated checks:

- Startup kit integrity (`signature.json` verification)
- Job folder signature verification (`job_runner.py`, `training_cmds.py`)
- PKI challenge-response auth for relay topologies (`authenticator.py`)

The logical condition these checks actually require is not "are we in secure mode" —
it is "does the relevant PKI artifact exist." Using `secure_train` as a proxy was a
shortcut that worked when POC (no PKI) and production (full PKI) were the only two
cases. Distributed provisioning breaks that assumption: a site may have full PKI
(mTLS certs, root CA) but no `signature.json`.

Each check needs its own correct gate:

| Check | Correct gate | `secure_train` gate? |
|-------|-------------|----------------------|
| Startup kit integrity (`signature.json`) | `valid_config == True` (i.e., `signature.json` present) | No — CC and HE only |
| Job folder signature | Submitter cert + key present; `require_signed_jobs` server config | No — independent of mode |
| PKI challenge-response auth | Relay topology detected | No change needed |

---

### Startup Kit Integrity Check: Scope It to CC and HE

The startup check fires in `fed_utils.py:security_init()` when `secure_train=True`
and `signature.json` is absent (`valid_config=False`), causing `sys.exit(1)`. This
is a false abort for the Manual Workflow — nothing was tampered, there was just no
centrally assembled kit to sign.

**Change 1 — `nvflare/private/fed/utils/fed_utils.py`**

```python
# Before:
if secure_train:
    insecure_list = _check_secure_content(site_type=site_type)
    if len(insecure_list):
        sys.exit(1)

# After (gate on valid_config — only meaningful when signature.json was generated):
if secure_train and SecurityContentService.security_content_manager.valid_config:
    insecure_list = _check_secure_content(site_type=site_type)
    if len(insecure_list):
        sys.exit(1)
```

CC and HE deployments always have `signature.json` → `valid_config=True` → check runs unchanged.
Plain non-CC, non-HE deployments have no `signature.json` → `valid_config=False` → check skipped correctly.

---

### Job Folder Signing: Who, With What, and When

The job folder signature is produced by the **job submitter** — the human or agent
running `nvflare job submit`. It is entirely separate from the site process identity
(the mTLS client/server cert).

FLARE uses certificate types that map to runtime roles:

| Type (`-t`) | Who holds it | Used for |
|-------------|-------------|---------|
| `server` | Site running the FL server process | mTLS server identity — authenticates the server endpoint to connecting clients |
| `client` | Site running the FL client process | mTLS client identity — authenticates the site to the server during training |
| `org_admin` | Org-level administrator | Sign job folders; admin API authentication |
| `lead` | Lead researcher / primary job submitter | Sign job folders; admin API authentication |
| `member` | Team member (limited access) | Sign job folders; admin API authentication |

The value is stored as `UNSTRUCTURED_NAME` in the cert and read at runtime as the role.
`project_admin` is not a valid `-t` value — the Project Admin self-provisions via `cert init`.

In the Manual Workflow a job submitter follows the same CSR flow:

```bash
nvflare cert csr -n researcher-alice -t lead -o ./alice-csr
# sends alice.csr to Project Admin
nvflare cert sign -r alice.csr -c ./ca -o ./signed -t lead
# receives alice.crt + rootCA.pem back
```

At submission time `push_folder()` calls:
```python
sign_folders(job_path, alice.key, alice.crt)
```
producing `__nvfl_sig.json` + `.__nvfl_submitter.crt` in each job subfolder.

The signature proves two things:
1. **Authorization** — `.__nvfl_submitter.crt` chains to `rootCA.pem`. The submitter
   was explicitly authorized by the Project Admin.
2. **Integrity** — all job files match `__nvfl_sig.json`. Files were not modified in
   transit from submitter to server/client.

This verification is MORE important in distributed provisioning than in centralized,
because the Manual Workflow has no central gatekeeper controlling who submits jobs —
the only proof of authorization is the cert chain.

**Change 2 — signing side (`push_folder` / `sign_folders`)**

Currently `sign_folders()` is called **unconditionally** in `push_folder`
(`file_transfer.py:308-310`) with no guard:

```python
client_key_file_path = api.client_key
private_key = load_private_key_file(client_key_file_path)  # raises if path absent/invalid
sign_folders(full_path, private_key, api.client_cert)
```

`load_private_key_file` is called before any guard can act — the crash happens on line
309, not at `sign_folders`. The guard must be placed **before the key load**:

```python
# After: check path existence before loading key
client_key_file_path = api.client_key
if client_key_file_path and os.path.exists(client_key_file_path) and api.client_cert:
    private_key = load_private_key_file(client_key_file_path)
    sign_folders(full_path, private_key, api.client_cert)
```

Simulator: `client_key` path absent → skip signing entirely.
POC and production (centralized or manual workflow): key file present → load and sign.

**Change 3 — verification side (`job_runner.py:167`, `training_cmds.py:109`)**

The correct gate is: is a signature file present, and does server policy require it?

```python
# Before:
if secure_train and not from_hub_site:
    verify_folder_signature(app_path, root_ca_path)

# After:
if not from_hub_site:
    sig_file = os.path.join(app_path, "__nvfl_sig.json")
    if os.path.exists(sig_file):
        verify_folder_signature(app_path, root_ca_path)
    elif server_config.require_signed_jobs:   # default True when rootCA.pem present
        raise RuntimeError(f"Unsigned job rejected — require_signed_jobs is enabled")
```

`require_signed_jobs` is a server-side configuration — see §`require_signed_jobs` Policy
for the full specification.

---

### `require_signed_jobs` Policy

#### Config location and schema

`require_signed_jobs` is a field in `fed_server.json` under the server site's startup kit:

```json
{
  "require_signed_jobs": true
}
```

It is read by the server process at startup via `SecurityContentService` / the normal
config loading path. It is not a client-side setting — clients verify whatever the
server deploys; the server decides whether unsigned submissions are accepted.

#### Default value logic

| Deployment | Default |
|------------|---------|
| `rootCA.pem` present in startup kit | `true` — any PKI deployment requires signed jobs |
| No `rootCA.pem` (simulator or bare POC without PKI) | `false` — no cert infrastructure, signing not possible |

If `require_signed_jobs` is absent from `fed_server.json`, the server infers the default
from `rootCA.pem` presence. Explicit `false` overrides the inferred default (useful for
a temporary migration window).

#### Backward compatibility

Existing centrally provisioned deployments all have `rootCA.pem` and all jobs signed
(because `push_folder` calls `sign_folders` unconditionally today). Setting
`require_signed_jobs=true` (the default) is safe — existing jobs already have
`__nvfl_sig.json`. No `fed_server.json` update is required for existing deployments.

#### Interaction with hub/relay (`from_hub_site`)

The `from_hub_site` flag is set when a job is forwarded from a hub to a leaf server
in a relay topology. The leaf server skips re-verification when this flag is set:

```python
if not from_hub_site:
    sig_file = os.path.join(app_path, "__nvfl_sig.json")
    if os.path.exists(sig_file):
        verify_folder_signature(app_path, root_ca_path)
    elif require_signed_jobs:
        raise RuntimeError("Unsigned job rejected — require_signed_jobs is enabled")
# from_hub_site=True: leaf trusts hub; skip re-verification
```

This bypass is valid only under the following **normative security invariant**, which
must be enforced by the hub implementation:

> **Hub trust invariant:**
> 1. The hub MUST call `verify_folder_signature()` on every forwarded job before setting
>    `from_hub_site=True` and dispatching to leaf servers.
> 2. The hub MUST forward job artifacts unchanged — signature files (`__nvfl_sig.json`,
>    `.__nvfl_submitter.crt`) MUST be preserved in the forwarded payload.
>
> If either condition is violated, the leaf's bypass becomes an authentication gap.
> Implementors adding hub/relay functionality MUST NOT set `from_hub_site=True` without
> completing the signature check.

The job signature travels with the payload end-to-end through the relay (see
§Transport Driver Effects), so the original submitter's authorization proof is preserved
as long as the invariant above holds.

#### Error contract for unsigned jobs

When `require_signed_jobs=True` and a job arrives without `__nvfl_sig.json`:

- **Server**: raises `RuntimeError("Unsigned job rejected — require_signed_jobs is enabled")`
  → job deploy returns `error_reply` → admin CLI exits with code 1
- **Client**: same path in `training_cmds.py` → `error_reply` → task fails with structured error
- **Error code** (for `--output json`): `UNSIGNED_JOB_REJECTED`

Operators can disable enforcement temporarily by setting `require_signed_jobs: false`
in `fed_server.json` and restarting the server. This should be treated as a temporary
migration measure, not a permanent policy.

---

### Transport Driver Effects on Job Signature Enforcement

The job folder signature is the only mechanism that authenticates the job *submitter*
end-to-end, regardless of transport. Its importance varies by driver:

#### mTLS (gRPC with mutual TLS — standard production)

Both server and client present X.509 certificates in the TLS handshake. The server
authenticates the connecting *site process* (client-type cert). The client authenticates
the *server process* (server-type cert).

The job folder signature authenticates the *submitter* (admin-type cert) — a completely
different identity from the site process. A site could have a valid mTLS client cert
but no authorized job submitter cert. The two checks are **orthogonal and both needed**.

```
mTLS:       site-process cert  ←→  server cert       (transport-level, per-connection)
Job sig:    admin cert  →  job folder  →  server verify  (application-level, per-job)
```

#### TLS only (one-way TLS — server cert only)

Only the server presents a cert. The client (submitter) is **not authenticated at the
transport layer**. Any party that can reach the admin port can attempt to submit jobs.

The job folder signature is the **only proof of submitter identity**. Without it, the
server has no cryptographic basis for accepting or rejecting a job. Job signature
enforcement is **critical** in TLS-only deployments.

#### HTTPS / REST

Typically one-way TLS with session token or JWT for API-level access control. The
session token gates access to the API endpoint but does not prove anything about the
job *content* — a compromised session token could be used to submit tampered jobs.

The job folder signature provides content-level integrity and submitter identity that
session tokens cannot. Both layers are needed:

```
HTTPS session token:  gates API access          (coarse, session-scoped)
Job sig:              proves submitter + content (fine-grained, per-job, cryptographic)
```

#### Relay / Hub topology

TLS is terminated at the relay. The end server sees the relay's cert, not the original
client's. The `from_hub_site` flag (`if not from_hub_site`) already handles this case —
the relay's forwarded traffic is trusted at the transport layer.

The job folder signature, however, travels with the job payload end-to-end through the
relay. The server and clients verify it against `rootCA.pem` regardless of how many
relay hops it traversed. It is the only end-to-end cryptographic proof of submitter
identity in relay topologies.

---

### Complete Change Summary

| Change | File | What changes | Why |
|--------|------|-------------|-----|
| 1 | `lighter/impl/signature.py` | Stop generating `signature.json` for non-CC, non-HE kits; detect HE via tenseal file presence | For FL runtime sites (server/client), `signature.json` presence now indicates CC or HE; non-CC, non-HE kits get none |
| 2 | `fed_utils.py` (`security_init`, `security_init_for_job`) | Gate startup check on `valid_config` | Now semantically correct — `valid_config=True` implies CC or HE (after Change 1) |
| 3 | `file_transfer.py` (`push_folder`) | Move key load inside existence guard (before `load_private_key_file`) | Prevents crash in simulator (no key file); POC and production behavior unchanged |
| 4 | `job_runner.py`, `training_cmds.py` | Gate verification on sig file presence + `require_signed_jobs`, not `secure_train` | Sig file presence is the correct condition; enforces policy explicitly |
| 5 | `fuel/hci/client/config.py` (`secure_load_admin_config`) | Gate admin config tamper check on `mgr.valid_config` | Without `signature.json` (non-CC, Manual Workflow) `fed_admin.json` returns `NOT_SIGNED`; strict `LoadResult.OK` check causes `ConfigError` on admin login |

### Result

After the five changes, both provisioning methods produce kits that are
**indistinguishable at runtime**. The server does not need to know — and cannot tell —
which method was used for a given site.

| Scenario | Startup kit check | Job sig — signing | Job sig — verify |
|----------|-------------------|--------------------|------------------|
| CC (centralized provision) | ✅ `valid_config=True` | ✅ cert present | ✅ sig present |
| HE — Option C (centralized provision) | ✅ `valid_config=True` (HE context files trigger signing in Step 1) | ✅ cert present | ✅ sig present |
| Standard, centralized provision | ⏭ `valid_config=False` (Change 1 stops generating `signature.json`) | ✅ cert present | ✅ sig present |
| Manual Workflow (distributed) | ⏭ `valid_config=False` | ✅ cert present | ✅ sig present |
| Mixed: some sites centralized, some manual | ⏭ per-site (each site resolves independently) | ✅ cert present (both) | ✅ sig present (both) |
| POC (`nvflare poc`) | ⏭ no `signature.json` | ✅ cert present (POC provisions PKI via its own CA; admin certs created; job signing works) | ✅ sig present; `require_signed_jobs=True` (rootCA.pem present) |
| Simulator (`nvflare simulator`) | ⏭ no `signature.json` | ⏭ no key file (fully local in-process, no PKI) | ⏭ no sig, `require_signed_jobs=False` |

---

### Mixed Federation: Centralized and Distributed Provisioning Coexisting

A single federation can have some sites centrally provisioned and others using the
Manual Workflow. This is the expected deployment model for organizations that:

- Have an existing federation (centrally provisioned) and need to add a new partner
  without a full re-provision
- Use centralized provisioning for known long-term participants and the Manual Workflow
  for new or temporary collaborators

**Why this works without coordination:**

Each site's startup behavior is determined entirely by its own local kit — the startup
check gates on `valid_config` (local `signature.json` presence), not on what other
sites have. A centrally provisioned client and a manually provisioned client joining the
same server both:

1. Connect to the server over mTLS — their certs both chain to the same root CA
2. Sign submitted jobs with their admin-type cert — the server verifies against the same
   `rootCA.pem`
3. Receive and verify deployed job apps — `require_signed_jobs` is a server-side policy
   applied uniformly to all jobs regardless of submitter's provisioning method

The server treats both identically. No per-site provisioning metadata is needed.

**The only requirement for coexistence:** all participants — centrally provisioned and
manually provisioned — must share the same root CA. The Project Admin who runs
`nvflare provision` for centralized sites is the same entity who runs
`nvflare cert sign` for manual-workflow CSRs.

**Adding a new site to an existing centrally provisioned federation:**

```
Existing sites:    centrally provisioned, mTLS certs from root CA
New partner site:  Manual Workflow CSR → signed by same root CA → start.sh
Result:            new site joins federation with full job-signing support
```

No re-provisioning of existing sites required. No server restart required. The new
site is authorized the moment its cert is signed by the project root CA.

---

## Dynamic Provisioning: Joining a Running Federation

The Manual Workflow produces a valid startup kit. This section defines what happens
**after** `start.sh` is run — how the new participant joins the live federation, what
(if anything) the server admin must do, and how the new participant participates in FL jobs.

### Layer 1: mTLS Connection (Automatic, No Server Action Required)

The server accepts any client whose X.509 certificate chains to `rootCA.pem`. No
server-side allowlist, no restart, no reconfiguration.

When the new client runs `start.sh`:
1. Client presents its `client.crt` in the mTLS handshake
2. Server verifies: `client.crt` was signed by the root CA (whose public key is in `rootCA.pem`)
3. Handshake completes → client is connected

**This happens automatically.** The Project Admin signing the CSR with `nvflare cert sign`
is the act of authorization at the transport layer. No further server-side action is needed
for the connection to succeed.

Similarly, an admin-type user (role `lead`, `org_admin`, `member`) running the admin
CLI connects to the admin port with their cert. The connection is accepted by the same
mTLS verification.

---

### Layer 2: Role Assignment (Automatic, Derived from Certificate)

Each certificate carries its runtime role in the `UNSTRUCTURED_NAME` X.509 attribute
(set by `-t` in `nvflare cert sign`). The FLARE server and authorization service read
this attribute to determine what the cert holder may do:

| Cert type (`-t`) | Runtime role | What the holder can do |
|-----------------|-------------|------------------------|
| `client` | FL client identity | Participate in federated training tasks |
| `server` | FL server identity | Accept client connections, run FL server process |
| `org_admin` | Org admin | Submit jobs, manage organization's participation |
| `lead` | Lead researcher | Submit jobs, full admin API access |
| `member` | Team member | Submit jobs with potentially restricted scope |

**Role assignment is automatic** — no server configuration update needed. The cert type
embedded by the Project Admin at sign time IS the role at runtime.

---

### Layer 3: Authorization Policy (Server Admin Action, Hot-Reloadable)

NVFlare's authorization policy (`authorization.json` in the server's startup kit) controls
fine-grained actions: which roles can submit jobs to which sites, who can abort a running
job, who can list clients, etc.

#### Default authorization behavior

If `authorization.json` is absent or does not explicitly reference the new participant's
site name, NVFlare falls back to **role-based defaults** from the `FLAuthorizer`:

- `org_admin` and `lead` can submit jobs and use most admin API commands by default
- `member` has a more restricted default set

For many deployments, the role-only defaults are sufficient and no `authorization.json`
update is needed when a new participant joins.

#### When `authorization.json` must be updated

Authorization policy must be updated when:
- The federation uses **site-name-scoped** policies (e.g., "hospital-1 may only submit
  jobs targeting clients in its own org")
- A new `org_admin` must be granted cross-organization admin rights
- The policy explicitly allowlists which site names may submit

#### How to update `authorization.json` without restart

NVFlare's `SecurityContentService` reads `authorization.json` at startup. However, the
policy is re-read on each authorization check if the server is configured to do so.

**Hot-reload path** (no server restart):
1. Server admin edits `authorization.json` on the running server's `local/` directory
2. Sends the FLARE admin command: `set_run_number` or uses the admin API's
   `reload_authorization_policy` command (if available in the target release)
3. The updated policy takes effect for subsequent requests

**Without hot-reload** (restart required):
- Edit `authorization.json` in `startup/` or `local/`
- Restart the server process: `./startup/stop_fl.sh && ./startup/start.sh`
- This is disruptive to running jobs — prefer scheduling the restart between jobs

**Recommendation:** Use role-based defaults where possible to avoid per-participant
`authorization.json` entries. Dynamic participants should be assigned cert types (`lead`,
`org_admin`) that grant sufficient default permissions without requiring policy updates.

---

### Layer 4: FL Job Participation (Explicit Enrollment per Job)

FL jobs in NVFlare specify their participants explicitly. A new FL client site does not
automatically receive tasks — it must be listed in the job definition.

#### For FL client sites (cert type `client`)

A new client site connects to the server but receives no tasks until a job includes it.
To enroll the new site:

1. Job submitter adds the new site's name (`CN` from the cert, e.g. `hospital-5`) to
   the job's `meta.json` `participants` list or the job's `app/config/config_fed_server.json`
2. Submits the job: `nvflare job submit -j ./my_job`
3. The server dispatches tasks to `hospital-5` as part of the job

**The site name is the CN from `nvflare cert csr -n <name>`**. The Project Admin must
communicate the exact site name back to the FL job submitter out-of-band (this is the
same out-of-band communication as step 5 of the Manual Workflow).

#### For admin users (cert types `org_admin`, `lead`, `member`)

Admin users do not need to be listed in job definitions. Their cert grants them access
to the admin API immediately after mTLS connection. They can list running jobs, submit
new jobs, and perform admin actions according to their role's authorization policy.

#### Site name coordination

The site name (CN) is chosen by the site admin at CSR generation time. There is no
central registry. Job submitters must know the name of each client they want to include.
The Project Admin is the natural coordination point: they receive the CSR (which contains
the CN), sign it, and can relay the site name back to job submitters as part of the
step-5 communication ("here is `hospital-5.crt`, `rootCA.pem`, server URI
`grpc://server:8002` — the site name is `hospital-5`").

---

### Summary: What Requires Action When a New Participant Joins

| Concern | Who acts | Required? | Notes |
|---------|----------|-----------|-------|
| mTLS connection | Nobody (automatic) | Automatic | Cert chains to root CA → connection accepted |
| Role assignment | Nobody (from cert) | Automatic | `UNSTRUCTURED_NAME` in cert = runtime role |
| Authorization policy | Server admin | Only if site-scoped policy used | Hot-reload possible; role defaults usually sufficient |
| FL job enrollment (client sites) | Job submitter | Yes, per job | Add site name to job's participant list |
| Admin API access (admin users) | Nobody | Automatic | Role from cert grants access immediately |
| Server restart | Nobody | Never required | mTLS and role checks are stateless |

**The key insight:** mTLS + cert-embedded roles make the federation access-control
self-contained at the crypto layer. The only operationally required step for a new
participant is enrolling them in FL jobs — and that is a job-level concern, not a
federation-level reconfiguration.

---

### Dynamic Provisioning Flow (Full End-to-End)

```
New client site (hospital-5)              Project Admin                  FL Job Submitter
         |                                      |                                |
1. nvflare cert csr                             |                                |
   -n hospital-5 -t client                      |                                |
   -o ./csr                                     |                                |
         |                                      |                                |
2. Send hospital-5.csr ─────────────────────► Receive CSR                       |
                                               |                                 |
3.                                  nvflare cert sign                            |
                                    -r hospital-5.csr                           |
                                    -c ./ca -o ./signed                         |
                                    -t client                                    |
                                               |                                 |
4. Receive client.crt + rootCA.pem ◄───────── Send signed cert + rootCA.pem    |
   Receive server URI: grpc://server:8002       + server URI                    |
                                               │                                 |
4b.                                            └──────── Notify job submitter: ─►
                                                         site name = hospital-5
5. nvflare package                             |                                |
   -n hospital-5 -t client                     |                                |
   -e grpc://server:8002                        |                                |
   --dir ./csr                                  |                                |
         |                                      |                                |
6. ./startup/start.sh                          |                                |
   (mTLS handshake succeeds —                  |                                |
    cert verified against rootCA.pem)          |                                |
         |                                      |                                |
         ├── Connected to server ──────────────►                                |
         |   [no tasks yet]                     |                                |
         |                                      |                                |
7.                                             |         Add hospital-5 to      |
                                               |         job participant list    |
                                               |         nvflare job submit ────►
                                               |                                |
8.         ◄── Receives FL tasks ─────────────────────────────────────────────── job running
         |
   [participates in FL training]
```

Step 4b is out-of-band coordination (email, chat, or the same channel used for the cert
delivery). The job submitter needs the site name; the Project Admin has it because they
received the CSR (which contains the CN = site name).

---

### Implementing the CC / Non-CC Separation

The provisioning side already branches on `CC_ENABLED` (`lighter/impl/signature.py`):

```python
if p.get_prop(PropKey.CC_ENABLED):
    sign_folders(ctx.get_ws_dir(p), root_pri_key, ...)   # full workspace signed
else:
    sign_folders(ctx.get_kit_dir(p), root_pri_key, ...)  # startup + local only
```

**The problem:** `signature.json` is generated for **both** CC and non-CC provisioned
kits — just with different scope. So `SecurityContentManager.valid_config=True` for both,
and the startup integrity check currently runs for both. For non-CC kits this is redundant
(mTLS is the trust anchor); for the Manual Workflow it causes a false abort.

**The fix is entirely on the provisioning side.** No new runtime flags needed.

#### Step 1 — Stop generating `signature.json` for non-CC, non-HE kits

**`lighter/impl/signature.py`**:

HE deployments are an exception: `server_context.tenseal` and `client_context.tenseal`
contain shared encryption keys. The `load_tenseal_context_from_workspace` runtime function
explicitly requires `LoadResult.OK` (not `NOT_SIGNED`) in secure mode — so the HE context
files must be signed. `SignatureBuilder` runs after `HEBuilder` in the builder sequence,
so it can detect HE by checking for tenseal file presence in the kit directory.

> **Design note — builder-order dependency:** The `he_present` file-existence check works
> only because `HEBuilder` runs before `SignatureBuilder`. This is the default builder
> ordering and is correct for the standard provisioning path. However, it is fragile to
> custom builder orderings. The preferred long-term fix is to store an explicit
> `HE_ENABLED` marker in the provisioning context or project config (analogous to
> `CC_ENABLED`) and gate on `p.get_prop(PropKey.HE_ENABLED)` instead. The file-existence
> approach is acceptable for this iteration because `HEBuilder` is always the generator
> of those files and the ordering is enforced by the provisioner, but implementors adding
> custom builder sequences MUST ensure `HEBuilder` precedes `SignatureBuilder`.

```python
# Before:
if p.get_prop(PropKey.CC_ENABLED):
    sign_folders(ctx.get_ws_dir(p), root_pri_key, ...)   # full workspace
else:
    sign_folders(ctx.get_kit_dir(p), root_pri_key, ...)  # startup + local (always — too broad)

# After:
kit_dir = ctx.get_kit_dir(p)
he_present = (
    os.path.exists(os.path.join(kit_dir, ProvFileName.SERVER_CONTEXT_TENSEAL)) or
    os.path.exists(os.path.join(kit_dir, ProvFileName.CLIENT_CONTEXT_TENSEAL))
)

if p.get_prop(PropKey.CC_ENABLED):
    sign_folders(ctx.get_ws_dir(p), root_pri_key, ...)   # CC: full workspace
elif he_present:
    sign_folders(kit_dir, root_pri_key, ...)              # HE: sign to protect context files
# else: nothing — plain non-CC, non-HE kits get no signature.json
```

For FL runtime sites (server and client), `signature.json` presence now reliably signals
CC or HE mode. It is not generated for standard non-CC, non-HE kits or for the Manual Workflow.

The `valid_config` flag remains accurate:
- CC kit: `valid_config=True` → startup check + admin config check run → correct
- HE kit (centralized): `valid_config=True` → startup check + admin config check run → correct
- Standard non-CC, non-HE kit: `valid_config=False` → checks skipped → correct
- Manual Workflow kit: `valid_config=False` → checks skipped → correct

#### Step 2 — Gate the startup integrity check on `valid_config`

**`nvflare/private/fed/utils/fed_utils.py`** — two functions: `security_init` (site
startup) and `security_init_for_job` (subprocess startup for each job). Both have the
same `secure_train` gate and both need the same change:

```python
# Before (in both security_init and security_init_for_job):
if secure_train:
    insecure_list = _check_secure_content(site_type=site_type)
    if len(insecure_list):
        sys.exit(1)

# After (both functions):
# valid_config=True iff signature.json exists — which is now CC or HE mode only (Step 1).
if secure_train and SecurityContentService.security_content_manager.valid_config:
    insecure_list = _check_secure_content(site_type=site_type)
    if len(insecure_list):
        sys.exit(1)
```

Because Step 1 makes `signature.json` a CC or HE artifact, `valid_config=True` now
implies CC or HE mode. The gate is not a workaround — it is semantically correct.

**Scope note — `SecurityContentService` consumers:**

`SecurityContentService` is also used by the authorization policy loader inside
`_check_secure_content()` (`fed_utils.py`). That function does check
`sig != LoadResult.OK` for the authorization config — it would append it to
`insecure_list` if unsigned. However, `_check_secure_content()` is only called when
`valid_config=True` (Step 2's gate). In non-CC mode (`valid_config=False` after Step 1),
the entire `_check_secure_content()` block is skipped — so the authorization check
never runs, not because it relaxes its `OK` requirement, but because Step 2 gates the
whole block. No additional code change needed there.

**Admin config loading (`secure_load_admin_config`) is a different case — see Step 5.**

#### Step 3 — Remove `secure_train` from job sig verification

The job signature check is independent of CC, startup mode, and provisioning method.
Its correct gate is PKI presence, not a mode flag.

**`nvflare/private/fed/server/job_runner.py`** and
**`nvflare/private/fed/client/training_cmds.py`**:

```python
# Before:
if secure_train and not from_hub_site:
    verify_folder_signature(app_path, root_ca_path)

# After:
if not from_hub_site:
    sig_file = os.path.join(app_path, "__nvfl_sig.json")
    if os.path.exists(sig_file):
        verify_folder_signature(app_path, root_ca_path)
    elif require_signed_jobs:   # server config; default True when rootCA.pem present
        raise RuntimeError("Unsigned job rejected — require_signed_jobs is enabled")
```

#### Step 4 — Guard signing on key file existence (before load)

**`nvflare/fuel/hci/client/file_transfer.py`** (`push_folder`):

`load_private_key_file` is called unconditionally at line 309, before any guard. The crash
occurs there, not at `sign_folders`. Move the check to before the load:

```python
# Before (unconditional — load_private_key_file crashes if key path absent):
client_key_file_path = api.client_key
private_key = load_private_key_file(client_key_file_path)
sign_folders(full_path, private_key, api.client_cert)

# After: guard wraps the load, not just sign_folders
client_key_file_path = api.client_key
if client_key_file_path and os.path.exists(client_key_file_path) and api.client_cert:
    private_key = load_private_key_file(client_key_file_path)
    sign_folders(full_path, private_key, api.client_cert)
```

Simulator: key path absent → skip. POC and production: key present → load and sign.

#### Step 5 — Fix `secure_load_admin_config` to allow unsigned in non-CC

**`nvflare/fuel/hci/client/config.py`** (`secure_load_admin_config`):

This function creates its own `SecurityContentManager` and strictly requires
`LoadResult.OK` for `fed_admin.json`. Without `signature.json` (after Step 1, or in
the Manual Workflow), the result is `NOT_SIGNED`, causing `ConfigError` and breaking
admin login.

The correct fix is to scope the strict check to when `signature.json` exists
(CC mode), identical to the `fed_utils.py` approach in Step 2:

```python
def secure_load_admin_config(workspace: Workspace):
    mgr = SecurityContentManager(content_folder=workspace.get_startup_kit_dir())
    _, result = mgr.load_json(WorkspaceConstants.ADMIN_STARTUP_CONFIG)

    # Before:
    # if result != LoadResult.OK:
    #     raise ConfigError(f"invalid {WorkspaceConstants.ADMIN_STARTUP_CONFIG}: {result}")

    # After: strict check only when signature.json is present (CC mode).
    # NOT_SIGNED is acceptable in non-CC and Manual Workflow — mTLS is the trust anchor.
    if mgr.valid_config and result != LoadResult.OK:
        raise ConfigError(f"invalid {WorkspaceConstants.ADMIN_STARTUP_CONFIG}: tampered ({result})")
    # If signature.json is absent (valid_config=False), skip tamper check — no sig to verify against.

    conf = FLAdminClientStarterConfigurator(workspace=workspace)
    conf.configure()
    return conf
```

`mgr.valid_config` is `True` iff `signature.json` is present in the startup kit — which
is CC or HE mode after Step 1. In plain non-CC, non-HE kits and the Manual Workflow,
`valid_config=False` and the check is skipped. The `INVALID_SIGNATURE` result (active tamper, file exists but
sig doesn't match) would still need to be handled; the gate above catches it when
`valid_config=True`.

---

#### What `secure_train` Retains

After these changes `secure_train` has exactly one remaining role: enabling PKI
challenge-response authentication in relay/bridge topologies (`authenticator.py`).
That use is correct and untouched.

| Gate (before) | Replaced by (after) | Step |
|---------------|---------------------|------|
| `secure_train` — startup kit integrity check | `valid_config` — now reliably means CC or HE (Step 1) | 2 |
| Unconditional `load_private_key_file` in `push_folder` | key file existence check before load | 3 (4 in impl) |
| `secure_train` — job sig verification | sig file present + `require_signed_jobs` | 4 |
| `LoadResult.OK` required in `secure_load_admin_config` | `mgr.valid_config and result != OK` | 5 |
| PKI challenge-response auth (`authenticator.py`) | unchanged — `secure_train` retained here | — |

---

### `nvflare cert` CLI Commands Required

| Command | Purpose |
|---------|---------|
| `nvflare cert init -n <project> -o <dir>` | Project Admin: generate root CA key + self-signed cert |
| `nvflare cert csr -n <name> -t client\|server\|org_admin\|lead\|member -o <dir>` | Site Admin: generate private key + CSR locally |
| `nvflare cert sign -r <csr> -c <ca_dir> -o <dir> -t <type>` | Project Admin: sign CSR with root CA, output signed cert (`-t` is required and authoritative — not inferred from CSR) |

### `nvflare package` CLI Changes Required

Current `nvflare package` (provisioning-based) expects pre-built startup kits.
For the Manual Workflow, it must accept:

```
nvflare package -n <name> -e grpc://<server>:<port> -t client|server \
    --cert <signed.crt> --key <private.key> --rootca <rootCA.pem>
```

And generate `fed_client.json` (or `fed_server.json`) with the endpoint and participant name,
plus `start.sh`. No `signature.json` is generated.

Target simplified UX (future):
```
nvflare package -n hospital-1 -e grpc://server:8002 -t client --cert-dir ./signed
```

Where `--cert-dir` auto-discovers `client.crt`, `client.key`, `rootCA.pem` by convention.

---

## CC Deployments: Out of Scope for Distributed Provisioning

Confidential Computing deployments are **incompatible** with the distributed provisioning
model by design. The purpose of CC is IP protection: the training code, model architecture,
and weights must not be accessible or modifiable by the site operator. This requires the
CVM image to be **built centrally** by the Project Admin (or model owner) and its
measurement attested before any execution.

If site admins assembled their own startup kits and launched their own CVMs, they would
control what runs inside the enclave — defeating the entire IP protection guarantee.

CC deployments continue to use centralized `nvflare provision`, which:
- Builds the CVM image centrally
- Signs the full startup kit with the root CA key (`signature.json`)
- Distributes sealed kits to sites

Distributed provisioning (this document) applies only to standard non-CC deployments.
HE deployments use centralized provisioning (Option C); see §HE Deployments.

---

## HE Deployments and Distributed Provisioning

### Background: Why HE Complicates Distributed Provisioning

FLARE's HE implementation uses TenSEAL with a symmetric scheme (`ENCRYPTION_TYPE.SYMMETRIC`).
`HEBuilder` generates one TenSEAL context during `nvflare provision` and serializes it
differently per role:

| Role | Serialized content | Can do |
|------|--------------------|--------|
| Server | relin_keys only | Aggregate encrypted ciphertext; cannot decrypt |
| Client | public_key + secret_key + relin_keys | Encrypt model updates; decrypt aggregated result |

All clients share the **same** secret key. This is fundamentally different from mTLS
(each site has a unique private key) — sharing the HE context is intentional and
required for the scheme to work. However, the shared context must originate from a single
generation step to ensure all participants use the same cryptographic parameters.

`load_tenseal_context_from_workspace` enforces `LoadResult.OK` (not `NOT_SIGNED`) in
secure mode. The HE context files must be signed by `signature.json`, which is why
Step 1's refinement detects HE via tenseal file presence and keeps signing for HE kits.

### Option A — Extend Manual Workflow with Out-of-Band HE Context Distribution (Not Implemented)

The HE context is a *shared parameter*, not a per-site *private* key — distributing it
out-of-band is technically feasible. The Project Admin would generate the context
independently and send `client_context.tenseal` to each new site alongside `client.crt`
and `rootCA.pem`. `nvflare package` would be extended with `--he-context` to include it
in the startup kit.

**Why this was not chosen:** The HE secret key (embedded in the client context) must be
kept confidential and consistent across all participants. Any site that receives a stale
or mismatched context silently produces incorrect ciphertext — aggregation fails with no
clear error. Managing context versioning, ensuring all sites are updated atomically when
parameters change, and preventing context drift across incrementally onboarded sites
creates operational complexity disproportionate to the benefit. Re-running
`nvflare provision` (Option C) provides a clean, atomic solution with no additional
tooling. This option is documented as a considered and rejected path.

### Option B — Asymmetric HE with Per-Site Key Generation (Future Release)

The long-term solution eliminates the shared secret by switching to public-key HE:

- Each client generates its own HE key pair locally — no central distribution needed
- Client sends the HE public key to the server (alongside its mTLS CSR, or separately)
- Server aggregates encrypted updates using each client's public key
- Only the designated decryptor (server or trusted aggregator) holds the private key
  needed to decrypt the aggregated result

This makes HE fully compatible with the manual workflow: sites join incrementally via
the CSR flow, including their HE public key, with no shared secret to distribute.

FLARE's current HE stack (`intime_accumulate_model_aggregator.py`, `model_encryptor.py`,
`model_decryptor.py`) assumes a common symmetric context and requires redesign:

**Design decisions deferred to future release:**
- Asymmetric HE scheme selection (CKKS public-key mode, BFV, TFHE)
- Private key placement: server-held, dedicated aggregator, or threshold split across clients
- Aggregation protocol changes in `InTimeAccumulateWeightedAggregator`
- Whether `nvflare cert csr` is extended to carry the HE public key alongside the mTLS CSR
- `signature.json` no longer needed for HE context (no shared file to protect)

### Option C — Centralized Provisioning for HE (Phase 1, Supported)

HE deployments use `nvflare provision` as today. `HEBuilder` generates a single TenSEAL
context with shared encryption keys and distributes it to every site's startup kit.
Existing behavior is fully preserved with no code changes beyond Step 1's `he_present`
refinement (which ensures `signature.json` continues to be generated for HE kits).

Adding a new HE site requires re-running `nvflare provision` to regenerate and
redistribute the shared TenSEAL context to all participants.

**When to use:** All HE deployments in the current release.

### HE Option Summary

| Option | Status | Shared secret | Manual workflow | Code changes | `signature.json` |
|--------|--------|---------------|-----------------|--------------|------------------|
| A | Not implemented — considered and rejected | Yes — out-of-band per site | Partial — context drift risk | `nvflare he init` + `nvflare package --he-context` | Required |
| B | Future release | No — per-site key pair | ✅ Fully compatible | HE aggregator redesign | Not needed |
| C | Phase 1 — supported | Yes — via `nvflare provision` | ❌ Requires re-provision | Step 1 `he_present` only | Required |
