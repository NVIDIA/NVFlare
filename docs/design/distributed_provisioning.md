# FLARE Distributed Provisioning

Created: 2026-03-27
Updated: 2026-04-02

---

## Problem Statement

Current FLARE provisioning (`nvflare provision`) is centralized: the Project Admin must
collect all participant details upfront, generate each site's **private key** centrally,
and distribute full startup kits over a secure channel. This creates three problems:

1. **Private keys in transit** — keys are generated centrally and must be sent to each site
2. **Centralized gathering** — all participant information must be collected before any kit
   can be generated

---

## Approach: Manual Workflow (Distributed Provisioning)

The Manual Workflow eliminates both problems without requiring new infrastructure.
Each site generates its own private key locally. The only things exchanged are:

- **Site → Project Admin**: a Certificate Signing Request (CSR) — public key only
- **Project Admin → Site**: signed certificate + `rootCA.pem` + server URI

The resulting startup kits are structurally identical to those produced by
`nvflare provision` and are fully compatible with all FLARE runtime components.

### Step-by-Step Workflow

| Step | Who | Action |
|------|-----|--------|
| 1 | Site Admin | `nvflare cert csr -n hospital-1 -o ./csr` |
| 2 | Site Admin | Send `hospital-1.csr` to Project Admin (email, file share, etc.) |
| 3 | Project Admin | `nvflare cert init --project my-project -o ./ca` *(one-time per federation)* |
| 4 | Project Admin | `nvflare cert sign -r hospital-1.csr -t client -c ./ca -o ./signed/hospital-1` |
| 5 | Project Admin | Return `hospital-1.crt` + `rootCA.pem` + server URI to site |
| 6 | Site Admin | `nvflare package -n hospital-1 -t client -e grpc://server:8002 --cert hospital-1.crt --key hospital-1.key --rootca rootCA.pem` |
| 7 | Site Admin | `cd hospital-1 && ./startup/start.sh` |

Step 3 is done once. Each new participant repeats steps 1–2 and 4–7 independently.

### Multi-Participant Variant

When a site needs kits for multiple participants (e.g. a client process plus one or more
admin users), a site-scoped project YAML can provision all of them in a single command:

```bash
nvflare package -e grpc://fl-server:8002 -p site.yaml --dir ./certs
```

All participants land in the same `prod_NN` directory. The `site.yaml` follows the same
schema as `nvflare provision` project.yaml and supports custom builders.

---

## Comparison: Centralized vs. Distributed Provisioning

| | Centralized (`nvflare provision`) | Distributed (Manual Workflow) |
|---|---|---|
| **Private key custody** | Project Admin generates and distributes | Each site generates locally; never leaves the machine |
| **Data distributed to site** | Full startup kit (keys, certs, config, scripts) | Signed cert + `rootCA.pem` + server URI (~a few KB) |
| **Data sent from site** | Nothing | CSR (~1 KB, public key only) |
| **Steps for Project Admin** | One command provisions all sites | Sign one CSR per participant |
| **Steps for Site Admin** | Unzip and run | Generate CSR → send → receive cert → package → run |
| **Participant info required upfront** | All participants before any kit is generated | Each participant joins independently, on demand |
| **Adding a new site** | Dynamic provisioning (sign new cert with existing root CA) | Same workflow; no impact on existing sites |
| **CC deployments** | Supported | Not supported |
| **HE deployments** | Supported | Not supported (future) |
| **Trust required in Project Admin** | Must trust Project Admin with your private key | Project Admin never sees private keys |

---

## Assumptions and Non-Goals

### Trust Model

This document covers standard (non-Confidential Computing) deployments.

- **mTLS is the trust anchor.** A participant is authorized if and only if its certificate
  chains to the project root CA. No startup kit integrity signature (`signature.json`) is
  needed for this check.
- **Trusted local site admin.** The site admin who runs `nvflare package` and `start.sh`
  is trusted not to modify their own startup kit maliciously. Startup kit integrity
  signatures are a CC and HE concern, not a general tamper check.
- **Local configuration customization is allowed.** Site admins may adjust local config
  (resource limits, log levels) without triggering any integrity failure.

### Non-Goals

- **Confidential Computing** — CC deployments are centralized by design; see §CC Deployments.
- **HE with distributed provisioning** — current HE requires a shared symmetric key generated
  centrally; per-site asymmetric HE is a future release item; see §HE Deployments.
- **Hierarchical FL (relay nodes)** — relay node ownership is ambiguous in a distributed model.
  Use `nvflare provision` for relay topologies.
- **Multi-root-CA federations** — all participants share one root CA per project.
- **Key rotation** — out of scope for this iteration.
- **Automated CSR exchange** — the channel is intentionally out-of-band.

---

## Signature Handling

FLARE has two independent signature systems.

### System 1: Startup Kit Integrity (`signature.json`)

**Purpose:** Detect tampering with startup kit configuration files after distribution.
Generated during `nvflare provision` by `SignatureBuilder` using the root CA private key.
Verified at runtime by `SecurityContentManager`.

**Decision for distributed provisioning:** `signature.json` is NOT generated for
non-CC, non-HE kits. In the Manual Workflow the site assembles its own kit locally —
there is no centrally distributed kit to protect. mTLS catches any meaningful tampering
(replacing `rootCA.pem` or the server URI causes mTLS to fail immediately).

`signature.json` continues to be generated for:
- **CC deployments** — part of the CVM attestation chain; required before mTLS exists
- **HE deployments** — protects the shared TenSEAL context files

### System 2: Job Submission Signature (`__nvfl_sig.json`)

**Purpose:** Authenticate the job submitter and prove job content was not modified in
transit. Generated by `push_folder` in `file_transfer.py` using the admin user's private
key at submission time. Verified by the server (`job_runner.py`) and each client
(`training_cmds.py`) before executing the job.

**Decision:** Job signing is fully preserved in the Manual Workflow. After the workflow,
the submitter holds a private key and a certificate signed by the project root CA —
exactly what `sign_folders()` requires. No changes to the signing or verification logic.

The server policy `require_signed_jobs` (default: `true` when `rootCA.pem` is present)
controls whether unsigned jobs are rejected. This policy applies uniformly regardless of
how participants were provisioned.

---

## Code Changes

Five targeted changes are required to support the Manual Workflow without breaking
existing centralized provisioning, CC, or HE deployments.

| # | File | Change |
|---|------|--------|
| 1 | `lighter/impl/signature.py` | Stop generating `signature.json` for non-CC, non-HE kits. Gate on `CC_ENABLED` or presence of TenSEAL context files. |
| 2 | `fed_utils.py` (`security_init`, `security_init_for_job`) | Gate startup integrity check on `SecurityContentService.security_content_manager.valid_config` instead of `secure_train`. After Change 1, `valid_config=True` implies CC or HE mode. |
| 3 | `file_transfer.py` (`push_folder`) | Guard `load_private_key_file` on key file existence before loading. Prevents crash in simulator (no key file); POC and production behavior unchanged. |
| 4 | `job_runner.py`, `training_cmds.py` | Replace `secure_train` gate on job sig verification with: verify if `__nvfl_sig.json` present; reject if absent and `require_signed_jobs=true`. |
| 5 | `fuel/hci/client/config.py` (`secure_load_admin_config`) | Gate strict `LoadResult.OK` check on `mgr.valid_config`. Without `signature.json` (non-CC, Manual Workflow), `fed_admin.json` returns `NOT_SIGNED`; the strict check causes admin login to fail. |

After these changes, `secure_train` retains exactly one role: enabling PKI
challenge-response authentication in relay/bridge topologies. All other checks use
semantically correct conditions.

---

## Backward Compatibility

Distributed provisioning is additive. The five code changes above are designed to leave
all existing deployments fully intact:

| Deployment | Behavior after changes |
|------------|----------------------|
| Centralized provisioning (non-CC, non-HE) | `signature.json` no longer generated (Change 1); startup check skipped (`valid_config=False`). Runtime behavior identical — mTLS was already the trust anchor. |
| Centralized provisioning (CC) | `signature.json` still generated; startup check still runs. No change. |
| Centralized provisioning (HE) | `signature.json` still generated for TenSEAL context files. No change. |
| Distributed provisioning (new) | No `signature.json`; all checks correctly skipped. |
| Mixed federation (some centralized, some distributed) | Each site resolves independently based on its own kit. Both connect to the same server over the same root CA. |
| Simulator | No PKI; job signing skipped (no key file). No change. |

`nvflare provision` is unchanged. The new `nvflare cert` and `nvflare package` commands
are purely additive — no existing CLI commands, configuration files, or runtime
behaviors are modified.

---

## Roles and Authorization

### Participant Types

The `-t` argument to `nvflare cert sign` sets the certificate type and determines what
the holder may do.

**Site participants** — FL process identities; no role embedded in the certificate:

| Type | Description |
|------|-------------|
| `server` | FL server process identity (mTLS server endpoint) |
| `client` | FL client (data site) process identity |

**Users** — human operators connecting via the admin API; role embedded in
`UNSTRUCTURED_NAME` in the certificate:

| Role | Description |
|------|-------------|
| `lead` | Lead researcher. Can submit jobs, manage own jobs, operate and deploy custom code on own site. |
| `org_admin` | Organization administrator. Can manage own site and running jobs from their org; cannot submit new jobs. |
| `member` | Read-only observer. Can view jobs and status; no submit or operate permissions. |

`project_admin` is not a `-t` value — the Project Admin self-provisions via `nvflare cert init`.

### Default Authorization Policy

The role embedded in the certificate is enforced at runtime against
`local/authorization.json.default` on each site:

| Permission | `lead` | `org_admin` | `member` |
|-----------|--------|------------|---------|
| `submit_job` | any site | none | — |
| `clone_job` | own jobs | none | — |
| `manage_job` (abort/delete) | own jobs | jobs from own org | — |
| `download_job` | own jobs | jobs from own org | — |
| `view` | any | any | any |
| `operate` | own site | own site | — |
| `byoc` | any | none | — |

Sites may customize `local/authorization.json.default` to tighten or loosen these rules.

---

## Joining a Running Federation

Once a site runs `start.sh`, two things happen automatically:

1. **mTLS connection** — the server accepts any client whose certificate chains to
   `rootCA.pem`. No server-side allowlist, no restart, no reconfiguration required.
   The Project Admin signing the CSR is the act of authorization.

2. **Role assignment** — the `UNSTRUCTURED_NAME` field in the certificate is read at
   login time and enforced against the site's authorization policy. No server
   configuration update is needed.

**Adding a new site to an existing centrally provisioned federation** follows the same
workflow — the new participant's CSR is signed by the same root CA. No re-provisioning
of existing sites is required.

---

## CLI Commands

### `nvflare cert init` — Project Admin, once per federation

```
nvflare cert init --project <name> -o <dir>
```

Produces: `rootCA.key` (keep secret), `rootCA.pem` (distribute to all sites), `ca.json`.

### `nvflare cert csr` — Site Admin

```
nvflare cert csr -n <participant-name> -o <dir>
```

Produces: `<name>.key` (stays on site, never shared), `<name>.csr` (send to Project Admin).

Note: `-t` is not accepted here. The certificate type is set authoritatively by the
Project Admin when signing — not by the site generating the CSR.

### `nvflare cert sign` — Project Admin

```
nvflare cert sign -r <name>.csr -t <type> -c <ca-dir> -o <dir>
```

The `-t` argument is authoritative — the type set here is what the signed certificate
will carry, regardless of anything in the CSR.

### `nvflare package` — Site Admin

```
# Explicit mode
nvflare package -n <name> -t <type> -e grpc://<server>:<port> \
    --cert <name>.crt --key <name>.key --rootca rootCA.pem

# Auto-discovery mode (files in one directory)
nvflare package -t <type> -e grpc://<server>:<port> --dir <dir>

# Multi-participant YAML mode
nvflare package -e grpc://<server>:<port> -p site.yaml --dir <dir>
```

Output goes to `<workspace>/<project>/prod_NN/<name>/`.

---

## CC Deployments: Out of Scope

Confidential Computing deployments are incompatible with distributed provisioning by
design. CC requires the CVM image to be built and attested centrally — if site admins
assembled their own kits, they would control what runs inside the enclave, defeating the
IP protection guarantee.

CC deployments continue to use centralized `nvflare provision` unchanged.

---

## HE Deployments

HE deployments are not supported in this release. FLARE's current HE implementation
uses a symmetric TenSEAL context where all clients share the same secret key, which
requires centralized generation via `nvflare provision`. Use centralized provisioning
for any federation that requires HE.

Asymmetric (per-site) HE key support is planned for a future release, at which point
HE will be fully compatible with the distributed provisioning workflow.
