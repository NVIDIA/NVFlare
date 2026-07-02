# Per-Job Certificates for Job Cells

This document describes per-job TLS credentials for job processes (SJ: server
job process, CJ: client job process). Today both job cells load the same
provisioned site certificates as their parent processes (SP: server parent,
CP: client parent), so every job process holds the site's long-lived private
key.

## Problem

Job processes run job-supplied code (custom Executors, Controllers, third-party
training code). The cell creation paths give them the site identity keys:

- SJ: `BaseServer.create_job_cell()` reads `server.crt` / `server.key` from the
  server startup kit — identical to SP.
- CJ: `FederatedClientBase._create_cell()` reads `client.crt` / `client.key`
  from the client startup kit — identical to CP.

Any code running inside a job can therefore read the site's private key and
impersonate the site indefinitely: register as a CP, decrypt message-level
traffic, or authenticate as the site after the job ends.

## Goal

Give each job process its own short-lived credential, scoped to one job, so
that job code never needs the site's long-lived key for cell communication.

Non-goals of this phase (see Future Work):

- Removing site keys from job workspaces / container bundles. The mechanism
  introduced here is the prerequisite for that hardening.
- Certificate revocation. Short validity plus job-workspace teardown bound the
  exposure window instead.

## Trust Model

```text
rootCA (private key exists only during provisioning)
├── server.crt / server.key        server startup kit      (unchanged)
├── client.crt / client.key        client startup kits     (unchanged)
└── job_ca.crt / job_ca.key        server startup kit ONLY (new)
        CA:TRUE, pathlen:0
        └── per-job leaf certs, issued at job deploy time
              CN=<site name>, job_id extension, short validity
```

Because the job CA chains to the existing root, no participant needs a new
trust anchor: a job cert presented together with `job_ca.crt` validates against
`rootCA.pem` with standard X.509 path validation, both in TLS handshakes and in
`verify_cert_chain()`.

Leaf certs keep `CN=<site name>` so every existing common-name-based identity
check continues to pass. The job binding is carried in a certificate extension
holding the job ID.

Only SP ever holds the job CA key. CP receives issued certificates; it does no
signing.

## Provisioning

`CertBuilder` gains an `enable_job_ca` option (default off):

```yaml
builders:
  - path: nvflare.lighter.impl.cert.CertBuilder
    args:
      enable_job_ca: true
```

When enabled, provisioning generates one additional pair signed by the root:
`job_ca.crt` / `job_ca.key`, written to the **server** startup kit only, with
the key at mode 0600. The pair is persisted in the certificate state file so
re-provisioning reuses it. `pathlen:0` prevents the job CA from issuing further
CAs.

Client startup kits are unchanged. Existing projects keep working without
re-provisioning; the feature simply stays off.

## Runtime Issuance (SP)

A `JobCertIssuer` in the SP process loads `job_ca.crt` / `job_ca.key` from the
startup kit. If the files are absent the issuer is disabled and the whole
feature falls back to current behavior.

During job deployment (`JobRunner._deploy_job`), for each participant the
issuer generates an RSA keypair and a leaf certificate:

- subject `CN=<site name>` (the participant name, `server` included)
- a job-ID extension identifying the job
- `notAfter` = deploy time + a bounded validity (default 30 days), clamped to
  the job CA's own expiry

The issued credential is a PEM bundle: leaf cert followed by `job_ca.crt`
(so TLS peers can build the chain to the root), plus the private key PEM.

## Distribution

**SJ (local write).** When the server app is deployed to the server workspace,
SP writes the SJ credential into the job run directory before the SJ process is
launched.

**CJ (push).** The job deploy message becomes per-site: the shared app bytes
stay a single payload reference, but each site's message carries an additional
header with that site's cert bundle and private key. The message travels over
the existing authenticated CP–SP channel (mTLS in secure mode). On the client,
the deploy processor writes the credential into the job run directory (key at
mode 0600) after the app passes signature verification and deploys.

No new channels, topics, or handshakes are introduced. Because the run
directory is part of the per-job workspace, detached launchers that bundle the
run directory (e.g. the Kubernetes no-shared-PVC workspace transfer) deliver
job credentials to job pods with no additional work.

## Workspace Layout

```text
<workspace>/
  startup/                  site certs (unchanged); server kit adds job_ca.*
  <run dir for job_id>/
    app_<site>/             deployed app (unchanged)
    job_cert/
      job.crt               leaf + job_ca.crt PEM bundle
      job.key               per-job private key, 0600
```

## Job Process Changes

At job-process startup, the starter configers check the job run directory for
`job_cert/job.crt` + `job_cert/job.key`. When present, the paths are added to
the process security config under new keys (`job_ssl_cert`,
`job_ssl_private_key`) alongside — not replacing — the site cert entries.

Only the cell creation paths prefer the job credential:

- `BaseServer.create_job_cell()` uses the job cert/key for the SJ cell when
  configured.
- `FederatedClientBase._create_cell()` uses the job cert/key for the CJ cell
  when configured.

`ssl_root_cert` remains `rootCA.pem` everywhere. All other consumers of the
site credential (auth-token verification, identity assertion) are unchanged in
this phase.

If the job credential is absent — feature disabled, old server, non-secure
mode, simulator — cell creation uses the site certs exactly as today.

## Site-Scope Rejection

A job cert is a valid `CN=<site>` certificate chaining to the project root, so
without an extra check a leaked job key could be replayed at site scope for its
validity window — most notably to register a rogue CP (client registration
accepts a caller-supplied cert chain). All site-scope identity assertions
funnel through `IdentityVerifier.verify_common_name()` (client registration,
admin login, and the client's verification of the server), and no job cell
ever legitimately asserts identity there. `verify_common_name()` therefore
rejects any certificate carrying the job-ID extension. The rejection is keyed
on the extension, not the issuer, so it holds regardless of which CA issued
the certificate (this also keeps future HA setups with multiple job CAs
simple).

## Compatibility

| Deployment | Behavior |
| ---------- | -------- |
| Kit without job CA (existing) | No job certs issued; job cells use site certs (today's behavior) |
| Server kit with job CA, current clients | CJ certs pushed and used; SJ cert used |
| Server kit with job CA, older client release | Client ignores the unknown deploy header; CJ falls back to site certs |
| Non-secure mode / simulator / POC default | Feature inactive |

## Future Work

1. **Job-FQCN binding in cellnet.** Site-scope rejection is implemented (see
   above). The remaining enforcement is at the cellnet message-crypto layer:
   accept a peer cert carrying a job-ID extension only from an FQCN whose job
   suffix matches, preventing one job's leaked key from impersonating another
   job's cell. Deferred until the sub-worker/workspace redesign settles the
   process and FQCN layout.
2. **Site-key isolation.** Stop shipping `client.key` / `server.key` into job
   workspaces, workspace-transfer bundles, and job-pod Secrets once job cells
   no longer need them. The CP registration auth token and its signature,
   currently passed to job processes on the command line, belong on the same
   list.
3. **CSR mode.** Optional variant where CP generates the keypair and sends a
   CSR, so private keys never transit SP→CP.

## Unresolved Questions

1. Should sub-worker cells (multi-GPU `sub_worker_process`, Client API
   subprocesses) also use the job credential, or continue on internal links?
2. Should job-cert validity be configurable per job (e.g. from job meta or a
   server config var) rather than a fixed default?
3. HA with multiple SPs: is per-server-kit job CA acceptable, or must all SPs
   share one job CA identity?

(Message-level encryption was verified to already support certificate chains:
`CredentialManager` loads multi-cert PEMs and `SimpleCellCipher` validates
leaf + intermediates against the root, so no changes were needed there.)
