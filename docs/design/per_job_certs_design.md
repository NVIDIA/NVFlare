# Per-Job Certificates for Job Cells

This document describes per-job TLS credentials for job processes (SJ: server
job process, CJ: client job process). Before this change both job cells loaded
the same provisioned site certificates as their parent processes (SP: server
parent, CP: client parent), so every job process held the site's long-lived
private key.

## Problem

Job processes run job-supplied code (custom Executors, Controllers, third-party
training code). The cell creation paths gave them the site identity keys:

- SJ: `BaseServer.create_job_cell()` read `server.crt` / `server.key` from the
  server startup kit — identical to SP.
- CJ: `FederatedClientBase._create_cell()` read `client.crt` / `client.key`
  from the client startup kit — identical to CP.

Any code running inside a job could therefore read the site's private key and
impersonate the site indefinitely: register as a CP, decrypt message-level
traffic, or authenticate as the site after the job ends.

## Goal

Give each job process its own short-lived credential, scoped to one job, make
that credential the only one the job process refers to, and have container and
scheduler launchers withhold the site private keys from the job entirely.

Non-goal: certificate revocation. Bounded validity plus job-workspace teardown
limit the exposure window instead.

## Trust Model

```text
rootCA (private key exists only during provisioning)
├── server.crt / server.key        server startup kit      (unchanged)
├── client.crt / client.key        client startup kits     (unchanged)
└── job_ca.crt / job_ca.key        server startup kit ONLY (new)
        CA:TRUE, pathlen:0, job-CA marker extension
        └── per-job leaf certs, issued at job deploy time
              CN=<site name>, job_id extension, bounded validity
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

### Extension OIDs

Both extensions live under NVIDIA's IANA private enterprise arc
(`1.3.6.1.4.1.5703`), sub-arc `300`:

| OID | Placed on | Meaning |
| --- | --------- | ------- |
| `1.3.6.1.4.1.5703.300.1` | job leaf certs | the job ID the credential is bound to |
| `1.3.6.1.4.1.5703.300.2` | the job CA cert | "issued by the job CA" marker |

Both are non-critical, so standard TLS stacks ignore them; only FLARE code reads
them. Neither can be stripped: the marker is inside the root-signed job CA cert,
the job ID inside the job-CA-signed leaf.

## Provisioning

`CertBuilder` gains an `enable_job_ca` option, on by default so every newly
provisioned project (including POC) gets the job CA. It can be turned off:

```yaml
builders:
  - path: nvflare.lighter.impl.cert.CertBuilder
    args:
      enable_job_ca: false
```

Provisioning generates one additional pair signed by the root:
`job_ca.crt` / `job_ca.key`, written to the **server** startup kit only, with
the key at mode 0600. The pair is persisted in the certificate state file so
re-provisioning reuses it (an expired stored job CA is regenerated). `pathlen:0`
prevents the job CA from issuing further CAs.

Client startup kits are unchanged. Kits provisioned before this feature keep
working without re-provisioning; the server finds no job CA and job cells stay
on site certificates. Re-provisioning an existing project adds the job CA
(the root is reused from the state file, so no other cert changes).

`nvflare package` uses `PrebuiltCertBuilder` with externally signed certs and
has no root key to sign an intermediate, so those kits have no job CA.

## Runtime Issuance (SP)

A `JobCertIssuer` in the SP process loads `job_ca.crt` / `job_ca.key` from the
startup kit. The issuer is only used in secure mode; it is disabled — and the
whole feature falls back to site certificates — when the files are absent or
when the job CA has less than a minimum remaining validity (so jobs never get
certs that expire mid-run).

During job deployment (`JobRunner._deploy_job`), for each participant the
issuer generates an RSA keypair and a leaf certificate:

- the subject CN matches the CN the site's own certificate presents: for the
  SJ it is read from the server certificate in the startup kit, and for each
  CJ it is the registered client name (registration enforces that this equals
  the client cert's CN) — so whatever identity enforcement passed with site
  certs passes with job certs
- a job-ID extension identifying the job
- `notBefore` backdated a few minutes to tolerate clock skew between the
  issuing server and the sites that validate the cert seconds later
- `notAfter` = issue time + `job_cert_valid_days` (server resources config,
  default 30), clamped to the job CA's own expiry. There is no renewal, so this
  is also the maximum job duration; operators running longer jobs raise it.

The issued credential is a PEM bundle: leaf cert followed by `job_ca.crt`
(so TLS peers can build the chain to the root), plus the private key PEM.

## Distribution

**SJ (local write).** After all apps are deployed to the server workspace
(`AppDeployer` recreates the run directory), SP writes the SJ credential into
the job run directory before the SJ process is launched.

**CJ (push).** The job deploy message becomes per-site: the shared app bytes
stay a single payload reference, but each site's message carries an additional
header with that site's cert bundle and private key. The message travels over
the existing authenticated CP–SP channel (mTLS in secure mode). On the client,
the deploy processor writes the credential into the job run directory (key at
mode 0600) after the app passes signature verification and deploys.

No new channels, topics, or handshakes are introduced. The private key transits
SP→CP inside the deploy message; a CSR variant is listed under Future Work.

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

At job-process startup, the starter configers look for `job_cert/job.crt` +
`job_cert/job.key` in the run directory. When both are present, the paths are
added under `job_ssl_cert` / `job_ssl_private_key` **and replace**
`ssl_cert` / `ssl_private_key`, so nothing in the job process refers to the
site key any more. `ssl_root_cert` remains `rootCA.pem`.

- `BaseServer.create_job_cell()` and `FederatedClientBase._create_cell()` use
  the job credential for the SJ/CJ cell. The CJ also pins its server-role
  credential to the job cert; otherwise, on listener-enabled sites, the site's
  server cert would be back-filled from the startup kit and preferred by
  message-level crypto.
- The startup content-integrity check (`signature.json` kits) no longer
  requires the site private key in job processes.
- The job process never registers with the server (it receives the CP's auth
  token and signature) and never asserts site identity, so it has no other use
  for the site key.

If the job credential is absent — feature disabled, old server, non-secure
mode, simulator — the job process uses the site certs exactly as before.

## Site-Scope Rejection

A job cert is a valid `CN=<site>` certificate chaining to the project root, so
without extra checks job credentials could be replayed at site scope — most
notably to register a rogue CP (client registration accepts a caller-supplied
cert chain). All site-scope identity assertions funnel through
`IdentityVerifier.verify_common_name()` (client registration, admin login, and
the client's verification of the server), and no job cell ever legitimately
asserts identity there. Two rejections cover two distinct threats:

1. **Leaked job leaf key**: any certificate carrying the job-ID extension is
   rejected. This is keyed on the extension, not the issuer, so it holds
   regardless of which CA issued the certificate (which also keeps future HA
   setups with multiple job CAs simple).
2. **Stolen job CA key**: an attacker holding `job_ca.key` can mint a clean
   site-named leaf *without* the extension. The job CA certificate therefore
   carries a root-signed marker extension, and any presented chain containing
   a marked CA is rejected. The attacker cannot strip the marker (the job CA
   cert is signed by the root) and cannot validate without presenting it.

With both checks, compromise of the job CA key no longer escalates to site or
admin identity; its blast radius is job cells only.

## Job Binding in Cellnet

Site-scope rejection stops a job credential from acting as a site. Job binding
stops one job's credential from acting as another job's cell:

- Every TLS driver exposes the peer certificate's job-ID extension as the
  `PEER_JOB_ID` connection property next to `PEER_CN`.
- `CellIdentityResolver.require_match()` rejects a peer whose certificate is
  bound to job X unless the FQCN it claims contains the segment X
  (`site-1.X`, `server.X`, and their descendants). The check runs at the
  connection handshake (`ConnManager`) and again on the certificate exchanged
  for message-level crypto (`CredentialManager`), which is the certificate
  later used to decrypt that peer's messages.

The rule is one-directional on purpose: a job cert may only appear on job
FQCNs, but a job FQCN may still present a site cert, so older clients that
fall back to site certificates keep working.

## Site-Key Isolation by Launcher

With the job process no longer referring to the site key, launchers withhold
the `*.key` files of the startup kit (this includes `job_ca.key` on the
server). Isolation is applied only when a job credential exists for the job;
kits without a job CA keep today's behavior, with a warning, so that the
fallback to site certificates still works.

| Launcher | Startup kit delivered to the job | Job credential |
| -------- | -------------------------------- | -------------- |
| Process | same host workspace, same user — no filesystem isolation possible | run dir |
| Docker | `startup/` bound file by file, read-only, `*.key` omitted | read-write job workspace bind |
| Kubernetes | startup Secret without `*.key` | `NVFLARE_JOB_CERT` / `NVFLARE_JOB_KEY` in the per-pod credential Secret (`secretKeyRef` env) |
| Slurm (apptainer / pyxis) | keyless staged copy under the 0700 job dir, bound at `<workspace>/startup` | run dir bind |
| Slurm (`sandbox: none`) | bare host process — no isolation possible | run dir |

Kubernetes needs the environment route because the pod's bootstrap cell, which
downloads the run directory, exists before the run directory does. The job
process pops both variables and writes the credential into the run directory
(`_install_job_cert_from_env`) before creating the bootstrap cell, which then
authenticates with the job credential instead of the site key. Workspace
bundles and result uploads exclude `job_cert/` so the key travels only once.
A Secret volume at `<run dir>/job_cert` was rejected because kubelet would
create the run directory root-owned and break extraction for non-root pods.

## Compatibility

| Deployment | Behavior |
| ---------- | -------- |
| Kit without job CA (pre-feature, `enable_job_ca: false`, `nvflare package`) | No job certs issued; job cells use site certs; launchers ship site keys as before |
| Server kit with job CA, current clients (default for new provision and POC) | CJ certs pushed and used; SJ cert used; launchers withhold site keys |
| Server kit with job CA, older client release | Client ignores the unknown deploy header; CJ falls back to site certs |
| Non-secure mode / simulator | Feature inactive |
| Relays | Relays keep their provisioned site certs; CJs behind a relay use job certs |
| Sub-worker cells, Client API trainers | Unchanged: sub-workers use unauthenticated local internal links; CellPipe trainers connect with the root CA only. Neither holds a site key. |

## Future Work

1. **CSR mode.** Optional variant where CP generates the keypair and sends a
   CSR, so private keys never transit SP→CP.
2. **Renewal** for jobs that outlive `job_cert_valid_days`.
3. **Externally issued job CA** for `nvflare package` / BYO-PKI kits, so those
   servers can issue job certs too.

## Unresolved Questions

1. Should sub-worker cells and Client API trainers also carry the job
   credential, or continue on their current links?
2. HA with multiple SPs: is per-server-kit job CA acceptable, or must all SPs
   share one job CA identity?

(Message-level encryption was verified to already support certificate chains:
`CredentialManager` loads multi-cert PEMs and `SimpleCellCipher` validates
leaf + intermediates against the root, so no changes were needed there.)
