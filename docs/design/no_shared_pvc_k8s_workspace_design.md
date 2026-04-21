# Kubernetes Job Workspaces Without a Shared PVC

## Status

Accepted. This describes the production architecture for delivering NVFlare
workspaces to Kubernetes-launched job pods without a shared workspace PVC.

## Problem

The previous Kubernetes launcher mounted one workspace PVC into the parent
participant pod and every job pod it launched. That is simple but has three
structural costs:

1. Concurrent jobs can read each other's workspace files, including private
   model weights and TLS private keys.
2. The PVC becomes a single operational dependency shared by all launched
   jobs.
3. The model requires a RWX volume (EFS, Filestore, NFS) which is expensive
   or unavailable in some managed clusters.

The design goal is to make workspace ownership explicit while keeping every
NVFlare workspace semantic unchanged:

- bootstrap material is managed as Kubernetes configuration;
- each job pod receives only its own runnable workspace;
- job artifacts return to the parent explicitly;
- Kubernetes networking and RBAC remain predictable.

## Workspace Data Classes

NVFlare workspace contents fall into different trust and lifecycle classes.
Each class is delivered by the primitive that fits it.

| Path | Contents | Delivery | Writable? |
| ---- | -------- | -------- | --------- |
| `<ws>/startup/` | TLS keys, certificates, `fed_server.json` / `fed_client.json` | **Kubernetes Secret** mounted read-only | no |
| `<ws>/local/` | Site config: `resources.json`, `authorization.json`, logging/privacy config | Bundled into the per-job **workspace ZIP** | yes (extracted into `emptyDir`) |
| `<ws>/<job_id>/` at start | Job app package, job config, metadata | Same **workspace ZIP** | yes |
| `<ws>/<job_id>/` at end | Logs, checkpoints, result artifacts | **HTTPS POST** back to the parent; extracted into the parent's workspace PVC | — |

Private keys travel through a Kubernetes-managed Secret. Everything else
that changes per job — the site-local config and the per-job bundle — travels
through one authenticated HTTPS channel with a capability URL.

## Architecture

### 1. Startup via Kubernetes Secret

The launcher upserts one namespaced Secret per site before each job launch:

- `nvflare-startup-<site>`: Secret built from the parent's `startup/`
  directory. Only real keys (`fed_*.json`, `*.crt`, `*.key`, `*.pem`) are
  included; provisioning shell scripts such as `start.sh` are excluded.

The Secret is mounted read-only inside the job pod's `emptyDir` at
`<ws>/startup/`. NVFlare code in the job pod sees the normal layout.

Private key material never leaves the cluster's etcd; kubelet projects the
Secret directly into the pod.

### 2. Per-job `emptyDir` workspace

The job pod mounts:

| Volume | Purpose |
| ------ | ------- |
| `workspace-job` — `emptyDir` with `sizeLimit` (default 1 Gi) | writable `<ws>/` root |
| `startup-kit` — Secret volume (`nvflare-startup-<site>`) | mounted at `<ws>/startup/`, ro |
| `nvfldata` — study data PVC | mounted at `/var/tmp/nvflare/data`, ro |

The container's `resources.requests.ephemeral-storage` matches the
`sizeLimit` so the scheduler reserves node capacity and the kubelet can
evict on overflow.

Nothing in the job pod's workspace is shared with any other pod.

### 3. Per-job workspace bundle over HTTPS with a capability URL

At launch time the parent's `<job_id>/` directory exists on the parent's
workspace PVC (provisioning plus the server-side app-deploy step wrote it
there). The launcher:

1. Records the `(job_id, workspace_root)` pair under a fresh 256-bit
   `secrets.token_urlsafe(32)`.
2. Ensures a single in-process `WorkspaceHTTPServer` is running — started
   the first time any job is launched by this parent process, shared
   across all subsequent jobs, guarded by a startup lock so concurrent
   first-launches cannot leak two servers.
3. Injects exactly one env var into the job pod:
   `NVFL_WORKSPACE_URL=https://<parent-pod-ip>:<ephemeral-port>/<token>`.

When the job pod calls `GET /<token>`, the server streams a ZIP built on
demand from the parent's `<ws>/local/` and `<ws>/<job_id>/` directories.
The ZIP is written to a temp file and streamed to the wire, so very large
workspaces do not blow up the parent's memory footprint.

At pod shutdown the runner/worker calls `upload_results()`: zips
`<ws>/<job_id>/`, `POST /<token>`. The server streams the request body to
a temp file, validates every ZIP entry (absolute paths, `..`, and anything
outside the expected prefixes are rejected), then extracts into the
parent's workspace PVC. On successful upload the token is removed from the
registry.

Request bodies are capped at a configurable `MAX_REQUEST_BODY_SIZE`
(default 1 GiB) so a runaway or malicious client cannot exhaust parent
memory or disk.

TLS uses the parent's existing NVFlare server or client certificate. The
pod-side client skips hostname verification — the parent is addressed by
pod IP and the NVFlare certificate is not expected to contain that IP in
its SAN. Confidentiality and integrity come from TLS; access control comes
from the unguessable per-job URL.

### 4. HTTPS Server Hardening

The `WorkspaceHTTPServer` is designed to stay up across bad clients and
routine handler errors:

- The serve loop wraps every `handle_request()` in `try/except`; any
  handler exception is logged and the loop keeps running.
- Request bodies are streamed to temp files, never held whole in RAM.
- `Content-Length` is required and validated; oversized requests are
  rejected before any bytes are read.
- ZIP members are validated for path safety and for membership in the
  allowed prefixes.
- `SO_REUSEADDR` is set so a quick restart does not hit `EADDRINUSE`.
- `add_job()` checks that the server thread is alive and raises a clear
  error if not, instead of silently returning a token for a dead server.

### 5. RBAC

The helm chart grants the parent pod's service account `create`, `get`,
and `update` on `secrets` in its own namespace. No cross-namespace access
is needed. No ConfigMap permissions are needed: nothing in this design
uses a ConfigMap. Job pods need no Kubernetes API access.

### 6. Deployment Shape

- Helm chart uses `strategy: Recreate` so the parent's RWO PVC can
  reattach cleanly on rollout (RollingUpdate would produce a multi-attach
  error during rollover).
- `pod_annotations` can be set per site via the deploy config. Notably,
  AWS deployments set `karpenter.sh/do-not-disrupt: "true"` on the parent
  pod so cluster consolidation cannot evict it mid-job.
- The scheduler's `expiration_period` is raised to 300 s in the provisioning
  template so the client-side resource reservation survives image-pull
  latency before the parent actually launches the job pod.

## Security Model

| Property | How achieved |
| -------- | ------------ |
| Private keys never leave etcd | `startup/` delivered via Kubernetes Secret; kubelet projects into the pod without transiting the NVFlare network. |
| Encryption in transit | TLS 1.2+ with the parent's existing NVFlare certificate. |
| Integrity in transit | TLS, plus ZIP-member validation on upload. |
| Access control | 256-bit capability URL per job. Knowing one token reveals nothing about another. |
| Per-job isolation | Each job gets a unique `<job_id>/` and its own `emptyDir`. No shared writable volume between jobs. |
| Token leakage blast radius | Bounded to a single job: the token is scoped to one `(workspace_root, job_id)` pair and one parent process. |
| ZIP unpacking safety | Uploaded ZIPs are rejected if any entry is absolute, contains `..`, or escapes the expected prefix. |
| Upload size | Request bodies capped at `MAX_REQUEST_BODY_SIZE`; oversized requests are refused. |

Hostname verification is intentionally off on the client side. The
capability URL is the access-control primitive; adding hostname matching
against the parent's pod IP would require minting per-pod certs and gains
no security here. A compromised job pod inside the cluster already has the
token through its own env var.

## Why In-Process HTTPS Is Adequate for Production

A reasonable reviewer will ask why the data plane is not backed by cloud
object storage. Nothing the object-store design buys is missing from this
design, given how the FL lifecycle already works:

| Concern | Mitigation in this design |
| ------- | ------------------------- |
| Parent pod crash mid-job | Helm chart blocks voluntary disruption (`Recreate` + Karpenter annotation on AWS). Requests/limits prevent OOMKills. Involuntary parent crashes are already fatal to any in-flight job because the FL controller lives in the same process — object storage does not change that. |
| Lost in-memory registry on parent restart | The FL scheduler treats parent restart as a fault and re-dispatches the job; the workspace PVC still holds every `<job_id>/` directory, so re-dispatch works. The stale URL held by the killed job pod is irrelevant — the pod is reaped with its parent. |
| Thread death from handler errors | Serve loop wraps `handle_request()` in `try/except`; handler exceptions log without killing the thread. |
| DoS via oversize upload | Content-Length capped; oversize requests refused before reading any bytes. |
| Port reuse after restart | `SO_REUSEADDR` on the server socket. |
| Silent server death | `add_job()` checks `self._thread.is_alive()` and raises if the server thread is dead. |
| Double-start of the HTTP server | Startup is guarded by a lock in the launcher, so two concurrent first launches cannot leak a server. |
| Path traversal on upload | Every ZIP entry is validated before extraction. |
| TLS handshake errors from stray clients | Wrapped in the serve-loop `try/except`; NetworkPolicy is the correct layer to admit only same-namespace pods. |

What object storage would uniquely buy — durable artifacts surviving a
parent-process crash on a specific pod instance — is not free in this
deployment either: the FL job lifecycle treats a parent restart as a fault
and reschedules, resetting the job pods anyway. The in-process HTTPS data
plane matches that lifetime. Adding external blob storage would increase
the moving parts (IAM, bucket lifecycle, signed-URL minting, cross-cloud
key management) without changing the failure envelope.

## Operational Behavior

| Event | Behavior |
| ----- | -------- |
| Job pod starts before parent serves workspace | `download_workspace()` retries with bounded backoff until the URL responds. |
| Workspace download fails | Main FL process does not start; pod fails with the HTTP error. Kubernetes records the pod failure; FL scheduler re-schedules. |
| Main process succeeds, upload fails | Upload error is raised from `upload_results()`; pod exits non-zero; FL surface flags the job as artifact-upload-failed. |
| Pod is deleted during upload | Partial results are absent on the parent. Same semantics as any pod-lifetime artifact. |
| Parent pod is rescheduled | Live job pods are treated as failed by FL; they are re-launched against the new parent. No cross-restart continuity is attempted. |

## Relationship to CellNet / F3 Streaming

CellNet is the FL runtime data plane. It is not used for workspace delivery
because FL runtime bootstrap requires the workspace to exist before CellNet
is initialized. Inverting that sequence would entangle the transport with
the runtime startup order, which is a larger change than this design
needs. Workspace delivery stays on a separate, simpler HTTPS path.

## Files

| File | Purpose |
| ---- | ------- |
| `nvflare/app_opt/job_launcher/workspace_http_server.py` | HTTPS server with capability-URL routing, streaming downloads/uploads, ZIP validation, pod-side `download_workspace()` / `upload_results()`. |
| `nvflare/app_opt/job_launcher/k8s_launcher.py` | Secret upsert, emptyDir + Secret + data-PVC volume spec, ephemeral-storage defaults, image lookup via `resource_spec[site][k8s][image]`. |
| `nvflare/private/fed/app/{server/runner_process.py,client/worker_process.py}` | Call `download_workspace()` at startup and `upload_results()` at shutdown. |
| `nvflare/lighter/templates/helm/{server,client}/role.yaml` | RBAC for `secrets`. |
| `nvflare/lighter/templates/helm/{server,client}/deployment.yaml` | `Recreate` strategy, `podAnnotations` passthrough. |
| `nvflare/lighter/templates/master_template.yml` | Resource-manager `expiration_period: 300`. |
| `devops/multicloud/deploy.py`, `devops/multicloud/*-server.yaml` | Multicloud deploy harness; per-cloud `pod_annotations` (Karpenter hold on AWS), RWO `nvflws`, launcher image via job meta. |
