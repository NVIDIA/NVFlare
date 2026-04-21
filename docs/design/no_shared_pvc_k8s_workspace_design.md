# Kubernetes Job Workspaces Without a Shared PVC

## Status

Proposal.

This document describes a production-oriented design for removing the shared
workspace PVC from Kubernetes-launched NVFlare job pods. It is intentionally
written from a design point of view. The parent-hosted ephemeral HTTPS server
prototype is useful as a proof of concept, but it should not be the final
production data plane.

## Problem

The current Kubernetes launcher mounts a shared workspace PVC into parent
participant pods and dynamically launched job pods. This is simple, but it has
three important costs:

1. Concurrent jobs can see each other's workspace files.
2. The PVC becomes a shared operational dependency for all launched jobs.
3. The storage model is awkward for clusters where RWX volumes are expensive,
   unavailable, or tightly controlled.

The design goal is not merely to remove one volume mount. The goal is to make
workspace ownership explicit:

- bootstrap material should be managed as Kubernetes configuration;
- each job should receive only its own runnable workspace;
- job artifacts should be durable or explicitly reported as lost;
- Kubernetes networking and RBAC should remain predictable.

## Workspace Data Classes

NVFlare workspace contents fall into different trust and lifecycle classes.
They should not all use the same transport.

| Path | Contents | Recommended owner | Recommended delivery |
| ---- | -------- | ----------------- | -------------------- |
| `startup/` | TLS keys, certificates, `fed_server.json`, `fed_client.json` | Provisioning or Helm | Kubernetes Secret, mounted read-only |
| `local/` | Site config such as `resources.json`, authorization and resource settings | Provisioning or Helm | ConfigMap for non-secret data, Secret for sensitive data |
| `run_<job_id>/` before start | Job app package, job config, metadata | Job launcher / workspace service | Per-job workspace bundle downloaded into job-local `emptyDir` |
| `run_<job_id>/` after completion | logs, checkpoints, result artifacts | Job pod | Upload to durable artifact location before job completion is reported |

This split is the most important part of the design. It removes the shared
filesystem without forcing secrets, site configuration, code packages, and
results through one mechanism.

## Recommended Architecture

### 1. Provision Bootstrap Resources Before Job Launch

`startup/` and `local/` should be prepared by provisioning or Helm, not by the
runtime job launcher.

Recommended shape:

- create immutable, hash-named Secret and ConfigMap resources, for example
  `nvflare-startup-<site>-<hash>` and `nvflare-local-<site>-<hash>`;
- mount them read-only into both parent participant pods and launched job pods;
- reference resource names from generated Helm values or participant
  configuration;
- give the runtime launcher read/use permissions only where possible.

This keeps the launcher focused on creating job pods. It also avoids giving the
participant runtime broad permissions to create and update Secrets in the
namespace.

`local/` should not be assumed non-sensitive forever. If a site places
authorization policy or credentials under `local/`, the provisioning layer
should be able to map selected files into a Secret instead of a ConfigMap.

### 2. Give Each Job Pod a Private Writable Workspace

The job pod should mount a writable `emptyDir` at the normal NVFlare workspace
path. Before the FL process starts, the pod downloads or receives exactly one
job bundle:

```text
/var/tmp/nvflare/workspace/
  startup/          read-only Secret mount
  local/            read-only ConfigMap or Secret mount
  run_<job_id>/     writable files from the per-job bundle
```

An init container is the cleanest place for the download phase because it
preserves the normal runner and worker startup sequence: the main container
only starts after `run_<job_id>/` exists.

The job bundle should not contain `startup/` or `local/`. It should contain only
the per-job runnable state.

### 3. Use a Stable Transfer Plane, Not an Ephemeral Parent Port

The prototype starts an HTTP server inside the parent process on an ephemeral
port. That proves the bundle can be transferred, but it creates difficult
production properties:

- job pods must discover and reach a dynamic port;
- NetworkPolicy and Service configuration become job-specific;
- TLS hostname verification is hard to make clean;
- parent process restarts break uploads;
- long-running jobs can outlive the transfer server;
- upload failure can be hidden behind a successful Kubernetes pod phase.

The production design should use one of two stable transfer planes.

#### Preferred: Object Store Backed Workspaces

Use S3, GCS, Azure Blob, or a cluster-local object store such as MinIO as the
job workspace and artifact store.

Flow:

1. The launcher packages `run_<job_id>/` and writes it to
   `jobs/<job_id>/workspace.zip`.
2. The job pod init container downloads the bundle using a signed URL or
   workload identity.
3. The main container runs with `emptyDir` as its workspace.
4. On completion, a sidecar, wrapper, or runner hook uploads
   `jobs/<job_id>/artifacts.zip`.
5. The parent marks the job complete only after the artifact upload succeeds,
   or records a distinct artifact-upload failure.

This is the strongest design for durability and restart behavior. It also fits
managed Kubernetes environments well because object storage already has
identity, encryption, audit logging, retention, and lifecycle controls.

#### Fallback: Stable Workspace Transfer Service

For deployments that cannot depend on object storage, run a stable
participant-local workspace transfer service.

Recommended shape:

- expose one fixed Service port per participant, not one ephemeral port per job;
- authenticate requests with short-lived per-job credentials;
- serve only the bundle for the authenticated job;
- keep transfer state outside the parent process lifetime where possible;
- support retries and explicit artifact-upload acknowledgement;
- keep the service behind Kubernetes NetworkPolicy that admits only job pods
  for the same participant.

This service can be a sidecar next to the parent participant process or a small
Deployment scoped to the participant. A sidecar is easier to deploy, but a
separate Deployment has cleaner restart semantics.

### 4. Make Artifact Upload Part of Job Correctness

Removing the shared PVC changes the meaning of pod success. A Kubernetes
`Succeeded` phase is not enough if logs, checkpoints, or result files failed to
leave the pod.

The launcher should distinguish:

- process success and artifact upload success;
- process failure with artifact upload success;
- process success with artifact upload failure;
- pod failure before artifact upload.

At minimum, upload failure should be visible in job status and logs. For
workflows that require checkpoints or result artifacts, upload failure should
make the job fail.

### 5. Keep Shared PVC Mode During Migration

The existing shared PVC mode should remain available while the no-shared-PVC
mode matures. The new mode should be explicit in deployment configuration, for
example:

```yaml
workspace:
  mode: shared_pvc | object_store | transfer_service
```

This avoids surprising existing deployments and lets operators choose the
durability and infrastructure tradeoff that matches their cluster.

## Security Model

The security boundary should be based on Kubernetes identity and per-job
authorization, not only on opaque tokens in environment variables.

Recommended controls:

- mount bootstrap Secret and ConfigMap resources read-only;
- prefer workload identity or projected service account tokens for object store
  access;
- use short-lived signed URLs when workload identity is unavailable;
- bind each job pod credential to one `job_id`, one participant, and one
  operation class: download workspace or upload artifacts;
- enforce NetworkPolicy between job pods and transfer services;
- avoid granting job pods list/read access to arbitrary Secrets or ConfigMaps;
- avoid giving the runtime launcher Secret update permissions unless dynamic
  Secret creation is explicitly required.

HMAC integrity checks are still useful, but they should be treated as defense in
depth. If the HMAC key and bearer token are both delivered through pod
environment variables, HMAC is not an independent trust root.

## Operational Behavior

A production no-shared-PVC mode should define these behaviors explicitly:

| Event | Expected behavior |
| ----- | ----------------- |
| Parent participant process restarts | Running job pods can still finish uploading artifacts, or status clearly records that artifacts cannot be recovered |
| Job pod starts before bundle is available | Init container retries with bounded backoff |
| Workspace download fails | Main FL process does not start; pod fails with a clear reason |
| Main process succeeds but upload fails | Job status records upload failure; optional retry continues before terminal status |
| Pod is deleted | Partial artifacts are either absent or marked incomplete |
| Transfer credential expires | Retry path obtains a fresh credential or fails explicitly |

These behaviors matter more than the transport implementation. They are what
make the design operable.

## Relationship to CellNet / F3 Streaming

Using the existing CellNet/F3 channel for workspace transfer is attractive
because it would reuse NVFlare's existing secure communication path. The
sequencing is the hard part: runner and worker processes need the workspace
before normal FL communication is fully initialized.

CellNet-based workspace transfer may be worth revisiting as a larger
architectural change, but it should not block the Kubernetes design. The
Kubernetes-native design should solve bootstrap and artifact durability first.

## Phased Plan

1. Keep shared PVC mode as the default.
2. Move `startup/` and `local/` ownership into provisioning or Helm-generated
   Secret and ConfigMap resources.
3. Add an explicit no-shared-PVC mode using job-local `emptyDir` and init
   container workspace download.
4. Implement object-store backed bundle and artifact transfer as the preferred
   production mode.
5. Optionally add a stable transfer-service backend for air-gapped or
   object-store-free clusters.
6. Add job status semantics for artifact upload success and failure.
7. Deprecate shared workspace PVC only after the new mode has retry,
   observability, and upgrade behavior covered.

## Recommendation

The right direction is to eliminate the shared workspace PVC, but not by making
the parent process host a short-lived per-job HTTP server on a dynamic port.

The production design should keep the data classification from the prototype:

- `startup/` through Kubernetes Secrets;
- `local/` through ConfigMaps or Secrets;
- `run_<job_id>/` through a per-job workspace bundle into `emptyDir`;
- artifacts through an explicit upload path with durable storage and status.

The preferred data plane is object storage with Kubernetes-native identity. The
fallback is a stable workspace transfer service with fixed Service networking,
short-lived credentials, retries, and explicit artifact acknowledgement.

This gives NVFlare the security benefit of per-job isolation without replacing
the shared PVC with a more fragile runtime dependency.
