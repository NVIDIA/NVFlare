# Kubernetes Job Workspaces Without a Shared PVC

This document describes the production no-shared-PVC workspace model used by
the Kubernetes launcher. The data plane is CellNet/F3 streaming over the
existing NVFlare communication stack. The parent participant does not expose a
separate HTTP service for workspace transfer, and dynamically launched job pods
do not mount the shared workspace PVC.

## Goal

The Kubernetes launcher removes the shared workspace PVC from dynamically
launched job pods so that each job gets its own private writable workspace
while still receiving the files it needs to run and returning its artifacts to
the parent participant.

This changes the workspace model in three important ways:

1. Job pods no longer share one writable workspace filesystem.
2. Workspace bootstrap and artifact transfer use CellNet/F3 streaming.
3. The job pod writes only to its own `emptyDir` workspace.

## Workspace Layout

The launched job pod uses this layout under the normal NVFlare workspace root:

```text
/var/tmp/nvflare/workspace/
  startup/          read-only Secret mount
  local/            downloaded as part of the workspace bundle
  <job_id>/         downloaded as part of the workspace bundle
```

`startup/` is not expected to already exist inside the job pod's `emptyDir`.
Before launching the pod, the launcher creates or updates a per-site Kubernetes
Secret from that participant site's startup-kit directory and mounts that
Secret at `/var/tmp/nvflare/workspace/startup`.

`startup/` and `local/` are handled differently:

| Path | Delivery | Notes |
| ---- | -------- | ----- |
| `startup/` | Kubernetes Secret mounted read-only | Created or updated by the launcher from the site startup kit |
| `local/` | Included in the per-job workspace bundle | Bundled whole so local config, resources, and custom code remain available |
| `<job_id>/` | Included in the per-job workspace bundle | Private runnable job state for exactly one job |

The job pod still mounts the study-data PVC separately at `/var/tmp/nvflare/data`.
The shared workspace PVC is not mounted into the job pod.

## Kubernetes Resources

For each launched job pod, the launcher creates a pod manifest with:

- an `emptyDir` mounted at `/var/tmp/nvflare/workspace`
- a read-only Secret mount at `/var/tmp/nvflare/workspace/startup`
- the study-data PVC mounted read-only at `/var/tmp/nvflare/data`

The launcher also creates or updates a startup Secret for the participant site.
That Secret contains the startup-kit files needed by the launched process, such
as certificates, keys, and JSON config files, and those files appear in the
pod under `/var/tmp/nvflare/workspace/startup` via the Secret mount.

## Transfer Architecture

Workspace transfer is implemented on top of the parent participant's existing
CellNet cell.

The parent process creates one `WorkspaceTransferManager` per parent CellNet
cell. That manager:

- registers request handlers for workspace download and results upload
- keeps per-job transfer state keyed by `job_id`
- packages the workspace bundle on demand
- publishes bundle and artifact files through the F3 file downloader

Each launched job uses a short-lived bootstrap CellNet child cell. The bootstrap
cell exists only to:

1. request the workspace bundle before normal FL startup
2. upload final job results during shutdown

The bootstrap cell does not use a separate trust model. It reuses the same
parent connection settings and CellNet authentication headers that the launched
worker or runner already uses to talk to its parent. Those auth headers are
installed before the bootstrap cell starts so the initial CellNet registration
handshake is accepted by the parent authenticator.

The job pod learns the parent owner FQCN through the
`NVFL_WORKSPACE_OWNER_FQCN` environment variable set by the launcher. The
launcher also injects a random per-job transfer capability in
`NVFL_WORKSPACE_TRANSFER_TOKEN`.

## Launch Flow

When the Kubernetes launcher starts a job, it performs these steps:

1. Resolve the job image and launcher-specific resource settings.
2. Reuse or create the `WorkspaceTransferManager` on the parent CellNet cell.
3. Register the job with the transfer manager using the raw `job_id` and the
   parent workspace root.
4. Create or update the startup Secret for the participant site.
5. Launch the job pod with:
   - `emptyDir` workspace
   - read-only startup Secret mount
   - read-only study-data PVC mount
   - `NVFL_WORKSPACE_OWNER_FQCN` in the environment
   - `NVFL_WORKSPACE_TRANSFER_TOKEN` in the environment

No shared workspace PVC is mounted into the launched job pod.

## Workspace Download Flow

Both the client worker process and the server runner process download the
workspace before continuing with normal FL startup.

The download sequence is:

1. The launched process reads `NVFL_WORKSPACE_OWNER_FQCN` and the per-job
   `NVFL_WORKSPACE_TRANSFER_TOKEN`.
2. It creates a short-lived bootstrap child cell using the startup kit and the
   existing parent connection settings. The bootstrap FQCN is
   `<owner_fqcn>.ws_transfer_<job_id>`. When the child process is a client
   worker, the bootstrap cell reuses that worker's `client_name`, auth token,
   token signature, and `ssid`. When the child process is a server runner, the
   bootstrap cell uses the same server-job auth identity as the main runner.
3. It sends `prepare_download` to the parent transfer manager with the `job_id`
   and the transfer token.
4. The parent validates that the transfer token matches the one issued for that
   job. The bootstrap FQCN is used for CellNet routing and to tell the parent
   which caller to download from later; it is not part of the authorization
   boundary.
5. The parent creates a temporary zip bundle containing:
   - `local/`
   - `<job_id>/`
6. The parent exposes that zip through the F3 file downloader and returns the
   file reference plus checksum metadata.
7. The child downloads the file through CellNet/F3, validates the checksum,
   validates all zip members, and extracts the bundle into the local workspace.

After extraction, the worker or runner continues with normal NVFlare startup
using the downloaded workspace files plus the mounted `startup/` Secret.

## Results Upload Flow

Both the client worker process and the server runner process attempt results
upload in their shutdown path.

The upload sequence is:

1. The process zips only `<job_id>/` from its local workspace.
2. It creates a short-lived bootstrap child cell.
3. It publishes the zip through the F3 file downloader.
4. It sends `publish_results` to the parent transfer manager with the `job_id`,
   transfer token, file reference, and checksum.
5. The parent validates the transfer token, then uses the request origin as the
   CellNet source address to download the artifact zip from the caller through
   CellNet/F3. As with download, the token is the authorization factor.
6. The parent validates the checksum and validates that every zip member stays
   under the exact `<job_id>/` directory.
7. The parent extracts the uploaded results back into the parent workspace and
   removes the per-job transfer state.

Upload is performed from the process shutdown path. If upload fails, the worker
or runner logs a warning with the failure details.

## Security Model

The implementation relies on the existing NVFlare secure communication stack and
adds job-scoped authorization on top of it.

Cross-job isolation comes from both storage layout and transfer authorization:

1. A launched job pod does not mount the shared workspace PVC at all. Its
   writable workspace is a fresh `emptyDir`, so there is no filesystem path in
   the pod that exposes another job's workspace data.
2. The only workspace content delivered into that `emptyDir` is `local/` plus
   the exact `<job_id>/` directory for that launched job. The parent does not
   package sibling job directories into the download bundle.
3. Each launched job receives a random transfer token generated by the parent
   transfer manager for that one job. The parent stores that token and rejects
   download or upload requests whose token does not match.
4. The bootstrap cell name includes the `job_id`, but that name is only used
   for CellNet routing and operational clarity. It is not a secret and is not
   treated as part of the authorization boundary.
5. On results upload, the parent validates that every zip member stays under
   the exact `<job_id>/` subtree before extracting. Even if job code creates a
   crafted archive, it cannot overwrite `startup/`, `local/`, or another job's
   directory in the parent workspace.

### Threat Model Note

`NVFL_WORKSPACE_TRANSFER_TOKEN` is a replayable bearer credential. If another
pod or process obtains the token, it can replay workspace download or upload
requests until the parent removes that job's transfer record.

This design therefore assumes that transfer-token disclosure is out of scope.
In particular, the launcher and runtime must treat the token as sensitive and
must not write it to logs, manifest dumps, debugging output, or other support
artifacts.

The key protections are:

- `startup/` is mounted read-only from a Kubernetes Secret
- secure mode bootstrap cells use `rootCA.pem` plus the available startup cert
  and key pair
- the launcher passes the parent listener's connection-security setting into
  the child process args, and the bootstrap cell installs its CellNet auth
  headers before `cell.start()` so the parent accepts the initial registration
- each job gets a random per-job transfer token (24 bytes, urlsafe-encoded)
- the parent rejects requests whose transfer token does not match
- the transfer token must never be logged or emitted in pod manifest debug
  output
- downloaded and uploaded archives are checksum-validated
- all zip members are path-validated before extraction
- uploaded result archives are restricted to the exact `<job_id>/` subtree

This prevents one job from overwriting arbitrary paths in the parent workspace
through a crafted archive upload.

## Cleanup and Lifecycle

The transfer manager keeps only active per-job transfer records.

Cleanup happens in these places:

- temporary download bundles are removed when the F3 download transaction ends
- uploaded artifact temp files are removed after upload processing
- job transfer state is removed when:
  - pod launch fails
  - the job handle reaches a terminal state
  - results upload succeeds

The transfer manager is reused per parent CellNet cell so multiple jobs do not
register duplicate workspace-transfer handlers on the same parent cell.
