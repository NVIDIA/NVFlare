# Study Runtime Configuration

FLARE-2972

## Overview

A single site-owned file, `local/study_runtime.yaml`, defines everything a study gets at runtime: data mounts, env
vars, secret-backed env vars and file mounts, an optional pod template, and (reserved) provider-prepared datasets
such as Databricks.

Trust boundary:

- The job selects only the study; study membership is enforced by the server before the job reaches a site.
- The site owns the study-to-runtime-resource mapping in `local/`.
- External governors (Kubernetes Secrets, Unity Catalog, database roles) authorize the study identity — never the
  human user, never the job package.

Goals: secrets as references only (never values in `local/` files); zero-configuration discovery (no launcher
arguments, no `resources.json` edits); Docker and Kubernetes conceptually aligned; FLARE-owned launch behavior
untouchable; datasets extensible by `type` so new providers need no format change; v1 deployments keep working.

Non-goals: job-requested Secrets; job-side runtime wiring; replacing `launcher_spec`/`resource_spec`; changing the
pod template merge machinery; implementing the Databricks provider (schema shape only).

## v1 Schema (shipped in 2.8 — frozen)

```yaml
# local/study_data.yaml
default:
  data:
    source: nvfldata     # Docker: host path; K8s: PVC claim name
    mode: ro             # ro | rw
```

`study -> dataset -> {source, mode}`, mounted at `/data/<study>/<dataset>`. File and parser stay exactly as-is.
The unreleased `study_job_spec_file_path` pod-template mechanism (PR #4804) is removed and replaced by v2
`pod_template`.

## v2 Schema: `local/study_runtime.yaml`

- Auto-discovered at a fixed workspace-relative path by both launchers. No launcher argument, no deploy-config key.
- Filename = format: `study_runtime.yaml` is v2 strict, `study_data.yaml` is v1 frozen. No content sniffing.
- Job side unchanged: the job still submits only `{"study": "..."}`.

```yaml
format_version: 2

studies:
  lung-cancer:
    container:
      name: lung-cancer-job              # optional stable main-container name (RFC 1123 label)

    pod_template: lung_cancer_pod.yaml   # K8s only; path relative to local/, or inline pod mapping

    datasets:
      reference:                         # type defaults to "mount" == v1 semantics
        source: nvfldata
        mode: ro

    env:
      DB_HOST: postgres.svc
      DB_NAME: lung_cancer

    secret_env:
      DB_USER: {source: study-db, key: username}
      DB_PASSWORD: {source: study-db, key: password}

    secret_mounts:
      db-ca:
        source: study-db-ca
        mount_path: /var/run/nvflare/secrets/db-ca
        mode: ro
        items:                           # optional; omitted = full Secret projection
          ca.crt: ca.crt
```

The schema is common across launchers but values are launcher-specific (`source`: PVC claim vs host path;
`secret_env.*.source`: Secret name vs env var; `pod_template`: K8s-only). A file is authored per deployment, not
portable across launcher types.

Inline `pod_template` example — pinning a study to H100 nodes needs only this file:

```yaml
format_version: 2

studies:
  h100-study:
    datasets:
      training: {source: h100-data-pvc, mode: ro}
    pod_template:                        # inline; a string value means a file path instead
      spec:
        nodeSelector:
          nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
        tolerations:
          - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
```

## Future Schema (reserved — rejected by the FLARE-2972 parser)

```yaml
studies:
  lung-cancer:
    datasets:
      training:
        type: databricks
        workspace: hospital-databricks
        manifest: medical.lung_cancer.approved_dicom_manifest
        volume: medical.lung_cancer.dicom
        auth:
          method: k8s_service_account   # or site_signed
          service_account: sa-lung-cancer
          audience: databricks
        mode: ro
        delivery: mount                 # vended cloud credentials; materialize is a future mode
```

No indirection layer: provider fields live inline on the dataset entry, repeated where needed, and contain no
secret values. `type: databricks` fails today with "not yet supported".

## Dataset Types

| Type | Status | Meaning |
| --- | --- | --- |
| `mount` (default) | FLARE-2972 | Mount an existing PVC (K8s) or host path (Docker). v1 semantics. |
| `databricks` | Future | Launcher acquires a study credential, vends downscoped storage credentials, mounts the approved volume via CSI. |

Every dataset type surfaces at `/data/<study>/<dataset>`; trainer code is identical regardless of source. The
discriminator is the single `type` field; provider-specific fields are validated per type.

## Runtime Interpretation

Kubernetes:

- `type: mount` dataset: `source` is a PVC claim name.
- `secret_env.*`: emitted as `valueFrom.secretKeyRef` (`source` = Secret name, `key` = key) on the main container.
- `secret_mounts.*`: Secret projected as a read-only volume at `mount_path`.
- `container.name`: stable per-study main-container name (RFC 1123 validated). Default stays the per-job generated
  name; templates may instead mark the main container with the `nvflare_job` sentinel.
- `pod_template`: inline mapping or path relative to `local/`; becomes the study's base manifest.

Docker:

- `type: mount` dataset: `source` is a host path.
- `env`: merged into the container environment (site values win over `default_job_env`).
- `secret_env.*.source`: parent-process env pass-through (`key` unused); pluggable site secret backend later.
- `secret_mounts.*.source`: host path bind-mounted read-only at `mount_path`. `items` is a hard error
  (Kubernetes-only Secret projection) — point `source` at a directory containing only the intended files.
- `container.name`: ignored. `pod_template`: hard error (K8s-only). `type: databricks`: deferred.

## Pod Templates

Merge order (later wins):

```text
1. Base manifest        built-in pod  OR  the study's pod_template
2. Typed v2 additions   dataset volumes/mounts, env, secret_env, secret_mounts
3. FLARE-owned fields   image, command/args, workspace + startup volumes,
                        transfer env, resources, restartPolicy: Never
```

- Typed entries merge by name onto the main container, located via `container.name` or the `nvflare_job` sentinel.
- FLARE-owned fields sit on top of both layers; neither template nor typed schema can break launch invariants.
- Guardrail: if typed entries (including an explicit `container.name`) are configured and a template has multiple
  containers, the main container must be identifiable — otherwise launch fails. (The existing `containers[0]`
  fallback would silently give a sidecar the study's `secret_env`.)
- The template is the escape hatch for cluster-shaped needs (service accounts, tolerations, affinity, sidecars,
  admission annotations); the typed schema covers the portable rest. The built-in default manifest stays in code —
  it carries the launch invariants; there is no editable default template file.
- Per-study service accounts (`pod_template.spec.serviceAccountName`) work on today's machinery — the client parent
  and job pod already run under different SAs (Helm SA vs namespace default), and the job pod needs zero RBAC: it
  never calls the K8s API (CP↔CJ traffic is cellnet; Secrets and volumes are mounted by the kubelet; image pulls
  use the launcher's explicit `imagePullSecrets`). Caveats: the SA must pre-exist (pod creation fails fast
  otherwise); do not rely on SA-attached imagePullSecrets — only the launcher-configured ones are carried; job pods
  should set `automountServiceAccountToken: false`, since the token is never used and the Databricks design mints
  tokens launcher-side via TokenRequest instead.

## Future: Per-Study Default Job Image

The job image is currently the one runtime resource the site does not own: it must come from job `meta.json`, and
`pod_template` deliberately cannot set the main container image (the FLARE-owned overlay replaces it). Follow-up
direction, recorded from PR review:

- `container.image` next to the existing `container.name` — a typed key, so it behaves identically on Docker and
  Kubernetes (unlike `pod_template`).
- Resolution: job-supplied image → study `container.image` → error. Only the "or raise" step in each launcher
  changes; the merge order is untouched.
- Job-supplied image wins by default; an enforce/pin mode for sites that forbid job-supplied images can come later.
- A site-supplied image is site-trusted content and is not BYOC-gated the way job-supplied images are.
- Re-read-per-launch means a site can roll a study to a new image version without restarting the parent.

(The related Docker cleanup — `image` in `default_job_container_kwargs` passed init validation but failed every
launch with a duplicate `containers.run` argument — is already fixed: init now rejects it, while
`docker_spec["image"]` remains the legitimate job image selector.)

## Parsing, Compatibility, Migration

```text
local/study_runtime.yaml present  → v2 strict parser (format_version: 2 required)
local/study_data.yaml only        → v1 frozen parser, behavior unchanged
both present (or v2 + legacy launcher args) → hard error
```

- `format_version: 2` required — the gate for future format evolution.
- Hard errors: unknown top-level keys, unknown per-study keys, unknown dataset `type`, a key in both `env` and
  `secret_env`, an `env` or `secret_env` name that is launcher-owned (`PYTHONPATH`, the workspace-transfer
  variables), v1/v2 coexistence.
- Missing referenced Secrets are not validated pre-launch (the launcher holds no Secret-read RBAC); the pending-pod
  failure classification already fails the job fast (`CreateContainerConfigError`, `FailedMount`).
- Migration is wholesale: move all studies to `study_runtime.yaml`, delete `study_data.yaml` and any
  `study_data_pvc_file_path` launcher arg.
- Workspace transfer: job workspace downloads exclude `study_runtime.yaml` plus any path-form `pod_template` files,
  read from this one well-known YAML. The `resources.json`-scraping exclusion logic is deleted.
- `nvflare deploy prepare` generates a commented `study_runtime.yaml` template in new kits (skipped when a legacy
  `study_data.yaml` exists) and emits the `study_data_pvc_file_path` launcher argument only for legacy kits. POC
  Docker provisioning writes its data mapping into `study_runtime.yaml`. The file travels with the startup kit's
  `local/`.

## Future: Databricks Provider

Identity: one Kubernetes service account per study is the study identity; the Databricks federation policy pins it
as subject. The launcher mints a short-lived, audience-bound SA token via the TokenRequest API and performs the
workload-identity exchange itself — no token enters the job pod. The SA needs zero RBAC; the launcher needs
`serviceaccounts/token` create. Alternative `auth.method: site_signed`: the FLARE client signs a short-lived JWT
with a hospital-held key (on-prem issuers, future Docker); federation policies can embed a static JWKS.

Delivery `mount`: a Unity Catalog volume is a governed object-storage prefix. Before pod creation the launcher
queries the approved manifest and calls the Temporary Volume Credentials API for a downscoped cloud credential
(read-only, path-scoped, ~1h), writes it into a per-job Secret, and emits a CSI volume (`nodePublishSecretRef`).
Renewal: the launcher re-vends and updates the Secret; the CSI driver must accept vended session credentials from a
Secret and support rotation, or job duration must fit the credential TTL. Requires the hospital's
`EXTERNAL USE SCHEMA` grant (vending is Public Preview today). The manifest result (labels, splits, cohort) is
delivered as a small file next to the mounted data. Mount granularity is the volume, so volume boundary = study
boundary is a hard rule. `delivery: materialize` (Files API download via init container; enables Docker) is
reserved for workspaces where vending or a qualifying CSI driver is unavailable.

Security invariants: the Databricks OAuth token and study assertion exist only in the launcher process. The only
pod-visible credential is the vended storage credential — read-only, path-scoped, short-lived, unable to call SQL
or workspace APIs, consumed by the CSI driver rather than the training container. FLARE records audit events for
resolution, token acquisition, manifest query, vend, renewal, and cleanup; per-object audit lives in cloud-storage
access logs (hospital-enabled prerequisite).

Interfaces: `StudyCredentialProvider` (study assertion → provider OAuth token) and `StudyDataProvider` (manifest
query + credential vending/mount config, or future materialization), shipped as an optional plugin package.

## Future: Database Sources (Postgres)

Live SQL sources are connections, not datasets — no provider seam:

- Already covered by v2: `env` (DB_HOST, DB_NAME) + `secret_env` (DB_USER, DB_PASSWORD) + `secret_mounts` (CA).
- Downscoping is native: a per-study Postgres role with `SELECT` on approved views bounds what SQL can touch.
- Credential-in-trainer is acceptable: the database is hospital-internal and network-scoped, unlike a SaaS token.
- RDS/Cloud SQL IAM auth can later replace static passwords — same `auth` slot, no schema change.

```text
Trainer queries at training time → connection config + per-study DB role
Trainer reads a fixed cohort     → dataset provider (future; requires materialize)
```

## Implementation Order

1. FLARE-2972: strict v2 parser; `env`, `secret_env`, `secret_mounts`, `container.name`, `pod_template`
   (inline + path); auto-discovery + v1-coexistence hard error; remove `study_job_spec_file_path`, its map file,
   and the `resources.json`-scraping transfer excludes; Docker + K8s emission; merge order + main-container
   guardrail.
2. `StudyCredentialProvider` / `StudyDataProvider` interfaces.
3. K8s mount wiring: per-job vended-credential Secret, CSI volume emission, per-cloud driver qualification,
   manifest delivery.
4. Databricks provider plugin with `delivery: mount`: federation exchange, manifest query, vending, renewal, audit,
   cleanup on completion/cancel/pod failure.
5. Future: `delivery: materialize` + init-container hook (enables Docker), SQL snapshot providers
   (`type: postgres`).
