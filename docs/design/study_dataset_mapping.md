# Study Dataset Mapping

## Introduction

Site-local dataset configuration in NVFlare today uses two separate, incompatible files — Docker reads a flat JSON map and Kubernetes reads a flat YAML map. Neither supports more than one data mount per study, and each launcher has its own parsing path. This design replaces both with a single unified `local/study_data.json` managed through the `nvflare study set-dataset` / `unset-dataset` CLI commands.

This document covers:
- the unified on-disk format
- launcher behavior for Docker, Kubernetes, and process mode
- validation rules and error codes
- migration from existing files
- concrete companion code changes required in the launchers

---

### Docker

The Docker launcher runs on the host machine directly. It reads `study_data.json` from the site's startup-kit `local/` directory at job-start time. No staging step is required — the CLI write is immediately visible to the launcher.

```
CLI writes:   <startup-kit>/local/study_data.json
Launcher reads: <startup-kit>/local/study_data.json   (same file, host filesystem)
```

### Kubernetes

The K8s launcher runs inside a pod. The `local/` directory is not automatically available inside the pod; it must be staged to the persistent `etc` volume (the etc-vol PVC) before the launcher can read it. The etc-vol is mounted at `/var/tmp/nvflare/etc/` inside the launcher pod.

```
CLI writes:      <startup-kit>/local/study_data.json        (local machine)
deploy.py stages: kubectl cp <startup-kit>/local/study_data.json <pod>:/etc-vol/local/study_data.json
Launcher reads:  /var/tmp/nvflare/etc/local/study_data.json  (inside pod, via etc-vol mount)
```

The launcher is configured with `study_data_file_path: /var/tmp/nvflare/etc/local/study_data.json` in `resources.json` (replacing the current `study_data_pvc_file_path` that points to the YAML file).

If the site admin updates `study_data.json` after initial deployment, they must re-run the `kubectl cp` step (or an equivalent `nvflare study sync-dataset` helper if one is added) and restart the launcher pod for the change to take effect.

### Process mode

The `SubprocessJobLauncher` reads `study_data.json` from the startup-kit `local/` directory, the same as Docker. No staging step is required — the file is on the host filesystem and directly visible to the launcher. No mounting is performed; the operator is responsible for pre-placing data at `/data/<study>/<dataset>` on the host before running the job. Missing study entries and missing paths are not enforced at launcher level — the job proceeds and fails at runtime if data is absent.

```
CLI writes:     <startup-kit>/local/study_data.json
Launcher reads: <startup-kit>/local/study_data.json   (same file, host filesystem)
Job accesses:   /data/<study>/<dataset>               (operator must pre-place data here)
```

---

## On-Disk Format

### Current State (two separate files)

| Deployment | File | Format | Limitation |
|------------|------|--------|------------|
| Docker | `local/study_data.json` | JSON flat map | one host path per study, fixed mount at `/var/tmp/nvflare/data` |
| Kubernetes | `study_data_pvc_file_path` (YAML) | YAML flat map | one PVC per study, requires `default` key, fixed mount at `/var/tmp/nvflare/data` |

**Current Docker format:**

```json
{
  "cancer-research": "/host/data/cancer-research",
  "multiple-sclerosis": "/host/data/ms-data"
}
```

**Current Kubernetes format:**

```yaml
default: default-data-pvc
cancer-research: cancer-research-pvc
multiple-sclerosis: ms-data-pvc
```

### Unified Format (target)

All three deployment modes converge on a single `local/study_data.json`. The file is keyed by study name, then by dataset name. Each dataset entry carries a `source` field — interpreted as a host filesystem path by Docker and subprocess launchers, or as a PVC claim name by the Kubernetes launcher — plus a required `mode` field.

A site runs in one deployment mode only; the launcher interprets `source` based on its own deployment context.

**Docker / subprocess deployment:**

```json
{
  "cancer-research": {
    "training":   {"source": "/host/data/cancer-train", "mode": "ro"},
    "validation": {"source": "/host/data/cancer-valid", "mode": "ro"},
    "output":     {"source": "/host/data/output",       "mode": "rw"}
  },
  "multiple-sclerosis": {
    "training":   {"source": "/host/data/ms-train", "mode": "ro"}
  }
}
```

**Kubernetes deployment:**

```json
{
  "cancer-research": {
    "training": {"source": "cancer-train-pvc", "mode": "ro"},
    "staging":  {"source": "cancer-stage-pvc", "mode": "rw"}
  },
  "multiple-sclerosis": {
    "training": {"source": "ms-train-pvc", "mode": "ro"}
  }
}
```

**Schema summary:**

```
study_data.json
└── <study-name>: object
    └── <dataset-name>: object
        ├── source: string    # host path (Docker/subprocess) or PVC name (Kubernetes)
        └── mode: "ro" | "rw" # required
```

The `default` study key is no longer required. If no entry exists for the job's study, Docker and K8s launchers fail with a hard error and do not start the job. The `SubprocessJobLauncher` does not enforce a hard error — a missing entry results in a runtime failure in the job code (see Launcher Behavior).

---

## Launcher Behavior

| Launcher | Reads from entry | Dataset accessible at | Mode enforcement | Missing study entry |
|----------|-----------------|----------------------|------------------|---------------------|
| `ClientDockerJobLauncher` | `source` (host path) + `mode` | `/data/<study>/<dataset>` — launcher bind-mounts `source` at this path | `ro` → read-only bind mount; `rw` → read-write bind mount | **Hard error** — job not started |
| `ClientK8sJobLauncher` | `source` (PVC name) + `mode` | `/data/<study>/<dataset>` — launcher mounts PVC at this path | `ro` → `volumeMount.readOnly: true`; `rw` → `volumeMount.readOnly: false`; PVC and storage class must already support the required access mode | **Hard error** — job not started |
| `SubprocessJobLauncher` | `source` (expected host path) | `/data/<study>/<dataset>` — operator must pre-place data at this path | N/A — mode is stored but not enforced | Not enforced — job proceeds; runtime failure if data is absent |

**Study lookup:** Docker and K8s launchers look up the job's study name in `study_data.json` and raise `STUDY_NOT_FOUND` if no entry exists — the job is not started. The `SubprocessJobLauncher` reads `study_data.json` as a reference but does not enforce a hard error; a missing entry results in a runtime failure when job code tries to access the expected data path.

**Missing field at job start:** If any launcher finds a dataset entry with no `source` key, it raises `BACKEND_FIELD_MISSING` and fails before starting the job.

**Data path not found:** For Docker deployments, the launcher verifies that `source` exists on the host filesystem before starting the job and raises `DATA_PATH_NOT_FOUND` if absent. For subprocess, this is not enforced at launcher level — a missing path results in a runtime failure in the job code. Kubernetes launchers do not perform this check; PVC existence is validated by the cluster at pod scheduling time.

### Dataset Isolation

Each job sees only its own study's dataset paths. The launcher reads only the entries for the job's study from `study_data.json` and constructs mounts (Docker/K8s) or path bindings (subprocess) exclusively from those. Datasets belonging to other studies are never exposed to the job.

- **Automatic** — study name from job metadata is the lookup key; no per-job configuration needed
- **Docker/K8s** — container filesystem contains only explicitly mounted paths; host paths are not otherwise accessible
- **Subprocess** — data must be pre-placed at `/data/<study>/<dataset>` on the host by the operator; the launcher does not validate path existence — a missing path results in a runtime failure in the job code

---

## Validation Rules

All validation runs before any file is written. A failed validation returns a structured error and exits with code 4.

| Input | Rule | Error code |
|-------|------|------------|
| `<study>` | Must match `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$`. Regex excludes `/`, `.`, and special characters — safe for direct use as a filesystem path component. | `INVALID_STUDY_NAME` |
| `<dataset>` | Must match `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$`. Same path-safety guarantee — used as a path component in `/data/<study>/<dataset>`. | `INVALID_DATASET` |
| `--pvc` | Must match `^[a-z0-9](?:[a-z0-9-]{0,251}[a-z0-9])?$` (Kubernetes resource name rules) | `INVALID_DATASET` |
| `--data-path` | Accepted declaratively; no path-traversal or existence check at CLI time — `..`-containing paths are not rejected | — |
| `--mode` | Must be exactly `ro` or `rw` | `INVALID_MODE` |
| `--data-path` / `--pvc` | Exactly one must be provided | `MISSING_REQUIRED_FLAG` |
| `--startup-kit` | Path must exist and be a valid startup-kit directory | exit 4 |

## Error Codes

| Error code | Exit | Meaning |
|------------|------|---------|
| `INVALID_STUDY_NAME` | 4 | Study name fails the name-validation regex |
| `INVALID_DATASET` | 4 | Dataset name fails the name-validation regex, or `--pvc` fails the Kubernetes resource name rules |
| `INVALID_MODE` | 4 | `--mode` value is not `ro` or `rw` |
| `MISSING_REQUIRED_FLAG` | 4 | Neither `--data-path` nor `--pvc` was provided to `set-dataset` |
| `DATA_PATH_NOT_FOUND` | 1 | Docker launcher finds a `source` entry but the configured host path does not exist on the host filesystem at job-start time. Not raised by the subprocess launcher — path absence results in a runtime failure in the job code. |
| `BACKEND_FIELD_MISSING` | 1 | Launcher finds the study/dataset entry but the `source` key is absent |
| `STUDY_NOT_FOUND` | 1 | Docker or K8s launcher cannot find an entry for the job's study in `study_data.json` at job-start time — job is not started. Not raised by `SubprocessJobLauncher`; a missing entry results in a runtime failure in the job code. |
| `NOT_YET_IMPLEMENTED` | 1 | Dataset commands are gated pending the companion launcher change; write behavior is disabled until the unified format is implemented |

---

## Migration from Current Files

### Docker (`local/study_data.json`)

Convert each flat entry to the nested format. Choose a dataset name such as `"data"` or something semantically meaningful.

```json
// Before
{ "cancer-research": "/host/data/cancer" }

// After
{
  "cancer-research": {
    "data": {"source": "/host/data/cancer", "mode": "ro"}
  }
}
```

Update any job code that reads from `/var/tmp/nvflare/data` to use `/data/<study>/<dataset-name>`. If job code is not updated, it will read from a path that does not exist in the container and raise a `DATA_PATH_NOT_FOUND` or file-not-found error at runtime.

### Kubernetes (`study_data_pvc.yaml`)

Convert the flat YAML to the nested JSON format. The `default` key is no longer required.

```yaml
# Before (YAML)
default: default-data-pvc
cancer-research: cancer-research-pvc
```

```json
// After (JSON)
{
  "cancer-research": {
    "data": {"source": "cancer-research-pvc", "mode": "ro"}
  }
}
```

Update job code paths in the same way as Docker.

### Process mode

Add entries for each study/dataset using `--data-path`. Set `source` to `/data/<study>/<dataset>` — the path where the operator will pre-place data on the host. The subprocess launcher reads this entry as a reference only; it does not validate path existence at job-start — a missing path produces a runtime failure in the job code.

```json
// Before: no entry required
// After
{
  "cancer-research": {
    "data": {"source": "/data/cancer-research/data", "mode": "ro"}
  }
}
```

Ensure data is available at `/data/cancer-research/data` on the host before submitting the job.

---

## Companion Code Changes

The following code changes are required to implement the unified format.

### `nvflare/app_opt/job_launcher/k8s_launcher.py`

- Replace `study_data_pvc_file_path` argument with `study_data_file_path` pointing to the unified JSON
- Replace `yaml.safe_load(...)` with `json.load(...)`
- Replace the flat `study → pvc` lookup with the nested `study → dataset → {source, mode}` lookup; interpret `source` as a PVC name
- Iterate over dataset entries for the active study; generate one `volume_list` entry and one `volume_mount_list` entry per dataset
- Set `volumeMount.readOnly` from the `mode` field (`ro` → `True`, `rw` → `False`)
- Remove the `default` YAML key assumption; fail hard if no entry exists for the job's study

### `nvflare/app_opt/job_launcher/subprocess_launcher.py`

- Add `study_data_file_path` argument (same as Docker launcher)
- At job-start, load `study_data.json` and look up the job's study entry as a reference; no hard error is raised for a missing entry — enforcement is not possible without a container boundary
- No mount is performed — operator is responsible for pre-placing data at `/data/<study>/<dataset>` before submitting the job; runtime failure in job code is the signal for a missing path

### `nvflare/app_opt/job_launcher/docker_launcher.py`

- Replace the flat `study → host_path` lookup with the nested `study → dataset → {source, mode}` lookup; interpret `source` as a host path
- Iterate over dataset entries; generate one bind-mount per dataset
- Mount each dataset at `/data/<study>/<dataset>` with the access mode from the `mode` field
- Fail hard if no entry exists for the job's study

### `devops/multicloud/deploy.py`

- Stop writing `etc/study_data_pvc.yaml`
- Write the unified `study_data.json` instead
- Patch `resources.json` to point `ClientK8sJobLauncher` at the new file path
- Update `kubectl cp` staging from `study_data_pvc.yaml` to `study_data.json`

### Tests

- `tests/unit_test/app_opt/job_launcher/subprocess_launcher_test.py` — add study lookup and `BACKEND_FIELD_MISSING` tests; verify no mount is attempted and no hard error is raised for a missing study entry or missing path
- `tests/unit_test/app_opt/job_launcher/k8s_launcher_test.py` — replace YAML fixture with JSON; add multi-dataset mount tests; add `volumeMount.readOnly` mode tests
- `tests/unit_test/app_opt/job_launcher/docker_launcher_test.py` — add multi-dataset mount tests; add mode tests
- `tests/unit_test/tools/multicloud_deploy_test.py` — replace `study_data_pvc.yaml` assertions with `study_data.json`
- `tests/unit_test/tool/study/` — `set-dataset` / `unset-dataset` CLI handler tests are specified in `docs/design/multistudy_cli.md`
