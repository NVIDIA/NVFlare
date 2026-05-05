# JobLauncher and JobHandle Design Document

## 1. Overview

NVFlare runs each federated job as an isolated execution unit — a subprocess, Docker container, or Kubernetes pod. Two abstractions govern this:

- **JobLauncherSpec** — starts a job and returns a handle.
- **JobHandleSpec** — represents the running job and provides lifecycle control (poll, wait, terminate).

The upper layers (server engine, client executor) program exclusively against these two interfaces. The concrete backend is determined entirely by **site policy**: whichever launcher is registered in the site's `resources.json` handles all jobs. The engine never inspects job metadata to pick a launcher.

```
┌──────────────────────────────────────────────────────────┐
│                    Upper Layer                           │
│     ServerEngine  /  ClientExecutor                      │
│                                                          │
│  1. get_job_launcher(job_meta, fl_ctx) → launcher        │
│  2. Build JOB_PROCESS_ARGS                               │
│  3. launcher.launch_job(job_meta, fl_ctx) → job_handle   │
│  4. job_handle.wait()  /  job_handle.terminate()         │
└──────────┬──────────────────────┬──────────┬─────────────┘
           │   BEFORE_JOB_LAUNCH  │          │
           │   (site's launcher   │          │
           │    always registers) │          │
           ▼                      ▼          ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ ProcessJob      │  │ DockerJob       │  │ K8sJob          │
│ Launcher        │  │ Launcher        │  │ Launcher        │
│ ─────────────── │  │ ─────────────── │  │ ─────────────── │
│ ProcessHandle   │  │ DockerJobHandle │  │ K8sJobHandle    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
    subprocess           Docker container       K8s Pod
```

---

## 2. Specification Layer (`nvflare/apis/job_launcher_spec.py`)

### 2.1 JobHandleSpec

Abstract base class representing a running job (`class JobHandleSpec(ABC)`).

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `terminate()` | `() -> None` | Stop the job immediately. |
| `poll()` | `() -> JobReturnCode` | Non-blocking query for current return code. Returns `UNKNOWN` while running. |
| `wait()` | `() -> None` | Block until the job finishes. |

### 2.2 JobLauncherSpec

Abstract base class for launching jobs (`class JobLauncherSpec(FLComponent, ABC)`). Extends `FLComponent` for event-system access.

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `launch_job(job_meta, fl_ctx)` | `(dict, FLContext) -> JobHandleSpec` | Start a job and return its handle. |

### 2.3 Supporting Types

**JobProcessArgs** — String constants for the keys the upper layer places in `FLContextKey.JOB_PROCESS_ARGS`. Each value is a `(flag, value)` tuple.

| Constant | Value | Used by |
|----------|-------|---------|
| `EXE_MODULE` | `"exe_module"` | Server, Client |
| `WORKSPACE` | `"workspace"` | Server, Client |
| `STARTUP_DIR` | `"startup_dir"` | Client |
| `APP_ROOT` | `"app_root"` | Server |
| `AUTH_TOKEN` | `"auth_token"` | Server, Client |
| `TOKEN_SIGNATURE` | `"auth_signature"` | Server, Client |
| `SSID` | `"ssid"` | Server, Client |
| `JOB_ID` | `"job_id"` | Server, Client |
| `CLIENT_NAME` | `"client_name"` | Client |
| `ROOT_URL` | `"root_url"` | Server |
| `PARENT_URL` | `"parent_url"` | Server, Client |
| `PARENT_CONN_SEC` | `"parent_conn_sec"` | Client |
| `SERVICE_HOST` | `"service_host"` | Server |
| `SERVICE_PORT` | `"service_port"` | Server |
| `TARGET` | `"target"` | Client |
| `SCHEME` | `"scheme"` | Client |
| `STARTUP_CONFIG_FILE` | `"startup_config_file"` | Server, Client |
| `OPTIONS` | `"options"` | Server, Client |

**JobReturnCode** — Standard exit semantics:

| Code | Value | Meaning |
|------|-------|---------|
| `SUCCESS` | 0 | Job completed successfully. |
| `EXECUTION_ERROR` | 1 | Job failed during execution. |
| `ABORTED` | 9 | Job was terminated/aborted. |
| `UNKNOWN` | 127 | Status cannot be determined (still running, or lost). |

**`add_launcher(launcher, fl_ctx)`** — Appends a launcher to the `FLContextKey.JOB_LAUNCHER` list on `fl_ctx`. Called by launchers during the `BEFORE_JOB_LAUNCH` event to register for the current job.

---

## 3. How the Upper Layer Uses Launchers

### 3.1 Launcher Selection — Site Policy, Not Job Config

The launcher is determined by which concrete `JobLauncherSpec` is registered in the site's `resources.json`. The engine calls `get_job_launcher()` from `nvflare/private/fed/utils/fed_utils.py`:

```python
def get_job_launcher(job_meta, fl_ctx) -> JobLauncherSpec:
    engine = fl_ctx.get_engine()
    with engine.new_context() as job_launcher_ctx:
        job_launcher_ctx.remove_prop(FLContextKey.JOB_LAUNCHER)
        job_launcher_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
        engine.fire_event(EventType.BEFORE_JOB_LAUNCH, job_launcher_ctx)
        job_launcher = job_launcher_ctx.get_prop(FLContextKey.JOB_LAUNCHER)
        if not (job_launcher and isinstance(job_launcher, list)):
            raise RuntimeError(f"There's no job launcher can handle this job: {job_meta}.")
    launcher = job_launcher[0]
    if not isinstance(launcher, JobLauncherSpec):
        raise RuntimeError(f"The job launcher must be JobLauncherSpec but got {type(launcher)}")
    return job_launcher[0]
```

Every registered `FLComponent` receives `BEFORE_JOB_LAUNCH`. All three launchers unconditionally call `add_launcher(self, fl_ctx)` — none inspect `job_meta` to decide whether to register. The site's `resources.json` contains exactly one launcher type; that launcher always handles every job on that site.

If a job lacks required configuration for the site's launcher (e.g. no image on a Docker/K8s site), `launch_job` raises a `RuntimeError` with a clear error message. There is no silent fallback to a different launcher.

### 3.2 Job-Level Launcher Configuration (`launcher_spec`)

`launcher_spec` is an optional top-level key in `meta.json` that carries per-launcher runtime configuration for the job — the container image, CPU/memory limits, shared memory size, etc. It is **not** used for launcher selection.

```json
{
  "launcher_spec": {
    "default": {
      "docker": { "image": "nvflare-pt:2.7" },
      "k8s":    { "image": "nvflare-pt:2.7" }
    },
    "site-1": {
      "docker": { "image": "nvflare-pt:2.7", "shm_size": "8g" },
      "k8s":    { "image": "nvflare-pt:2.7", "cpu": "4", "memory": "16Gi", "num_of_gpus": 1, "ephemeral_storage": "8Gi" }
    }
  }
}
```

Resolution via `get_job_launcher_spec(job_meta, site_name, mode)` in `nvflare/utils/job_launcher_utils.py`:

1. Merge `launcher_spec["default"][mode]` with `launcher_spec[site_name][mode]` (site wins on conflict).
2. Fall back to the nested `resource_spec[site][mode]` format for backward compatibility when `launcher_spec` is absent.

`resource_spec` (distinct from `launcher_spec`) is scheduler-facing: the scheduler reads it at job admission time to decide if the site has the required hardware. Docker and K8s both support flat `resource_spec[site]["num_of_gpus"]` as a backward-compatible GPU fallback when `num_of_gpus` is not present in `launcher_spec`.

### 3.3 Server Side (`ServerEngine`)

Location: `nvflare/private/fed/server/server_engine.py`

```
_start_runner_process(job, job_clients, snapshot, fl_ctx)
│
├─ 1. job_launcher = get_job_launcher(job.meta, fl_ctx)
│     (fires BEFORE_JOB_LAUNCH; JOB_PROCESS_ARGS not yet set)
│
├─ 2. Build job_args dict with server-specific JobProcessArgs
│
├─ 3. fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args)
│
├─ 4. job_handle = job_launcher.launch_job(job.meta, fl_ctx)
│
├─ 5. Store in run_processes[job_id]
│
└─ 6. Background thread → wait_for_complete → job_handle.wait()
```

### 3.4 Client Side (`ClientExecutor`)

Location: `nvflare/private/fed/client/client_executor.py`

```
start_app(job_id, job_meta, ...)
│
├─ 1. job_launcher = get_job_launcher(job_meta, fl_ctx)
│
├─ 2. Build job_args dict with client-specific JobProcessArgs
│
├─ 3. fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args)
│
├─ 4. job_handle = job_launcher.launch_job(job_meta, fl_ctx)
│
├─ 5. Fire EventType.AFTER_JOB_LAUNCH
│
└─ 6. Background thread → _wait_child_process_finish → job_handle.wait()
```

### 3.5 Return Code Resolution

`get_return_code()` in `fed_utils.py` uses a two-tier strategy:

1. **File-based** — Check for `FLMetaKey.PROCESS_RC_FILE` in the job's run directory. The child writes its own return code here before exiting.
2. **Handle-based** — Fall back to `job_handle.poll()`.

---

## 4. The Three Implementations

### 4.1 Process Launcher (Subprocess)

**Files:**

| File | Class |
|------|-------|
| `nvflare/app_common/job_launcher/process_launcher.py` | `ProcessHandle`, `ProcessJobLauncher` |
| `nvflare/app_common/job_launcher/server_process_launcher.py` | `ServerProcessJobLauncher` |
| `nvflare/app_common/job_launcher/client_process_launcher.py` | `ClientProcessJobLauncher` |

#### ProcessHandle

Wraps a `ProcessAdapter` that manages a `subprocess.Popen` or a PID.

| Method | Implementation |
|--------|---------------|
| `terminate()` | Delegates to `adapter.terminate()`. |
| `poll()` | Maps exit code: 0 → `SUCCESS`, 1 → `EXECUTION_ERROR`, 9 → `ABORTED`, `None` → `UNKNOWN`. |
| `wait()` | Delegates to `adapter.wait()`. |

#### ProcessJobLauncher

| Step | Action |
|------|--------|
| 1 | Copy `os.environ`. If `app_custom_folder` is non-empty, call `add_custom_dir_to_path()`. |
| 2 | Call `self.get_command(job_meta, fl_ctx)` (abstract). |
| 3 | `shlex.split(command)`, spawn via `spawn_process(argv, new_env)`. |
| 4 | Return `ProcessHandle`. |

**Event registration** — unconditionally registers:

```python
def handle_event(self, event_type, fl_ctx):
    if event_type == EventType.BEFORE_JOB_LAUNCH:
        add_launcher(self, fl_ctx)
```

**Server/Client subclasses** override `get_command()`:

- `ServerProcessJobLauncher` → `generate_server_command(fl_ctx)`
- `ClientProcessJobLauncher` → `generate_client_command(fl_ctx)`

---

### 4.2 Docker Launcher

**File:** `nvflare/app_opt/job_launcher/docker_launcher.py`

See [docker_job_launcher_design.md](docker_job_launcher_design.md) for deployment topology, networking, security posture, and operational details.

**Class hierarchy:**

```
JobHandleSpec (ABC)
  └── DockerJobHandle

JobLauncherSpec (FLComponent, ABC)
  └── DockerJobLauncher           (abstract: get_module_args)
        ├── ClientDockerJobLauncher
        └── ServerDockerJobLauncher
```

#### DockerJobHandle state mapping

| Docker status | `JobReturnCode` | Notes |
|---------------|-----------------|-------|
| `created`, `running`, `paused`, `restarting` | `UNKNOWN` | Still in progress |
| `exited` (code 0) | `SUCCESS` | |
| `exited` (code ≠ 0) | `EXECUTION_ERROR` | Reads `container.attrs["State"]["ExitCode"]` |
| `dead` | `ABORTED` | Killed externally |

Mirrors the `K8sJobHandle` `terminal_state` pattern: once `terminal_state` is set it is never cleared; all subsequent `poll()`/`wait()` calls return immediately.

#### DockerJobLauncher

Constructor parameters (set in `resources.json`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `workspace` | env `NVFL_DOCKER_WORKSPACE` | Host path of the NVFlare workspace (bind-mounted into job containers). |
| `network` | `"nvflare-network"` | Docker network. Must already exist. |
| `default_python_path` | `"/usr/local/bin/python"` | Default Python executable inside job containers. Job meta can override with `launcher_spec[site][docker].python_path`. |
| `timeout` | `30` | Seconds to wait for container to reach `running`. |
| `default_job_container_kwargs` | `{}` | Site-level `docker run` kwargs applied to every job container. Job-level `launcher_spec` wins on conflict. |
| `default_job_env` | `{}` | Site-level environment variables injected into every job container. |

Launch sequence:

| Step | Action |
|------|--------|
| 1 | Read `job_image` from `get_job_launcher_spec(job_meta, site_name, "docker").get("image")`. Raise `RuntimeError` if absent. |
| 2 | Override `PARENT_URL`: replace `localhost` with the site container name so SJ/CJ connects back via Docker DNS. |
| 3 | Build `command = [python_path, "-u", "-m", exe_module] + module_args`, using `launcher_spec[site][docker].python_path` when present and `default_python_path` otherwise. |
| 4 | Resolve `num_of_gpus` from `launcher_spec[site][docker]`, falling back to flat `resource_spec[site]` if no mode keys present. |
| 5 | Merge `default_job_container_kwargs` with job-level `launcher_spec` keys (job wins). Set `device_requests` from `num_of_gpus` if not already in merged kwargs. |
| 6 | `docker_client.containers.run(job_image, command=..., network=..., volumes=..., **merged_kwargs)`. |
| 7 | `job_handle.enter_states([DockerStatus.RUNNING])`. Return handle. |

**Event registration** — unconditional:

```python
def handle_event(self, event_type, fl_ctx):
    if event_type == EventType.BEFORE_JOB_LAUNCH:
        add_launcher(self, fl_ctx)
```

---

### 4.3 Kubernetes Launcher

**File:** `nvflare/app_opt/job_launcher/k8s_launcher.py`

**Class hierarchy:**

```
JobHandleSpec (ABC)
  └── K8sJobHandle

JobLauncherSpec (FLComponent, ABC)
  └── K8sJobLauncher              (abstract: get_module_args)
        ├── ClientK8sJobLauncher
        └── ServerK8sJobLauncher
```

#### Pod Name Sanitization

`uuid4_to_rfc1123(job_id)`: lowercase, strip non-`[a-z0-9-]` chars, prefix `"j"` if leading digit, strip trailing hyphens, truncate to 63 chars.

#### K8sJobHandle

| Method | Implementation |
|--------|---------------|
| `terminate()` | `delete_namespaced_pod(grace_period_seconds=0)`. Always sets `terminal_state = TERMINATED` regardless of outcome. |
| `poll()` | Returns `terminal_state` if set; otherwise calls `_query_state()` mapped through `JOB_RETURN_CODE_MAPPING`. |
| `wait()` | Loops `_query_state()`; sets `terminal_state` when `SUCCEEDED` or `TERMINATED`; sleeps 1s. No timeout. |
| `_query_phase()` | Calls `read_namespaced_pod`. On 404: sets `terminal_state = TERMINATED`. Returns `PodPhase.UNKNOWN` on any error. |
| `enter_states()` | Polls every 1s. Exits on: (1) stuck-in-pending → `terminate()`, (2) terminal pod phase → set `terminal_state`, (3) wall-clock timeout → `terminate()`. Returns `True` on state reached, `False` otherwise. |

Pod phase mapping:

| Pod Phase | JobState | JobReturnCode |
|-----------|----------|---------------|
| `Pending` | `STARTING` | `UNKNOWN` |
| `Running` | `RUNNING` | `UNKNOWN` |
| `Succeeded` | `SUCCEEDED` | `SUCCESS` |
| `Failed` | `TERMINATED` | `ABORTED` |
| `Unknown` | `UNKNOWN` | `UNKNOWN` |

#### K8sJobLauncher

Constructor parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `config_file_path` | required | Path to kubeconfig. Loaded lazily on first `launch_job`. |
| `workspace_pvc` | required | PVC claim name for workspace volume. |
| `study_data_pvc_file_path` | required | YAML file mapping study/dataset names to PVC claim names. Validated lazily; missing study entries skip data PVC mounts. |
| `timeout` | `None` | Wall-clock seconds for `enter_states([RUNNING])`; also `_max_stuck_count`. |
| `namespace` | `"default"` | Kubernetes namespace. |
| `pending_timeout` | `120` | Stuck-detection threshold (poll iterations) when `timeout` is `None`. |
| `default_python_path` | `"/usr/local/bin/python"` | Default Python executable in the pod command. Job meta can override with `launcher_spec[site][k8s].python_path`. |
| `ephemeral_storage` | `"1Gi"` | Default job pod workspace `emptyDir` size and `ephemeral-storage` request/limit. Job meta can override with `launcher_spec[site][k8s].ephemeral_storage`. |

Launch sequence:

| Step | Action |
|------|--------|
| 0 | Lazy init: load kubeconfig and create `CoreV1Api`. |
| 1 | Sanitize job ID via `uuid4_to_rfc1123`. Extract `site_name`, `job_image` from `get_job_launcher_spec(job_meta, site_name, "k8s")`. Raise if `WORKSPACE_OBJECT` missing. |
| 2 | Read `JOB_PROCESS_ARGS`; raise if absent or `EXE_MODULE` missing. Resolve dataset PVC mounts from `study_data_pvc_file_path` when the YAML file contains entries for the job study. |
| 3 | Build `job_config`: name, image, args from `get_module_args()`. Use `launcher_spec[site][k8s].python_path` for the pod command when present, falling back to `default_python_path`. Add the workspace `emptyDir.sizeLimit` and `resources.requests/limits["ephemeral-storage"]` from `launcher_spec[site][k8s].ephemeral_storage` when present, falling back to the launcher default. Add K8s `resources.limits` from `launcher_spec` `num_of_gpus`, `cpu`, and `memory` when present; `num_of_gpus` falls back to flat `resource_spec[site]` for backward compatibility. Missing study entries skip data PVC mounts. |
| 4 | Create `K8sJobHandle`. |
| 5 | `core_v1.create_namespaced_pod()`. On any exception: set `terminal_state = TERMINATED`, return handle. |
| 6 | `job_handle.enter_states([RUNNING])`. On any `BaseException`: `terminate()` then re-raise. |
| 7 | Return handle. |

The K8s launcher loads `study_data_pvc_file_path` once per launcher instance. Restart the parent site process to pick up hand edits to this runtime file.

**Event registration** — unconditional (site policy, not job config):

```python
def handle_event(self, event_type, fl_ctx):
    if event_type == EventType.BEFORE_JOB_LAUNCH:
        add_launcher(self, fl_ctx)
```

---

## 5. Object-Oriented Design Summary

### 5.1 Full Class Hierarchy

```
JobHandleSpec (ABC)
├── ProcessHandle          (wraps ProcessAdapter / subprocess.Popen)
├── DockerJobHandle        (wraps Docker container + terminal_state pattern)
└── K8sJobHandle           (wraps CoreV1Api + pod name + terminal_state pattern)

JobLauncherSpec (FLComponent, ABC)
├── ProcessJobLauncher     (abstract: get_command)
│   ├── ServerProcessJobLauncher
│   └── ClientProcessJobLauncher
├── DockerJobLauncher      (abstract: get_module_args)
│   ├── ServerDockerJobLauncher
│   └── ClientDockerJobLauncher
└── K8sJobLauncher         (abstract: get_module_args)
    ├── ServerK8sJobLauncher
    └── ClientK8sJobLauncher
```

### 5.2 Design Patterns

**Strategy Pattern** — Each launcher is a strategy for running jobs. The engine programs against `JobLauncherSpec`; the concrete strategy is determined by site configuration.

**Template Method Pattern** — Each base launcher implements `launch_job()` with a fixed algorithm, delegating the variable part to an abstract hook:

| Base Launcher | Abstract hook | Returns |
|---------------|---------------|---------|
| `ProcessJobLauncher` | `get_command(job_meta, fl_ctx)` | Shell command string |
| `DockerJobLauncher` | `get_module_args(job_args)` | `{flag: value}` dict |
| `K8sJobLauncher` | `get_module_args(job_id, fl_ctx)` | `{flag: value}` dict |

**Observer Pattern** — Launchers register for `BEFORE_JOB_LAUNCH` through the `FLComponent` event system. Decouples launcher registration from the engine's control flow.

---

## 6. Comparison: Process vs Docker vs Kubernetes

| Aspect | Process | Docker | Kubernetes |
|--------|---------|--------|------------|
| **Execution unit** | OS subprocess | Docker container | K8s Pod |
| **Isolation** | Shared host env | Per-job image; own env | Per-job image; pod isolation |
| **Image required** | No | Yes — `launcher_spec[site][docker][image]` | Yes — `launcher_spec[site][k8s][image]` |
| **No-image behavior** | N/A | `launch_job` raises `RuntimeError` | `launch_job` raises `RuntimeError` |
| **Workspace access** | Direct filesystem | Host bind mount → `/var/tmp/nvflare/workspace` | PersistentVolumeClaims |
| **Data access** | Direct filesystem | Optional dataset bind mounts from `study_data.yaml` | Optional dataset PVC mounts from `study_data.yaml` |
| **PARENT_URL** | `tcp://localhost:port` | Derived at runtime: `tcp://<site_name>:<port>` (Docker DNS) | Baked into comm_config at provision time |
| **GPU config** | `GPUResourceManager` → `CUDA_VISIBLE_DEVICES` | `device_requests` via `launcher_spec` or flat `resource_spec` | `nvidia.com/gpu` limit via `launcher_spec` or flat `resource_spec` |
| **Resource manager** | `GPUResourceManager` | `PassthroughResourceManager` | `PassthroughResourceManager` |
| **Start verification** | None | `enter_states(["running"])` with timeout | `enter_states([RUNNING])` with stuck detection |
| **Terminate** | `SIGTERM`/`SIGKILL` | `container.stop()` + `container.remove()` | `delete_namespaced_pod(grace_period=0)` |
| **Command format** | Shell string (`sys.executable -m ...`) | `[python, "-u", "-m", module] + args` list | `command: [python]` + `args` list in pod spec |
| **Dependencies** | stdlib only | `docker` Python SDK | `kubernetes` Python SDK + kubeconfig |
| **Typical use** | Simulator, single-machine POC | Multi-job isolation on bare metal / VM | Production cluster |

---

## 7. Resource Management

### 7.1 PassthroughResourceManager

`PassthroughResourceManager` always approves resource requests and performs no local tracking. Use with Docker or K8s launchers where the container runtime handles actual resource allocation.

| Method | Behavior |
|--------|----------|
| `check_resources()` | Always returns `(True, <token>)`. |
| `cancel_resources()` | No-op. |
| `allocate_resources()` | Returns `{}`. |
| `free_resources()` | No-op. |

### 7.2 GPUResourceManager `ignore_host` flag

`GPUResourceManager(ignore_host=True)` skips the startup check that validates declared GPUs against host hardware. Useful in K8s where the NVFlare process may run on a CPU node.

---

## 8. Sequence Diagram

```
  Engine                  fed_utils              Launcher                Handle
    │                        │                      │                      │
    │  get_job_launcher()    │                      │                      │
    │───────────────────────>│                      │                      │
    │                        │  fire BEFORE_JOB_LAUNCH                     │
    │                        │─────────────────────>│                      │
    │                        │                      │ add_launcher(self)   │
    │                        │<─────────────────────│ (always, site policy)│
    │    return launcher     │                      │                      │
    │<───────────────────────│                      │                      │
    │                        │                      │                      │
    │  launcher.launch_job(job_meta, fl_ctx)        │                      │
    │─────────────────────────────────────────────->│                      │
    │                        │                      │  create exec unit    │
    │                        │                      │─────────────────────>│
    │                        │                      │  return handle       │
    │<─────────────────────────────────────────────────────────────────────│
    │  store handle in run_processes                │                      │
    │  [background thread] handle.wait()            │                      │
    │─────────────────────────────────────────────────────────────────────>│
    │                                               │        blocks/polls  │
    │  (on abort) handle.terminate()                │                      │
    │─────────────────────────────────────────────────────────────────────>│
    │  get_return_code() → check RC file or handle.poll()                  │
    │<─────────────────────────────────────────────────────────────────────│
```

---

## 9. Configuration

Each site configures exactly one launcher in `resources.json` (or `local/resources.json` for local overrides). The configured launcher handles all jobs on that site.

### 9.1 Process Launcher (default)

```json
{
  "id": "job_launcher",
  "path": "nvflare.app_common.job_launcher.client_process_launcher.ClientProcessJobLauncher",
  "args": {}
}
```

### 9.2 Docker Launcher

```json
{
  "id": "job_launcher",
  "path": "nvflare.app_opt.job_launcher.docker_launcher.ClientDockerJobLauncher",
  "args": {
    "workspace": "/host/path/to/workspace",
    "network": "nvflare-network",
    "timeout": 30,
    "default_job_container_kwargs": {"shm_size": "8g"},
    "default_job_env": {"NCCL_P2P_DISABLE": "1"}
  }
}
```

See [docker_job_launcher_design.md](docker_job_launcher_design.md) for the full deployment guide including `start_docker.sh`, Docker network setup, and `study_data.yaml`.

### 9.3 Kubernetes Launcher

```json
{
  "id": "job_launcher",
  "path": "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
  "args": {
    "config_file_path": "/path/to/kubeconfig",
    "workspace_pvc": "nvflare-workspace-pvc",
    "study_data_pvc_file_path": "/path/to/study_data.yaml",
    "timeout": 120,
    "namespace": "nvflare"
  }
}
```

The `study_data_pvc_file_path` YAML maps study and dataset names to PVC claim names. Missing study entries mean no data PVC is mounted:

```yaml
default:
  training:
    source: default-data-pvc
    mode: ro
study-alpha:
  training:
    source: alpha-training-pvc
    mode: ro
  output:
    source: alpha-output-pvc
    mode: rw
```

For K8s, each dataset `source` is a trusted PVC claim name that is inserted into the pod manifest. The multicloud deploy tool validates generated values against `pvc_config`; manually edited runtime files are trusted site-operator input. For Docker, the same YAML shape is used but `source` is a trusted host path instead of a PVC claim name.

---

## 10. Future Improvements

1. **Unified cleanup** — Standardize cleanup policy (auto-remove on exit, configurable retention for debugging) across Docker and K8s handles.

2. **Consistent timeout policy** — The Process launcher has no start timeout. Docker and K8s launchers both call `enter_states` with a configurable timeout.

3. **Singularity/Apptainer support** — HPC environments where Docker is unavailable. Would share the `DockerJobLauncher` template method structure but replace the Docker SDK calls with CLI invocations (`singularity exec`). No daemon, runs in host network, so `PARENT_URL = localhost:port`.

4. **`ContainerJobLauncher` base class** — When a second container runtime needs support, extract a base with `_run_container()` / `_get_status()` / `_stop_container()` as the runtime-specific interface. Everything above (PARENT_URL override, PYTHONPATH, GPU, event handling) is runtime-agnostic.

5. **Observability** — Add optional `get_info()` to `JobHandleSpec` so the engine can log launcher-specific details (pod name, namespace, PID, container ID) for debugging.

6. **Orphaned container recovery** — On SP/CP restart, scan for and terminate job containers from the previous session that are still running.
