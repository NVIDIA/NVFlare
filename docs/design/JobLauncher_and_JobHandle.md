# JobLauncher and JobHandle Design Document

## 1. Overview

NVFlare runs each federated job as an isolated execution unit -- a subprocess, Docker container, or Kubernetes pod. Two abstractions govern this:

- **JobLauncherSpec** -- starts a job and returns a handle.
- **JobHandleSpec** -- represents the running job and provides lifecycle control (poll, wait, terminate).

The upper layers (server engine, client executor) program exclusively against these two interfaces. The concrete backend is selected at runtime through an event-based mechanism, so the engine never imports or names a specific launcher type.

```
┌──────────────────────────────────────────────────────────┐
│                    Upper Layer                           │
│     ServerEngine  /  ClientExecutor                      │
│                                                          │
│  1. Build JOB_PROCESS_ARGS                               │
│  2. get_job_launcher(job_meta, fl_ctx) → launcher        │
│  3. launcher.launch_job(job_meta, fl_ctx) → job_handle   │
│  4. job_handle.wait()  /  job_handle.terminate()         │
└──────────┬──────────────────────┬────────────────────────┘
           │   BEFORE_JOB_LAUNCH  │
           │   event selects one  │
           ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ ProcessJob      │  │ DockerJob       │  │ K8sJob          │
│ Launcher        │  │ Launcher        │  │ Launcher        │
│ ─────────────── │  │ ─────────────── │  │ ─────────────── │
│ ProcessHandle   │  │ DockerJobHandle │  │ K8sJobHandle    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
    subprocess           container              pod
```

---

## 2. Specification Layer (`nvflare/apis/job_launcher_spec.py`)

### 2.1 JobHandleSpec

Abstract base class representing a running job. All methods are `@abstractmethod`.

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `terminate()` | `() -> None` | Stop the job immediately. |
| `poll()` | `() -> JobReturnCode` | Non-blocking query for the job's current return code. Returns `UNKNOWN` while still running. |
| `wait()` | `() -> None` | Block until the job finishes (or is terminated). |

### 2.2 JobLauncherSpec

Abstract base class for launching jobs. Extends `FLComponent`, which gives it access to the event system.

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `launch_job(job_meta, fl_ctx)` | `(dict, FLContext) -> JobHandleSpec` | Start a job and return its handle. |

### 2.3 Supporting Types

**JobProcessArgs** -- String constants for the keys the upper layer places in `FLContextKey.JOB_PROCESS_ARGS`. These are the standardized parameters every job process needs (workspace path, auth token, job ID, parent URL, etc.).

| Constant | Value | Used by |
|----------|-------|---------|
| `EXE_MODULE` | `"exe_module"` | Server, Client |
| `WORKSPACE` | `"workspace"` | Server, Client |
| `STARTUP_DIR` | `"startup_dir"` | Client |
| `APP_ROOT` | `"app_root"` | Server |
| `AUTH_TOKEN` | `"auth_token"` | Client |
| `TOKEN_SIGNATURE` | `"auth_signature"` | Server, Client |
| `SSID` | `"ssid"` | Server, Client |
| `JOB_ID` | `"job_id"` | Server, Client |
| `CLIENT_NAME` | `"client_name"` | Client |
| `ROOT_URL` | `"root_url"` | Server |
| `PARENT_URL` | `"parent_url"` | Server, Client |
| `PARENT_CONN_SEC` | `"parent_conn_sec"` | Client |
| `SERVICE_HOST` | `"service_host"` | Server |
| `SERVICE_PORT` | `"service_port"` | Server |
| `HA_MODE` | `"ha_mode"` | Server |
| `TARGET` | `"target"` | Client |
| `SCHEME` | `"scheme"` | Client |
| `STARTUP_CONFIG_FILE` | `"startup_config_file"` | Server, Client |
| `RESTORE_SNAPSHOT` | `"restore_snapshot"` | Server |
| `OPTIONS` | `"options"` | Server, Client |

**JobReturnCode** -- Standard exit semantics (extends `ProcessExitCode`):

| Code | Value | Meaning |
|------|-------|---------|
| `SUCCESS` | 0 | Job completed successfully. |
| `EXECUTION_ERROR` | 1 | Job failed during execution. |
| `ABORTED` | 9 | Job was terminated/aborted. |
| `UNKNOWN` | 127 | Status cannot be determined (still running, or lost). |

**`add_launcher(launcher, fl_ctx)`** -- Appends a launcher to the `FLContextKey.JOB_LAUNCHER` list on `fl_ctx`. Called by launchers during the `BEFORE_JOB_LAUNCH` event to volunteer for the current job.

---

## 3. How the Upper Layer Uses Launchers

### 3.1 Event-Based Launcher Selection

The engine never directly instantiates a launcher. Instead, it calls `get_job_launcher()` from `nvflare/private/fed/utils/fed_utils.py`:

```python
def get_job_launcher(job_meta, fl_ctx) -> JobLauncherSpec:
    engine = fl_ctx.get_engine()
    with engine.new_context() as job_launcher_ctx:
        job_launcher_ctx.remove_prop(FLContextKey.JOB_LAUNCHER)
        job_launcher_ctx.set_prop(FLContextKey.JOB_META, job_meta, ...)
        engine.fire_event(EventType.BEFORE_JOB_LAUNCH, job_launcher_ctx)
        job_launcher = job_launcher_ctx.get_prop(FLContextKey.JOB_LAUNCHER)
        ...
    return job_launcher[0]
```

Every registered `FLComponent` receives the `BEFORE_JOB_LAUNCH` event. Each launcher inspects `job_meta` and, if it can handle the job, calls `add_launcher(self, fl_ctx)`. The first launcher to register wins.

**Selection rule in practice:**

| Condition | Launcher selected |
|-----------|-------------------|
| `extract_job_image(job_meta, site_name)` returns `None` | **ProcessJobLauncher** (no container image → run as subprocess) |
| `extract_job_image(job_meta, site_name)` returns an image | **DockerJobLauncher** or **K8sJobLauncher** (whichever is configured as a component) |

### 3.2 Server Side (`ServerEngine`)

Location: `nvflare/private/fed/server/server_engine.py`

```
_start_runner_process(job, job_clients, snapshot, fl_ctx)
│
├─ 1. Build job_args dict with server-specific JobProcessArgs
│     (WORKSPACE, APP_ROOT, PARENT_URL, AUTH_TOKEN, HA_MODE, ...)
│
├─ 2. fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args)
│
├─ 3. job_launcher = get_job_launcher(job.meta, fl_ctx)
│
├─ 4. job_handle = job_launcher.launch_job(job.meta, fl_ctx)
│
├─ 5. Store in run_processes[job_id][RunProcessKey.JOB_HANDLE]
│
└─ 6. Start background thread → wait_for_complete(workspace, job_id, job_handle)
       │
       ├─ job_handle.wait()          # blocks until job finishes
       └─ get_return_code(job_handle, job_id, workspace, logger)
```

**Abort path** (`abort_app_on_server`):

1. Attempt to send an abort command to the child via the cell messaging system.
2. On failure, retrieve `job_handle` from `run_processes` and call `job_handle.terminate()`.

### 3.3 Client Side (`ClientExecutor`)

Location: `nvflare/private/fed/client/client_executor.py`

```
start_app(job_id, job_meta, ...)
│
├─ 1. Build job_args dict with client-specific JobProcessArgs
│     (WORKSPACE, STARTUP_DIR, CLIENT_NAME, PARENT_URL, AUTH_TOKEN, ...)
│
├─ 2. fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args)
│
├─ 3. job_launcher = get_job_launcher(job_meta, fl_ctx)
│
├─ 4. job_handle = job_launcher.launch_job(job_meta, fl_ctx)
│
├─ 5. Store in run_processes[job_id][RunProcessKey.JOB_HANDLE]
│
└─ 6. Start background thread → _wait_child_process_finish(...)
       │
       ├─ job_handle.wait()
       └─ get_return_code(job_handle, job_id, workspace, logger)
```

**Abort path** (`_terminate_job`):

1. Wait up to 10 seconds for the child to exit gracefully (polling `job_handle.poll()`).
2. Call `job_handle.terminate()`.

### 3.4 Return Code Resolution

`get_return_code()` in `fed_utils.py` uses a two-tier strategy:

1. **File-based** -- Check for `FLMetaKey.PROCESS_RC_FILE` in the job's run directory. The child process writes its own return code to this file before exiting. This is the preferred source because it carries the child's own assessment.
2. **Handle-based** -- Fall back to `job_handle.poll()`, which maps the underlying execution unit's status to a `JobReturnCode`.

---

## 4. The Three Implementations

### 4.1 Process Launcher (Subprocess)

**Files:**

| File | Class |
|------|-------|
| `nvflare/app_common/job_launcher/process_launcher.py` | `ProcessHandle`, `ProcessJobLauncher` |
| `nvflare/app_common/job_launcher/server_process_launcher.py` | `ServerProcessJobLauncher` |
| `nvflare/app_common/job_launcher/client_process_launcher.py` | `ClientProcessJobLauncher` |

**Class hierarchy:**

```
JobHandleSpec
  └── ProcessHandle

JobLauncherSpec (FLComponent)
  └── ProcessJobLauncher
        ├── ServerProcessJobLauncher
        └── ClientProcessJobLauncher
```

#### ProcessHandle

Wraps a `ProcessAdapter` (from `nvflare/utils/process_utils.py`) that manages a `subprocess.Popen` or a PID.

| Method | Implementation |
|--------|---------------|
| `terminate()` | Delegates to `adapter.terminate()` (sends SIGTERM/SIGKILL). |
| `poll()` | Calls `adapter.poll()`. Maps exit code 0 → `SUCCESS`, 1 → `EXECUTION_ERROR`, 9 → `ABORTED`, other → `EXECUTION_ERROR`, `None` → `UNKNOWN`. |
| `wait()` | Delegates to `adapter.wait()` (blocks on `subprocess.Popen.wait()`). |

#### ProcessJobLauncher

| Step | Action |
|------|--------|
| 1 | Copy `os.environ` and add `app_custom_folder` to `PYTHONPATH`. |
| 2 | Call `self.get_command(job_meta, fl_ctx)` (abstract -- implemented by server/client subclasses). |
| 3 | Parse command with `shlex.split()`, spawn the process via `spawn_process(argv, new_env)`. |
| 4 | Return `ProcessHandle(process_adapter=...)`. |

**Event registration:**

```python
def handle_event(self, event_type, fl_ctx):
    if event_type == EventType.BEFORE_JOB_LAUNCH:
        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        job_image = extract_job_image(job_meta, fl_ctx.get_identity_name())
        if not job_image:        # no container image → use subprocess
            add_launcher(self, fl_ctx)
```

**Server/Client subclasses** only override `get_command()`:

- `ServerProcessJobLauncher.get_command()` → `generate_server_command(fl_ctx)` → `python -m <module> -w <workspace> ...`
- `ClientProcessJobLauncher.get_command()` → `generate_client_command(fl_ctx)` → `python -m <module> -w <workspace> -n <client_name> ...`

---

### 4.2 Docker Launcher

**File:** `nvflare/app_opt/job_launcher/docker_launcher.py`

**Class hierarchy:**

```
JobHandleSpec
  └── DockerJobHandle

JobLauncherSpec (FLComponent)
  └── DockerJobLauncher
        ├── ClientDockerJobLauncher
        └── ServerDockerJobLauncher
```

#### DockerJobHandle

Wraps a Docker SDK `Container` object.

| Method | Implementation |
|--------|---------------|
| `terminate()` | `container.stop()`. |
| `poll()` | Re-fetches container via `docker.from_env().containers.get(id)`. Maps status: `EXITED` → `SUCCESS`, `DEAD` → `ABORTED`, all others → `UNKNOWN`. Removes the container on terminal states. |
| `wait()` | `enter_states([EXITED, DEAD], timeout)` -- polls container status in a 1-second loop until a terminal state is reached. |

Docker container states and their mappings:

| Docker Status | JobReturnCode |
|---------------|---------------|
| `created` | `UNKNOWN` |
| `restarting` | `UNKNOWN` |
| `running` | `UNKNOWN` |
| `paused` | `UNKNOWN` |
| `exited` | `SUCCESS` |
| `dead` | `ABORTED` |

#### DockerJobLauncher

Constructor parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `mount_path` | `"/workspace"` | Container-side mount point for the host workspace. |
| `network` | `"nvflare-network"` | Docker network the container joins. |
| `timeout` | `None` | Maximum seconds to wait for the container to reach `RUNNING`. |

Launch sequence:

| Step | Action |
|------|--------|
| 1 | Extract `job_image` from `job_meta` via `extract_job_image()`. |
| 2 | Build `PYTHONPATH` with `app_custom_folder`. |
| 3 | Call `self.get_command(job_meta, fl_ctx)` → `(container_name, command_string)`. |
| 4 | Read `NVFL_DOCKER_WORKSPACE` env var for the host-side workspace path. |
| 5 | `docker_client.containers.run(image, command, name, network, volumes, detach=True)`. |
| 6 | `DockerJobHandle(container).enter_states([RUNNING], timeout)`. |
| 7 | If timeout or error → `handle.terminate()`, return `None`. Otherwise return handle. |

**Event registration:** Same pattern as Process but with the opposite condition -- registers when `extract_job_image()` returns a truthy value.

**Server/Client subclasses** override `get_command()`:

- `ClientDockerJobLauncher` → returns `("{client_name}-{job_id}", generate_client_command(fl_ctx))`.
- `ServerDockerJobLauncher` → returns `("server-{job_id}", generate_server_command(fl_ctx))`.

---

### 4.3 Kubernetes Launcher

**File:** `nvflare/app_opt/job_launcher/k8s_launcher.py`

**Class hierarchy:**

```
JobHandleSpec
  └── K8sJobHandle

JobLauncherSpec (FLComponent)
  └── K8sJobLauncher
        ├── ClientK8sJobLauncher
        └── ServerK8sJobLauncher
```

#### K8sJobHandle

Wraps a Kubernetes Pod managed through the `CoreV1Api`.

| Method | Implementation |
|--------|---------------|
| `terminate()` | Calls `delete_namespaced_pod(grace_period_seconds=0)` in a try/except. `terminal_state = TERMINATED` is set when the delete succeeds, or when the `ApiException` has status 404 (pod already gone). For any other `ApiException`, the error is logged and `terminal_state` is left unchanged. |
| `poll()` | If `terminal_state` is set, maps it through `JOB_RETURN_CODE_MAPPING` and returns a `JobReturnCode`. Otherwise calls `_query_state()` and maps the result the same way. Both paths consistently return `JobReturnCode`. |
| `wait()` | Direct while loop: returns immediately if `terminal_state` is set; otherwise calls `_query_state()` and when `SUCCEEDED` or `TERMINATED` is reached, persists that state into `terminal_state` (so subsequent `poll()` calls remain accurate) and returns. Sleeps 1 second per iteration. No timeout. |
| `_query_phase()` | Calls `read_namespaced_pod` and returns the raw pod phase string (e.g. `"Pending"`, `"Running"`). On `ApiException`, returns `POD_Phase.UNKNOWN.value`. |
| `_query_state()` | Calls `_query_phase()` and maps the raw phase through `POD_STATE_MAPPING` to a `JobState`. Used by `poll()` and `wait()`. |
| `enter_states()` | Per iteration: calls `_query_phase()` once, passes the raw phase to both `_stuck()` and directly to `POD_STATE_MAPPING.get()` — single K8s API call per poll cycle. Returns `True` when target state is reached, `False` on timeout or stuck detection. |

Pod phase mapping:

| Pod Phase | JobState | JobReturnCode |
|-----------|----------|---------------|
| `Pending` | `STARTING` | `UNKNOWN` |
| `Running` | `RUNNING` | `UNKNOWN` |
| `Succeeded` | `SUCCEEDED` | `SUCCESS` |
| `Failed` | `TERMINATED` | `ABORTED` |
| `Unknown` | `UNKNOWN` | `UNKNOWN` |

> Note: `POD_Phase.TERMINATED` has been removed from the enum. `POD_STATE_MAPPING` now covers only the five real Kubernetes pod phases: `Pending`, `Running`, `Succeeded`, `Failed`, `Unknown`.

**Stuck detection:** `_stuck_count` starts at `0`. A separate `_stuck_grace_period = 10` is added to `timeout` to form `_max_stuck_count = timeout + _stuck_grace_period`, giving a grace window of ~10 extra poll cycles before stuck detection activates. If `timeout` is `None`, `_max_stuck_count` is also `None` and stuck detection is disabled entirely. `enter_states()` passes the raw phase string from `_query_phase()` directly to `_stuck()`. `_stuck()` compares `current_phase == POD_Phase.PENDING.value` (i.e. `"Pending" == "Pending"`), incrementing `_stuck_count` on each match. When `_stuck_count > _max_stuck_count`, `_stuck()` returns `True`, `enter_states()` calls `terminate()` (which sets `terminal_state = TERMINATED` when the delete call succeeds or returns 404) and returns `False`. Note: `_stuck_count` and `_max_stuck_count` are poll-iteration counts (each ~1 second), not wall-clock seconds — the semantics coincide only because each poll sleeps exactly 1 second.

#### K8sJobHandle Pod Manifest

The handle constructs the pod manifest internally from a `job_config` dict:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: <job_id>
spec:
  restartPolicy: Never
  containers:
    - name: container-<job_id>
      image: <job_image>
      command: ["/usr/local/bin/python"]
      args: ["-u", "-m", "<exe_module>", "-w", "<workspace>", ...]
      volumeMounts:
        - name: nvflws
          mountPath: /var/tmp/nvflare/workspace
        - name: nvfldata
          mountPath: /var/tmp/nvflare/data
        - name: nvfletc
          mountPath: /var/tmp/nvflare/etc
      resources:
        limits:
          nvidia.com/gpu: <num_of_gpus>   # omitted if None
      imagePullPolicy: Always
  volumes:
    - name: nvflws
      persistentVolumeClaim:
        claimName: <workspace_pvc>
    - name: nvfldata
      persistentVolumeClaim:
        claimName: <data_pvc>
    - name: nvfletc
      persistentVolumeClaim:
        claimName: <etc_pvc>
```

#### K8sJobLauncher

Constructor parameters:

| Parameter | Purpose |
|-----------|---------|
| `config_file_path` | Path to kubeconfig file. Loaded via `config.load_kube_config()`. |
| `workspace_pvc` | PVC claim name for the NVFlare workspace. |
| `etc_pvc` | PVC claim name for configuration/etc data. |
| `data_pvc_file_path` | Path to a YAML file mapping PVC names to mount paths for training data. |
| `timeout` | Maximum seconds to wait for pod to reach `Running` (also used as stuck threshold). |
| `namespace` | Kubernetes namespace (default: `"default"`). |

Launch sequence:

| Step | Action |
|------|--------|
| 1 | Extract `job_image`, `site_name`, and optional `num_of_gpus` from `job_meta`. |
| 2 | Read `JOB_PROCESS_ARGS` from `fl_ctx`; extract `EXE_MODULE` as the container command. |
| 3 | Build `job_config` dict: name, image, container name, command, volume mounts/PVCs, `module_args` from `get_module_args()`, set list, GPU resources. |
| 4 | Create `K8sJobHandle(job_id, core_v1, job_config, namespace, timeout)` which builds the pod manifest. |
| 5 | `core_v1.create_namespaced_pod(body=pod_manifest, namespace)`. |
| 6 | Call `job_handle.enter_states([RUNNING], timeout)`. The return value is not checked. If stuck detection fires, `terminate()` is called inside `enter_states` (sets `terminal_state = TERMINATED` via `finally`) before returning the handle, so the caller can detect failure via `poll()`. On plain timeout (no stuck), the handle is returned with `terminal_state` unset and the pod may still be starting. |
| 7 | On `ApiException` from `create_namespaced_pod` → `job_handle.terminate()` then return the handle. Unlike Docker (which returns `None` on failure), the K8s launcher always returns a handle; callers detect failure when `poll()` or `wait()` resolves. |

**Server/Client subclasses** override `get_module_args()`:

- `ClientK8sJobLauncher` → Filters `JOB_PROCESS_ARGS` through `get_client_job_args(include_exe_module=False, include_set_options=False)` to produce the dict of `-flag value` pairs for the container args list.
- `ServerK8sJobLauncher` → Same pattern with `get_server_job_args(...)`.

**Key difference from Process/Docker:** The K8s launcher does not build a shell command string. Instead, it passes the Python executable as `command` and constructs a structured `args` list (`["-u", "-m", "<module>", "-w", "<workspace>", ...]`) directly in the pod spec.

---

## 5. Object-Oriented Design Summary

### 5.1 Full Class Hierarchy

```
JobHandleSpec (abstract)
├── ProcessHandle          (wraps ProcessAdapter / subprocess.Popen)
├── DockerJobHandle        (wraps docker.Container)
└── K8sJobHandle           (wraps CoreV1Api + pod name)

JobLauncherSpec (abstract, extends FLComponent)
├── ProcessJobLauncher     (abstract: get_command)
│   ├── ServerProcessJobLauncher
│   └── ClientProcessJobLauncher
├── DockerJobLauncher      (abstract: get_command)
│   ├── ServerDockerJobLauncher
│   └── ClientDockerJobLauncher
└── K8sJobLauncher         (abstract: get_module_args)
    ├── ServerK8sJobLauncher
    └── ClientK8sJobLauncher
```

### 5.2 Design Patterns

**Strategy Pattern** -- Each launcher is a strategy for running jobs. The engine programs against `JobLauncherSpec`; the concrete strategy is selected at runtime through the event system.

**Template Method Pattern** -- Each base launcher (`ProcessJobLauncher`, `DockerJobLauncher`, `K8sJobLauncher`) implements `launch_job()` with a fixed algorithm, delegating the variable part to an abstract method:

| Base Launcher | Template method calls | Abstract hook |
|---------------|----------------------|---------------|
| `ProcessJobLauncher` | `launch_job()` → `get_command()` | `get_command(job_meta, fl_ctx) -> str` |
| `DockerJobLauncher` | `launch_job()` → `get_command()` | `get_command(job_meta, fl_ctx) -> (str, str)` |
| `K8sJobLauncher` | `launch_job()` → `get_module_args()` | `get_module_args(job_id, fl_ctx) -> dict` |

Server and client subclasses provide the implementation of these hooks, producing the correct command-line arguments for each role.

**Observer Pattern** -- Launchers register for the `BEFORE_JOB_LAUNCH` event through the `FLComponent` event system. This decouples launcher registration from the engine's control flow entirely.

---

## 6. Comparison: Process vs Docker vs Kubernetes

| Aspect | Process | Docker | Kubernetes |
|--------|---------|--------|------------|
| **When selected** | No `job_image` for site | `job_image` present | `job_image` present |
| **Execution unit** | OS subprocess | Docker container | Kubernetes Pod |
| **Isolation** | Shared host, inherited env | Container isolation, mounted workspace | Pod isolation, PVC-backed volumes |
| **Command format** | Shell command string (`python -m ...`) | Shell command inside `/bin/bash -c` | Structured `command` + `args` list in pod spec |
| **Workspace access** | Direct filesystem (same host) | Host directory bind-mounted to container | PersistentVolumeClaims |
| **Data access** | Direct filesystem | Via bind mount | Via PVC (configured in YAML) |
| **Start verification** | None (spawn returns immediately) | Poll for `RUNNING` state with timeout | `enter_states([RUNNING], timeout)` with stuck detection; return value not checked — on stuck, `terminal_state` is set so caller can detect via `poll()` |
| **Wait mechanism** | `subprocess.Popen.wait()` (OS-level block) | Poll container status for `EXITED`/`DEAD` | Direct while loop via `_query_state()`; no timeout; exits when `terminal_state` set or `SUCCEEDED`/`TERMINATED` reached |
| **Terminate** | `SIGTERM`/`SIGKILL` via `ProcessAdapter` | `container.stop()` | `delete_namespaced_pod(grace_period=0)`; `terminal_state` set to `TERMINATED` on success or 404; left unchanged (error logged) for other exceptions |
| **Return code source** | Process exit code or RC file | Container status mapping or RC file | Pod phase mapping or RC file; `poll()` now consistently returns `JobReturnCode` via `JOB_RETURN_CODE_MAPPING` |
| **GPU support** | Inherited from host environment | Not explicitly managed | `nvidia.com/gpu` resource limit in pod spec |
| **Dependencies** | stdlib only | `docker` Python SDK | `kubernetes` Python client + kubeconfig |
| **Typical use case** | Simulator, single-machine POC | Multi-container on single host | Production cluster with shared storage |

---

## 7. Backend (BE) Resource Management

When using the K8s launcher, the NVFlare process managing jobs may not itself run on a GPU node. The K8s scheduler handles actual resource allocation externally. Two passthrough classes support this case:

### 7.1 BEResourceManager (`nvflare/app_common/resource_managers/BE_resource_manager.py`)

`BEResourceManager(ResourceManagerSpec, FLComponent)` is a "Best Effort" resource manager: it always approves resource allocation requests and performs no local tracking, allowing jobs to attempt to run and fail at runtime if resources are genuinely unavailable:

| Method | Behavior |
|--------|----------|
| `check_resources()` | Always returns `(True, <uuid_token>)` -- never rejects a job. |
| `cancel_resources()` | No-op. |
| `allocate_resources()` | Returns empty dict `{}`. |
| `free_resources()` | No-op. |
| `report_resources()` | Returns `{}` (empty dict, conforming to the `ResourceManagerSpec` contract). |

Use this when the container orchestration backend (K8s) is responsible for all resource accounting.

### 7.2 BEResourceConsumer (`nvflare/app_common/resource_consumers/BE_resource_consumer.py`)

`BEResourceConsumer(ResourceConsumerSpec)` implements a no-op `consume()`. Use this alongside `BEResourceManager` when no local resource consumption reporting is needed.

### 7.3 GPUResourceManager `ignore_host` flag (`nvflare/app_common/resource_managers/gpu_resource_manager.py`)

`GPUResourceManager` gained an `ignore_host=False` parameter. When `True`, the constructor skips the startup validation that checks whether the declared `num_of_gpus` and `mem_per_gpu_in_GiB` match actual host hardware. This is needed in K8s deployments where the NVFlare process runs on a node without GPUs but still needs to track a GPU resource pool for job scheduling purposes.

---

## 8. Sequence Diagram

The following shows the end-to-end flow for launching and managing a job, applicable to both server and client:

```
  Engine                  fed_utils              Launcher(s)             Handle
    │                        │                      │                      │
    │  get_job_launcher()    │                      │                      │
    │───────────────────────>│                      │                      │
    │                        │  fire BEFORE_JOB_LAUNCH                     │
    │                        │─────────────────────>│                      │
    │                        │                      │ check job_meta       │
    │                        │                      │ add_launcher(self)   │
    │                        │<─────────────────────│                      │
    │    return launcher     │                      │                      │
    │<───────────────────────│                      │                      │
    │                        │                      │                      │
    │  launcher.launch_job(job_meta, fl_ctx)        │                      │
    │─────────────────────────────────────────────->│                      │
    │                        │                      │  create exec unit    │
    │                        │                      │  (process/container  │
    │                        │                      │   /pod)              │
    │                        │                      │─────────────────────>│
    │                        │                      │  return handle       │
    │<─────────────────────────────────────────────────────────────────────│
    │                        │                      │                      │
    │  store handle in run_processes                │                      │
    │                        │                      │                      │
    │  [background thread]   │                      │                      │
    │  handle.wait()         │                      │                      │
    │─────────────────────────────────────────────────────────────────────>│
    │                        │                      │        blocks/polls  │
    │                        │                      │                      │
    │  ... (on abort) ...    │                      │                      │
    │  handle.terminate()    │                      │                      │
    │─────────────────────────────────────────────────────────────────────>│
    │                        │                      │                      │
    │  ... (on completion) . │                      │                      │
    │  get_return_code()     │                      │                      │
    │───────────────────────>│                      │                      │
    │                        │  check RC file       │                      │
    │                        │  or handle.poll()    │                      │
    │                        │─────────────────────────────────────────────>│
    │   return_code          │                      │                      │
    │<───────────────────────│                      │                      │
```

---

## 9. Configuration

Launchers are registered as FL components in the site's `resources.json`. The configurator loads them at startup so they receive events.

### 9.1 Process Launcher (default)

```json
{
  "id": "job_launcher",
  "path": "nvflare.app_common.job_launcher.server_process_launcher.ServerProcessJobLauncher",
  "args": {}
}
```

### 9.2 Docker Launcher

```json
{
  "id": "job_launcher",
  "path": "nvflare.app_opt.job_launcher.docker_launcher.ClientDockerJobLauncher",
  "args": {
    "mount_path": "/workspace",
    "network": "nvflare-network",
    "timeout": 60
  }
}
```

Requires the `NVFL_DOCKER_WORKSPACE` environment variable to be set on the host to identify the workspace directory to bind-mount.

### 9.3 Kubernetes Launcher

```json
{
  "id": "job_launcher",
  "path": "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
  "args": {
    "config_file_path": "/path/to/kubeconfig",
    "workspace_pvc": "nvflare-workspace-pvc",
    "etc_pvc": "nvflare-etc-pvc",
    "data_pvc_file_path": "/path/to/data_pvc.yaml",
    "timeout": 120,
    "namespace": "nvflare"
  }
}
```

The `data_pvc_file_path` YAML file maps PVC names to mount paths:

```yaml
my-data-pvc: /var/tmp/nvflare/data
```

> Note: Currently only the PVC name (the YAML key) is used. The mount path value is ignored — the data volume is always mounted at the hardcoded path `/var/tmp/nvflare/data`.

---

## 10. Future Improvements

1. **Explicit launcher selection** -- Today "has image" → Docker or K8s, "no image" → Process. Allow an explicit `launcher_type` field in job meta or deploy map so a site can support multiple container backends or provide fallback ordering (e.g., try K8s, fall back to Docker).

2. **Consistent GPU handling** -- The K8s launcher reads `num_of_gpus` from the resource spec; the Docker and Process launchers do not. Standardize resource declaration so job definitions remain portable across backends.

3. **Unified cleanup** -- Standardize container/pod cleanup policy across launchers (auto-remove on exit, configurable retention for debugging) and centralize it in the handle or engine.

4. **Consistent timeout policy and failure semantics** -- The Process launcher has no start timeout. Docker polls for `RUNNING` and returns `None` on failure. K8s polls for `Running` with stuck detection (terminates and sets `terminal_state` on stuck) but does not act on plain startup timeout — if a pod is slow to start but not stuck in `Pending`, the handle is returned with `terminal_state` unset. Consider terminating explicitly on timeout and unifying failure return across all launchers (either always `None` or always a terminated handle).

5. **Observability** -- Add an optional `get_info()` method to `JobHandleSpec` so the engine can log launcher-specific details (container ID, pod name, namespace, PID) for debugging and operations.

6. **Testing** -- Provide `MockJobLauncher` and `MockJobHandle` implementations for unit tests that verify server/client flow without starting real processes or containers.
