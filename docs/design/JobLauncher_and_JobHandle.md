# JobLauncher and JobHandle Design Document

## 1. Overview

NVFlare runs each federated job as an isolated execution unit -- a subprocess or Kubernetes pod. Two abstractions govern this:

- **JobLauncherSpec** -- starts a job and returns a handle.
- **JobHandleSpec** -- represents the running job and provides lifecycle control (poll, wait, terminate).

The upper layers (server engine, client executor) program exclusively against these two interfaces. The concrete backend is selected at runtime through an event-based mechanism, so the engine never imports or names a specific launcher type.

```
┌──────────────────────────────────────────────────────────┐
│                    Upper Layer                           │
│     ServerEngine  /  ClientExecutor                      │
│                                                          │
│  1. get_job_launcher(job_meta, fl_ctx) → launcher        │
│  2. Build JOB_PROCESS_ARGS                               │
│  3. launcher.launch_job(job_meta, fl_ctx) → job_handle   │
│  4. job_handle.wait()  /  job_handle.terminate()         │
└──────────┬──────────────────────┬────────────────────────┘
           │   BEFORE_JOB_LAUNCH  │
           │   event selects one  │
           ▼                      ▼
┌─────────────────┐           ┌─────────────────┐
│ ProcessJob      │           │ K8sJob          │
│ Launcher        │           │ Launcher        │
│ ─────────────── │           │ ─────────────── │
│ ProcessHandle   │           │ K8sJobHandle    │
└─────────────────┘           └─────────────────┘
    subprocess                     pod
```

---

## 2. Specification Layer (`nvflare/apis/job_launcher_spec.py`)

### 2.1 JobHandleSpec

Abstract base class representing a running job (`class JobHandleSpec(ABC)`). All methods are decorated with `@abstractmethod`; Python enforces that any concrete subclass implements all three. Attempting to instantiate a subclass with any abstract method unimplemented raises `TypeError` at instantiation time.

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `terminate()` | `() -> None` | Stop the job immediately. |
| `poll()` | `() -> JobReturnCode` | Non-blocking query for the job's current return code. Returns `UNKNOWN` while still running. |
| `wait()` | `() -> None` | Block until the job finishes (or is terminated). |

### 2.2 JobLauncherSpec

Abstract base class for launching jobs (`class JobLauncherSpec(FLComponent, ABC)`). Extends `FLComponent` for event-system access. Adding `ABC` to the bases means Python selects `ABCMeta` as the metaclass automatically (since `ABCMeta` is a subclass of `type`), so `@abstractmethod` on `launch_job` is enforced at runtime for all subclasses.

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `launch_job(job_meta, fl_ctx)` | `(dict, FLContext) -> JobHandleSpec` | Start a job and return its handle. |

### 2.3 Supporting Types

**JobProcessArgs** -- String constants for the keys the upper layer places in `FLContextKey.JOB_PROCESS_ARGS`. Each value in the dict is a `(flag, value)` tuple, e.g. `JobProcessArgs.JOB_ID → ("-n", job_id)`. These are the standardized parameters every job process needs (workspace path, auth token, job ID, parent URL, etc.).

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
| `HA_MODE` | `"ha_mode"` | Server |
| `TARGET` | `"target"` | Client |
| `SCHEME` | `"scheme"` | Client |
| `STARTUP_CONFIG_FILE` | `"startup_config_file"` | Server, Client |
| `RESTORE_SNAPSHOT` | `"restore_snapshot"` | (defined but not set as a standalone entry; server embeds `restore_snapshot=<bool>` into `OPTIONS` via `--set`) |
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

Every registered `FLComponent` receives the `BEFORE_JOB_LAUNCH` event. Each launcher inspects `job_meta` and, if it can handle the job, calls `add_launcher(self, fl_ctx)`. The first launcher to register wins.

**Selection rule in practice:**

| Condition | Launcher selected |
|-----------|-------------------|
| `extract_job_image(job_meta, site_name)` returns `None` | **ProcessJobLauncher** (no container image → run as subprocess) |
| `extract_job_image(job_meta, site_name)` returns an image | **K8sJobLauncher** (configured as a component) |

### 3.2 Server Side (`ServerEngine`)

Location: `nvflare/private/fed/server/server_engine.py`

```
_start_runner_process(job, job_clients, snapshot, fl_ctx)
│
├─ 1. job_launcher = get_job_launcher(job.meta, fl_ctx)
│     (fires BEFORE_JOB_LAUNCH; JOB_PROCESS_ARGS not yet set)
│
├─ 2. Build job_args dict with server-specific JobProcessArgs
│     (EXE_MODULE, JOB_ID, WORKSPACE, STARTUP_CONFIG_FILE,
│      APP_ROOT, HA_MODE, AUTH_TOKEN, TOKEN_SIGNATURE,
│      PARENT_URL, ROOT_URL, SERVICE_HOST, SERVICE_PORT,
│      SSID, OPTIONS)
│
├─ 3. fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args)
│
├─ 4. job_handle = job_launcher.launch_job(job.meta, fl_ctx)
│
├─ 5. Store in run_processes[job_id]
│        {JOB_HANDLE: job_handle, JOB_ID: job_id, PARTICIPANTS: job_clients}
│
└─ 6. Start background thread → wait_for_complete(workspace, job_id, job_handle)
       │
       ├─ job_handle.wait()          # blocks until job finishes
       ├─ wait up to 2s for UPDATE_RUN_STATUS message to arrive
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
├─ 1. job_launcher = get_job_launcher(job_meta, fl_ctx)
│     (fires BEFORE_JOB_LAUNCH; JOB_PROCESS_ARGS not yet set)
│
├─ 2. Build job_args dict with client-specific JobProcessArgs
│     (EXE_MODULE, JOB_ID, CLIENT_NAME, AUTH_TOKEN, TOKEN_SIGNATURE,
│      SSID, WORKSPACE, STARTUP_DIR, PARENT_URL, SCHEME, TARGET,
│      STARTUP_CONFIG_FILE, OPTIONS, optionally PARENT_CONN_SEC)
│
├─ 3. fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args)
│
├─ 4. job_handle = job_launcher.launch_job(job_meta, fl_ctx)
│
├─ 5. Fire EventType.AFTER_JOB_LAUNCH event
│
├─ 6. Store in run_processes[job_id]
│        {JOB_HANDLE: job_handle, STATUS: ClientStatus.STARTING}
│
└─ 7. Start background thread → _wait_child_process_finish(...)
       │
       ├─ job_handle.wait()
       └─ get_return_code(job_handle, job_id, workspace, logger)
```

**Abort path** (`_terminate_job`):

1. Poll `self.run_processes.get(job_id)` every 50 ms for up to 10 seconds; if the entry disappears the job finished gracefully.
2. Always call `job_handle.terminate()` regardless of whether the graceful exit was detected.

### 3.4 Return Code Resolution

`get_return_code()` in `fed_utils.py` uses a two-tier strategy:

1. **File-based** -- Check for `FLMetaKey.PROCESS_RC_FILE` in the job's run directory. The child process writes its own return code to this file before exiting. This is the preferred source because it carries the child's own assessment.
2. **Handle-based** -- Fall back to `job_handle.poll()`, which maps the underlying execution unit's status to a `JobReturnCode`.

---

## 4. The Two Implementations

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

Wraps a `ProcessAdapter` (from `nvflare/utils/process_utils.py`) that manages a `subprocess.Popen` or a PID. The constructor accepts any one of: a `ProcessAdapter` directly, a `subprocess.Popen` object, or an integer `pid`.

| Method | Implementation |
|--------|---------------|
| `terminate()` | Delegates to `adapter.terminate()` (sends SIGTERM/SIGKILL). |
| `poll()` | Calls `adapter.poll()`. Maps exit code 0 → `SUCCESS`, 1 → `EXECUTION_ERROR`, 9 → `ABORTED`, other → `EXECUTION_ERROR`, `None` → `UNKNOWN`. |
| `wait()` | Delegates to `adapter.wait()` (blocks on `subprocess.Popen.wait()`). |

#### ProcessJobLauncher

| Step | Action |
|------|--------|
| 1 | Copy `os.environ`. If `app_custom_folder` is non-empty, call `add_custom_dir_to_path()`: appends the folder to `sys.path` and serializes the result into `PYTHONPATH` in the child environment. |
| 2 | Call `self.get_command(job_meta, fl_ctx)` (abstract -- implemented by server/client subclasses). |
| 3 | Parse command with `shlex.split(command, posix=True)`, spawn the process via `spawn_process(argv, new_env)`. |
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

- `ServerProcessJobLauncher.get_command()` → `generate_server_command(fl_ctx)` → `sys.executable -m <module> -w <workspace> ...`
- `ClientProcessJobLauncher.get_command()` → `generate_client_command(fl_ctx)` → `sys.executable -m <module> -w <workspace> -n <client_name> ...`

---

### 4.2 Kubernetes Launcher

**File:** `nvflare/app_opt/job_launcher/k8s_launcher.py`

**Class hierarchy:**

```
JobHandleSpec (ABC)
  └── K8sJobHandle

JobLauncherSpec (FLComponent, ABC)
  └── K8sJobLauncher
        ├── ClientK8sJobLauncher
        └── ServerK8sJobLauncher
```

#### Pod Name Sanitization

`launch_job` calls `uuid4_to_rfc1123(job_meta.get(JobConstants.JOB_ID))` before constructing the pod. This converts a raw UUID4 job ID into an RFC 1123-compliant Kubernetes pod name:

1. Lowercase the string.
2. Strip characters that are not alphanumeric or hyphens (`[^a-z0-9-]`).
3. Prefix with `"j"` if the first character is a digit (Kubernetes pod names must start with a letter).
4. Strip trailing hyphens.
5. Truncate to 63 characters.

The sanitized name is used as both the pod name (`metadata.name`) and the `job_id` stored in `K8sJobHandle` for all subsequent API calls (`terminate`, `_query_phase`).

#### K8sJobHandle

Wraps a Kubernetes Pod managed through the `CoreV1Api`.

| Method | Implementation |
|--------|---------------|
| `terminate()` | Calls `delete_namespaced_pod(grace_period_seconds=0)`. `terminal_state = TERMINATED` is always set regardless of outcome: on success, on 404 `ApiException` (pod already gone, logged at `info`), on any other `ApiException` (logged at `error`), and on any other `Exception` such as network or serialization errors (also logged at `error`). This guarantees that callers holding a handle never poll indefinitely after calling `terminate()`, even when the K8s API is unreachable. |
| `poll()` | If `terminal_state` is set, maps it through `JOB_RETURN_CODE_MAPPING` and returns a `JobReturnCode`. Otherwise calls `_query_state()` and maps the result the same way. Both paths consistently return `JobReturnCode`. |
| `wait()` | Direct while loop: returns immediately if `terminal_state` is set; otherwise calls `_query_state()` and when `SUCCEEDED` or `TERMINATED` is reached, persists that state into `terminal_state` (so subsequent `poll()` calls remain accurate) and returns. Sleeps 1 second per iteration. No timeout. |
| `_query_phase()` | Calls `read_namespaced_pod` and returns the raw pod phase string (e.g. `"Pending"`, `"Running"`). On `ApiException`, logs the error and returns `POD_Phase.UNKNOWN.value`. On any other `Exception` (e.g. network errors, `urllib3.exceptions.MaxRetryError`), logs the error and also returns `POD_Phase.UNKNOWN.value` — preventing unhandled exceptions from propagating through `enter_states`/`wait`/`poll` and orphaning running pods. |
| `_query_state()` | Calls `_query_phase()` and maps the raw phase through `POD_STATE_MAPPING` to a `JobState`. Used by `poll()` and `wait()`. |
| `enter_states()` | Takes only `job_states_to_enter`; no `timeout` parameter — reads `self.timeout` from the instance. Per iteration: calls `_query_phase()` once, passes the raw phase to `_stuck_in_pending()` and to `POD_STATE_MAPPING.get()` — single K8s API call per poll cycle. Three early-exit paths, evaluated in this order: (1) **Stuck detection** — calls `self.terminate()` (sets `terminal_state = TERMINATED`) then returns `False`; (2) **Terminal phase** — if `pod_phase` is `"Failed"` or `"Succeeded"`, sets `terminal_state` from `POD_STATE_MAPPING` then returns `False` *without* calling `terminate()`. The terminal-phase check is evaluated before the plain-timeout check so that a job completing exactly when the timeout expires is recorded as `SUCCEEDED`/`TERMINATED` (per `POD_STATE_MAPPING`) rather than being misclassified as `TERMINATED` by `terminate()`; (3) **Plain timeout** (`self.timeout` elapsed) — calls `self.terminate()`, sets `terminal_state = TERMINATED`, returns `False`. Setting `terminal_state` in all `False`-return paths prevents `wait()` from looping indefinitely if the pod is GC'd before `wait()` runs. Returns `True` when target state is reached. |

Pod phase mapping:

| Pod Phase | JobState | JobReturnCode |
|-----------|----------|---------------|
| `Pending` | `STARTING` | `UNKNOWN` |
| `Running` | `RUNNING` | `UNKNOWN` |
| `Succeeded` | `SUCCEEDED` | `SUCCESS` |
| `Failed` | `TERMINATED` | `ABORTED` |
| `Unknown` | `UNKNOWN` | `UNKNOWN` |

**Stuck detection:** `_stuck_count` starts at `0`. `_max_stuck_count` is set in the constructor as: `timeout if timeout is not None else pending_timeout`. So stuck detection is **always active** — if `timeout` is provided, `_max_stuck_count = timeout`; if `timeout` is `None`, `_max_stuck_count = pending_timeout` (default 30). The method `_stuck_in_pending(current_phase)` increments `_stuck_count` each time `current_phase == "Pending"` and returns `True` when `_stuck_count >= _max_stuck_count`. When the phase is **not** `Pending`, `_stuck_count` is **reset to 0** — so a pod that transitions out of Pending (e.g. briefly reaches Running then goes back to Pending) starts its stuck count fresh. When it returns `True`, `enter_states()` calls `terminate()` and returns `False`. Plain startup timeout (wall-clock elapsed > `timeout`) also calls `terminate()` before returning `False`. In both cases `terminal_state` is always set to `TERMINATED` by `terminate()`, regardless of whether the API call succeeds or raises. Note: `_stuck_count` and `_max_stuck_count` are poll-iteration counts (~1 second each).

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
      command: ["<python_path>"]   # default: /usr/local/bin/python; configurable via K8sJobLauncher.python_path
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

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `config_file_path` | (required) | Path to kubeconfig file. Loaded via `config.load_kube_config()` at init time. |
| `workspace_pvc` | (required) | PVC claim name for the NVFlare workspace. |
| `etc_pvc` | (required) | PVC claim name for configuration/etc data. |
| `data_pvc_file_path` | (required) | Path to a YAML file mapping PVC names to mount paths for training data. Read and validated at init time; raises `ValueError` if the file is empty/contains no entries, or if the parsed YAML is not a `dict`. Only the first key (PVC name) is used. |
| `timeout` | `None` | Maximum wall-clock seconds for `enter_states([RUNNING])`. Also used as `_max_stuck_count`. If `None`, `pending_timeout` governs stuck detection instead. |
| `namespace` | `"default"` | Kubernetes namespace. |
| `pending_timeout` | `30` | Stuck-detection threshold (poll iterations) when `timeout` is `None`. Passed to `K8sJobHandle`. |
| `python_path` | `"/usr/local/bin/python"` | Absolute path to the Python executable used as the container `command`. Override for non-standard images where Python is not at `/usr/local/bin/python` (e.g. `/usr/bin/python3`). Passed through to `K8sJobHandle`. |

Launch sequence:

| Step | Action |
|------|--------|
| 1 | Validate and sanitize job ID: `raw_job_id = job_meta.get(JobConstants.JOB_ID)`; raises `RuntimeError` if missing or falsy. Then `job_id = uuid4_to_rfc1123(raw_job_id)`. All subsequent operations use this RFC 1123-compliant name. Extract `job_image`, `site_name`, and optional `num_of_gpus` from `job_meta`. |
| 2 | Read `JOB_PROCESS_ARGS` from `fl_ctx`; raises `RuntimeError` if the dict is absent. Raises `RuntimeError` if `EXE_MODULE` is missing or falsy. Extract the module name via `_, job_cmd = exe_module_entry`. |
| 3 | Build `job_config` dict: name (`job_id`), image, container name (`container-{job_id}`), command, volume mounts/PVCs, `module_args` from `get_module_args()`. `set_list` is conditionally set: if `fl_ctx.get_prop(FLContextKey.ARGS)` is non-None and `getattr(args, "set", None)` is non-None, `set_list = args.set` is added (see note below). Using `getattr` rather than direct attribute access guards against non-standard `ARGS` objects that lack a `set` attribute. If `num_of_gpus` is truthy (non-None **and** non-zero), adds `job_config["resources"] = {"limits": {"nvidia.com/gpu": num_of_gpus}}`; a value of `0` is intentionally excluded to avoid injecting `nvidia.com/gpu: 0` into the pod spec, which would explicitly request zero GPUs and can affect scheduling on GPU-enabled clusters differently than omitting the limit entirely. |
| 4 | Create `K8sJobHandle(job_id, core_v1, job_config, namespace=self.namespace, timeout=self.timeout, pending_timeout=self.pending_timeout)` which builds the pod manifest. |
| 5 | `core_v1.create_namespaced_pod(body=pod_manifest, namespace)` in a `try/except Exception` block. On any exception — including `ApiException` (K8s API error) and lower-level errors such as network timeouts or serialization failures — `job_handle.terminate()` is called (always sets `terminal_state = TERMINATED`) and the handle is returned. The scope of this handler is intentionally limited to this single API call; it does not swallow exceptions from the polling loop in step 6. |
| 6 | Call `job_handle.enter_states([RUNNING])` in a separate `try/except BaseException` block. On any exception (including `KeyboardInterrupt`) → `job_handle.terminate()` then re-raise. This ensures a pod already created in step 5 is not orphaned if the blocking poll loop is interrupted. The return value is captured: if `False`, logs a warning `"unable to enter running phase {job_id}"`. Inside `enter_states`: stuck detection and plain timeout both call `terminate()` (always sets `terminal_state = TERMINATED`) then return `False`; if the pod reaches a terminal phase (`Failed`/`Succeeded`), `terminal_state` is set from `POD_STATE_MAPPING` and `enter_states` returns `False` without calling `terminate()`. Setting `terminal_state` in all `False`-return paths prevents `wait()` from looping if the pod is GC'd before `wait()` runs. |
| 7 | Return `job_handle`. The K8s launcher always returns a handle; callers detect launch failure when `poll()` or `wait()` resolves. |

> **`set_list` note:** `args.set` is the CLI `--set` items stored in `FLContextKey.ARGS` at the time `launch_job` is called. The server and client both make a deep copy of `FLContextKey.ARGS`, append `print_conf=True` (and server also appends `restore_snapshot=<bool>`) to that copy, and embed the expanded string into `JOB_PROCESS_ARGS[OPTIONS]`. They do **not** write the modified copy back to `FLContextKey.ARGS`. As a result, the K8s launcher's `set_list` contains only the original CLI `--set` items — **without** `print_conf=True` or `restore_snapshot=...`. The Process launcher receives those extra flags through `OPTIONS`, which K8s excludes from `module_args` (`get_*_job_args(include_set_options=False)`).

**Server/Client subclasses** override `get_module_args()`:

- `ClientK8sJobLauncher` → Calls `_job_args_dict(job_args, get_client_job_args(False, False))` — filters `JOB_PROCESS_ARGS` excluding `EXE_MODULE` and `OPTIONS`, producing a `{flag: value}` dict for the container `args` list.
- `ServerK8sJobLauncher` → Same pattern with `get_server_job_args(False, False)`.

**Key difference from Process:** The K8s launcher does not build a shell command string. Instead, it passes the Python executable as `command` and constructs a structured `args` list (`["-u", "-m", "<module>", "-w", "<workspace>", ...]`) directly in the pod spec.

---

## 5. Object-Oriented Design Summary

### 5.1 Full Class Hierarchy

```
JobHandleSpec (ABC)
├── ProcessHandle          (wraps ProcessAdapter / subprocess.Popen)
└── K8sJobHandle           (wraps CoreV1Api + pod name)

JobLauncherSpec (FLComponent, ABC)
├── ProcessJobLauncher     (abstract: get_command)
│   ├── ServerProcessJobLauncher
│   └── ClientProcessJobLauncher
└── K8sJobLauncher         (abstract: get_module_args; inherits ABCMeta from JobLauncherSpec)
    ├── ServerK8sJobLauncher
    └── ClientK8sJobLauncher
```

### 5.2 Design Patterns

**Strategy Pattern** -- Each launcher is a strategy for running jobs. The engine programs against `JobLauncherSpec`; the concrete strategy is selected at runtime through the event system.

**Template Method Pattern** -- Each base launcher (`ProcessJobLauncher`, `K8sJobLauncher`) implements `launch_job()` with a fixed algorithm, delegating the variable part to an abstract method:

| Base Launcher | Template method calls | Abstract hook |
|---------------|----------------------|---------------|
| `ProcessJobLauncher` | `launch_job()` → `get_command()` | `get_command(job_meta, fl_ctx) -> str` |
| `K8sJobLauncher` | `launch_job()` → `get_module_args()` | `get_module_args(job_id, fl_ctx) -> dict` |

Server and client subclasses provide the implementation of these hooks, producing the correct command-line arguments for each role.

**Observer Pattern** -- Launchers register for the `BEFORE_JOB_LAUNCH` event through the `FLComponent` event system. This decouples launcher registration from the engine's control flow entirely.

---

## 6. Comparison: Process vs Kubernetes

| Aspect | Process | Kubernetes |
|--------|---------|------------|
| **When selected** | No `job_image` for site | `job_image` present |
| **Execution unit** | OS subprocess | Kubernetes Pod |
| **Isolation** | Shared host, inherited env | Pod isolation, PVC-backed volumes |
| **Command format** | Shell command string (`sys.executable -m ...`) | Structured `command: ["/usr/local/bin/python"]` + `args` list in pod spec |
| **Workspace access** | Direct filesystem (same host) | PersistentVolumeClaims |
| **Data access** | Direct filesystem | Via PVC (configured in YAML) |
| **Start verification** | None (spawn returns immediately) | `enter_states([RUNNING])` called (reads `self.timeout`); return value is checked — `False` logs a debug message but handle is still always returned; on stuck detection or plain timeout, `terminate()` is called (sets `terminal_state = TERMINATED`); callers detect failure via `poll()` |
| **Wait mechanism** | `subprocess.Popen.wait()` (OS-level block) | Direct while loop via `_query_state()`; no timeout; exits when `terminal_state` set or `SUCCEEDED`/`TERMINATED` reached |
| **Terminate** | `SIGTERM`/`SIGKILL` via `ProcessAdapter` | `delete_namespaced_pod(grace_period=0)`; `terminal_state` always set to `TERMINATED` — on success, 404, other `ApiException`, or any lower-level `Exception` (network/serialization). Errors are logged but never propagated. |
| **Return code source** | Process exit code or RC file | Pod phase mapping or RC file; `poll()` consistently returns `JobReturnCode` via `JOB_RETURN_CODE_MAPPING` |
| **GPU support** | Inherited from host environment | `nvidia.com/gpu` resource limit set in pod spec from `job_meta` resource spec (`num_of_gpus`); omitted if not specified |
| **Dependencies** | stdlib only | `kubernetes` Python client + kubeconfig |
| **Typical use case** | Simulator, single-machine POC | Production cluster with shared storage |

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
    │                        │                      │  (process/pod)       │
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

### 9.2 Kubernetes Launcher

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

1. **Explicit launcher selection** -- Today "has image" → K8s, "no image" → Process. Allow an explicit `launcher_type` field in job meta or deploy map so a site can support multiple backends or provide fallback ordering.

2. **Consistent GPU handling** -- The K8s launcher applies `num_of_gpus` from the job resource spec as a pod `nvidia.com/gpu` limit; the Process launcher ignores it entirely. Standardize resource declaration so job definitions remain portable across backends.

3. **Unified cleanup** -- Standardize pod cleanup policy (auto-remove on exit, configurable retention for debugging) and centralize it in the handle or engine.

4. **Consistent timeout policy and failure semantics** -- The Process launcher has no start timeout. K8s checks the `enter_states` return value and logs a debug message on failure (termination already happens inside `enter_states` for stuck/timeout paths), always returning a handle. Consider exposing a distinct "startup failed" state on the handle so callers can react without polling.

5. **Observability** -- Add an optional `get_info()` method to `JobHandleSpec` so the engine can log launcher-specific details (pod name, namespace, PID) for debugging and operations.

6. **Testing** -- Provide `MockJobLauncher` and `MockJobHandle` implementations for unit tests that verify server/client flow without starting real processes or pods.
