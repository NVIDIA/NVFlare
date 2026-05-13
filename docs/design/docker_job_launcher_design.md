# Docker Job Launcher Design

## Overview

The Docker Job Launcher provides container-based job execution for NVFlare deployments where each site has Docker available.

The federation topology is the same as always: one server site and N client sites, each on their own machine. The difference from process mode is that SP/CP and all their job processes run as Docker containers instead of bare subprocesses. Each site manages its own containers independently — there is no shared orchestrator across sites.

The primary value is **dependency isolation**: each job can specify its own Docker image with a different ML framework version, CUDA version, or set of libraries, without affecting the site's host environment or other jobs.

Current architecture: Docker launcher support is enabled by preparing an existing server or client startup kit with `nvflare deploy prepare` and a `runtime: docker` config. Runtime choice is a site-local post-provision step, not a centralized `project.yml` decision.

---

## Target Persona

- Developer or researcher who wants job isolation (different image per job) without cluster overhead
- Site operator running Docker on bare metal or VM where a K8s cluster is not available or not needed
- Small-scale deployment where Docker is available on each site machine

---

## Assumptions

### Topology
- **Server and clients are typically on separate machines**, but they can also run on the same machine (e.g. local testing). Docker mode does not change the federation topology.
- **Each site runs on one Docker host** — SP runs on the server machine, each CP runs on its own client machine. No Docker Swarm, no multi-host networking within a site.
- **SP/CP runs as a container** — this mirrors the K8s model and avoids host/container networking issues for `PARENT_URL` (see Networking section).

### Startup
- Site admin runs `nvflare deploy prepare <kit> --config docker.yaml` to generate a prepared Docker kit, then runs `startup/start_docker.sh` from that prepared kit.
- `start_docker.sh` is the supported startup path for a Docker-prepared site. Standard process-mode startup scripts are removed from the prepared Docker output so the active launcher choice is explicit.
- Job containers (SJ/CJ) are **not** started by the site admin. They are started dynamically by the `DockerJobLauncher` when SP/CP receives a runnable job.
- **The site admin is responsible for building the Docker image** before running `start_docker.sh`. Deploy preparation writes the image name into the generated script; it does not build or push images.

### Site Consistency Rule
Within a single site, SP/CP and all SJ/CJ containers must use the same launcher mode. If SP/CP starts as a Docker container, all jobs on that site run as Docker containers. Mixed mode within a site is not supported.

| Mode | SP/CP startup | SJ/CJ startup |
|---|---|---|
| Process | `start.sh` → `python ...` | `ProcessJobLauncher` → `subprocess.Popen` |
| Docker | `start_docker.sh` → `docker run` | `DockerJobLauncher` → `docker run` |
| K8s | `kubectl apply` | `K8sJobLauncher` → K8s Pod |

### Networking
- A Docker network (`nvflare-network` by default) is created automatically by `start_docker.sh` if it does not exist.
- SP/CP and all SJ/CJ containers on the same site join this network. It is used **only for intra-site parent↔child communication** (`PARENT_URL`). Docker's built-in DNS resolves container names — no `host.docker.internal` hacks needed.
- **Cross-site communication (CP → SP) does not use the Docker network.** It goes over the host network via the published `fed_learn_port`, using the same HTTPS/gRPC as process mode. From CP's perspective, it connects to the server's hostname/IP just as it would in process mode.
- The server parent container publishes `fed_learn_port` and, when it is distinct, `admin_port`. Both values come from the input startup kit.

```mermaid
graph TD
    subgraph server_machine["Server Machine"]
        SP["SP container\n(name: server)\nnvflare-server:latest"]
        SJ["SJ container\nnvflare-job:latest"]
        DS1["/var/run/docker.sock"]
        WS1["host workspace\n(bind mount)"]
        SP -- "creates via docker.sock" --> SJ
        SP -. "docker.sock mounted" .-> DS1
        SP -- "workspace bind mount" --> WS1
        SJ -- "isolated workspace view\nread-write job bind" --> WS1
        SJ -- "PARENT_URL\ntcp://server:8004\n(Docker DNS)" --> SP
    end

    subgraph client_machine["Client Machine"]
        CP["CP container\n(name: site-1)\nnvflare-site:latest"]
        CJ["CJ container\nnvflare-job:latest"]
        DS2["/var/run/docker.sock"]
        WS2["host workspace\n(bind mount)"]
        CP -- "creates via docker.sock" --> CJ
        CP -. "docker.sock mounted" .-> DS2
        CP -- "workspace bind mount" --> WS2
        CJ -- "isolated workspace view\nread-write job bind" --> WS2
        CJ -- "PARENT_URL\ntcp://site-1:8102\n(Docker DNS)" --> CP
    end

    CP -- "FL comms\nhttps://server:8002" --> SP
    Admin["Admin CLI"] -- "https://server:8002" --> SP
```

### Docker Socket and Security Posture

SP/CP container mounts `/var/run/docker.sock` to create SJ/CJ containers at job time. SJ/CJ containers do **not** get the Docker socket — they have no reason to create further containers. The launcher enforces this boundary.

#### Why not Docker-in-Docker (DinD)?

DinD runs a separate Docker daemon inside the SP/CP container. It does not improve security — it just moves the problem: socket mounting requires Docker socket access (root-equivalent), DinD requires `--privileged` (also root-equivalent). Neither is better than the other from a security standpoint, and DinD adds operational complexity (storage driver conflicts, no shared image cache, heavier).

Socket mounting is the right pragmatic choice for this use case: simpler, better debuggability, SJ/CJ containers visible in `docker ps` on the host.

#### Security comparison

| Mode | Security posture | Notes |
|---|---|---|
| Process | Least privilege — runs as a normal OS user | No Docker involved; job subprocesses inherit the same OS user |
| Docker (standard) | Elevated — socket access is root-equivalent | Mounting `/var/run/docker.sock` lets SP/CP control the Docker daemon, which runs as root |
| Docker (rootless) | Reduced risk — daemon runs as non-root user | Rootless Docker limits the blast radius of socket access; still more privileged than process mode but not root-equivalent |
| K8s | Strongest — RBAC, least-privilege ServiceAccount | Proper security primitives; cluster operator controls what each pod can do |

**With standard Docker (the default), Docker mode requires more privilege than process mode.** The Docker daemon runs as root, and mounting its socket gives SP/CP root-equivalent access to the host — it can create privileged containers, bind-mount the host filesystem, etc. Process mode has no such requirement: SP/CP runs as a normal OS user with no elevated access.

**With rootless Docker**, the daemon runs as a non-root user, so socket access does not give root-equivalent. This significantly reduces the risk. However, rootless Docker has limitations (restricted networking, no `--privileged` containers) and requires explicit setup — it is not the default.

**The value of Docker mode is dependency isolation** — running different job images with different ML framework or CUDA versions on the same machine — not security. Site operators who need stronger security guarantees should use the K8s launcher.

SJ/CJ containers do not receive the Docker socket, which limits lateral movement if a job container is compromised. But SP/CP itself has elevated host access by design (standard Docker) or reduced-but-still-elevated access (rootless Docker).

### Resource Management

Docker mode should use **`PassthroughResourceManager` — it approves every resource request unconditionally; no pre-scheduling check is done at scheduling time. If a requested resource (e.g. GPU) is unavailable, the job container fails at startup.

`GPUResourceManager` is not suitable for Docker mode: it would require the SP/CP container itself to have GPU passthrough just to count available GPUs, but SP/CP never uses GPUs — it only manages the federation.

The default provisioning injects `GPUResourceManager`. If a site is **exclusively** running Docker-mode jobs, override it in `workspace/local/resources.json` **before starting SP/CP** — the resource manager is loaded once at startup and is not re-read at job submission time:

```json
{
  "components": [
    {
      "id": "resource_manager",
      "path": "nvflare.app_common.resource_managers.passthrough_resource_manager.PassthroughResourceManager",
      "args": {}
    }
  ]
}
```

Remove the `resource_consumer` component (`GPUResourceConsumer`) as well — it is only needed alongside `GPUResourceManager`.

If a job package is intended to be portable across deployments and carries both Docker- and process-related resource entries, the site's configured launcher still determines which mode is active at runtime. On a Docker-configured site, prefer `PassthroughResourceManager` because the Docker launcher bypasses the process-mode GPU scheduler and passes GPU settings directly via `device_requests`.

**Future:** A `DockerResourceManager` that queries the host GPU inventory without needing SP/CP GPU passthrough is a natural follow-up.

### Workspace / Storage
- SP/CP containers receive a read-write bind mount of the host workspace directory.
- SJ/CJ containers receive an empty tmpfs workspace root at `/var/tmp/nvflare/workspace` with `1777`
  permissions, read-only bind mounts for `startup/` and `local/`, and a read-write bind mount of only the
  current job directory at `/var/tmp/nvflare/workspace/<job_id>`.
- The container-internal workspace mount point is always `/var/tmp/nvflare/workspace` (hardcoded).
- Docker mode does not need workspace transfer: the job sees startup/local files and its own extracted app directly through bind mounts, while Docker prevents it from reading or writing other job directories through the workspace.
- SJ/CJ containers use the current job workspace as their process working directory, so relative job outputs persist on the host.

```
workspace/               ← read-write in SP/CP; not mounted wholesale into SJ/CJ
  startup/               ← read-only in SJ/CJ
  local/
    study_data.yaml      ← read-only in SJ/CJ
  job_001/               ← over-mounted read-write only for job_001's SJ/CJ
  job_002/               ← over-mounted read-write only for job_002's SJ/CJ
```

### Custom Code (BYOC)

Both custom code modes work in Docker mode with no extra configuration:

- **Job-level** (`app/custom/` in job zip) — extracted to `workspace/<job-id>/app_<site>/custom/` at job time. `DockerJobLauncher` sets this as the first entry on `PYTHONPATH` inside SJ/CJ containers.
- **Site-level** (`workspace/local/custom/`) — shared code across all jobs. `DockerJobLauncher` appends this to `PYTHONPATH` in SJ/CJ containers (after job-level, so job code takes precedence on name collision). Same priority as process mode.

Alternatively, site-level code can be baked into the job image — it will be importable as a regular installed package.

### Host Workspace Path

`DockerJobLauncher` needs the **host path** of the workspace to pass to the Docker daemon as a volume bind source. `start_docker.sh` resolves this at startup and passes it via `NVFL_DOCKER_WORKSPACE`:

```bash
HOST_WORKSPACE="$(cd "$DIR/.." && pwd)"
docker run ... -e NVFL_DOCKER_WORKSPACE="$HOST_WORKSPACE" ...
```

`DockerJobLauncher.__init__` reads `NVFL_DOCKER_WORKSPACE` if `workspace` is not set in `resources.json`. Docker connectivity and network existence are validated lazily on first `launch_job` — not at init time — so SJ/CJ containers can load the component without needing Docker access.

### Container Permissions

```
SP/CP container (site admin grants via start_docker.sh)
  ├── /var/run/docker.sock mounted            ← can create job containers
  ├── --user $(id -u):$(id -g)               ← runs as calling user (workspace files not root-owned)
  ├── --group-add <docker-socket-gid>         ← grants socket access; omitted when GID is 0 or unavailable (macOS Docker Desktop)
  ├── workspace bind mount at /var/tmp/nvflare/workspace
  ├── nvflare-network                         ← intra-site: SP↔SJ / CP↔CJ (PARENT_URL, Docker DNS)
  └── host network (-p fed_learn_port)        ← cross-site: CP→SP over HTTPS, same as process mode

SJ/CJ container (DockerJobLauncher controls)
  ├── NO Docker socket                        ← cannot create further containers
  ├── empty tmpfs workspace root at /var/tmp/nvflare/workspace (1777 mode)
  ├── startup bind mount at /var/tmp/nvflare/workspace/startup (read-only)
  ├── local bind mount at /var/tmp/nvflare/workspace/local (read-only)
  ├── job workspace bind mount at /var/tmp/nvflare/workspace/<job_id> (read-write)
  ├── optional data bind mounts at /data/<study>/<dataset>
  └── nvflare-network                         ← intra-site to SP/CP only (PARENT_URL)
```

---

## Job Configuration Reference

All Docker-mode job configuration lives in `meta.json`. This is the single reference for job authors.

A complete example:

```json
{
  "name": "my-fl-job",
  "study": "study_a",
  "deploy_map": {
    "app": ["server", "site-1", "site-2"]
  },
  "launcher_spec": {
    "default": {
      "docker": {"image": "nvflare-pt:latest"}
    },
    "site-1": {
      "docker": {
        "image": "nvflare-pt:latest",
        "shm_size": "8g",
        "ipc_mode": "host"
      }
    }
  },
  "resource_spec": {
    "site-1": {"num_of_gpus": 1}
  },
  "min_clients": 1
}
```

`deploy_map` is launcher-agnostic — it maps app names to site name lists, exactly as in process mode. Launcher-specific configuration lives in `launcher_spec`; GPU placement hints for the scheduler live in `resource_spec`.

A job may carry launcher entries for multiple modes (e.g. both `docker` and `k8s`) for portability across deployments, but the active launcher is determined only by the site's configured launcher in `resources.json`.

### Job Image

The `image` field in `launcher_spec[site][docker]` specifies the Docker image for SJ/CJ containers on that site. A `default` entry under `launcher_spec` applies to all sites not explicitly listed:

- Per-site. Different sites can specify different images.
- The site admin must pull or build the image before the job runs. The launcher does not pull images.
- If no `image` is resolvable for a site that has `DockerJobLauncher` configured, the job fails immediately with a clear error. There is no silent fallback to process mode.

### GPU and Additional Container Flags

Docker-specific runtime flags live under `launcher_spec[site][docker]`. Resource requests such as `num_of_gpus` remain in `resource_spec`, the same as process-mode jobs. Docker SDK keys use underscores, not hyphens:

```json
"launcher_spec": {
  "site-1": {
    "docker": {
      "image": "nvflare-pt:latest",
      "shm_size": "8g",
      "ipc_mode": "host"
    }
  }
},
"resource_spec": {
  "site-1": {
    "num_of_gpus": 1
  }
}
```

`DockerJobLauncher` translates the flat `resource_spec[site].num_of_gpus` field to `device_requests: [{"Count": N, "Capabilities": [["gpu"]]}]` before calling `docker run`. For fine-grained control (specific GPU UUIDs, driver constraints), set `device_requests` directly in the Docker launcher spec.

New jobs should put launcher/container settings in `launcher_spec` and scheduler resource requests such as `num_of_gpus` in `resource_spec`. Do not use `resource_spec[site][docker]` for new metadata; that shape mixes scheduler resources with launcher settings and was only part of earlier migration experiments.

Job-level `launcher_spec[site][docker]` is merged with site-level defaults from `default_job_container_kwargs` in `local/resources.json`; job-level wins on conflict. Reserved keys controlled by the launcher (`volumes`, `mounts`, `network`, `environment`, `command`, `name`, `detach`, `user`, `working_dir`) cannot be overridden.

Site-level default environment variables can be set with `default_job_env` in `local/resources.json`. Launcher-controlled variables like `USER`, `HOME`, and `PYTHONPATH` still take precedence.

Site-level defaults (set by site admin in `local/resources.json`):

```json
{
  "id": "docker_launcher",
  "path": "nvflare.app_opt.job_launcher.docker_launcher.ClientDockerJobLauncher",
  "args": {
    "default_job_container_kwargs": {"ipc_mode": "host"},
    "default_job_env": {"NCCL_P2P_DISABLE": "1"}
  }
}
```

### Dataset / Study Data

Set `"study"` in `meta.json` to the name of the study whose data the job needs:

```json
{ "study": "study_a" }
```

The site admin creates `workspace/local/study_data.yaml` mapping study and dataset names to host data paths:

```yaml
study_a:
  training:
    source: /host/data/study_a/training
    mode: ro
  output:
    source: /host/data/study_a/output
    mode: rw
default:
  training:
    source: /host/data/default/training
    mode: ro
```

At launch time, `DockerJobLauncher` looks up the study name and bind-mounts each configured dataset into the SJ/CJ container at `/data/<study>/<dataset>`. In Docker mode, `source` is the host path passed to Docker and `mode` must be `ro` or `rw`. If the file doesn't exist or the study has no entry, no data volume is added and the launcher logs a warning.

This YAML schema replaces the legacy flat `study -> path` map. A stale flat-format file now fails validation instead of being ignored. If a configured dataset host `source` path does not exist, Docker reports the bind-mount failure when the job container is created.

---

## Job Launch Sequence

```mermaid
sequenceDiagram
    participant Admin as Admin CLI
    participant SP as SP/CP Container
    participant Launcher as DockerJobLauncher
    participant Docker as Docker Daemon
    participant SJ as SJ/CJ Container

    Admin->>SP: submit job (meta.json)
    SP->>SP: schedule job
    SP->>Launcher: fire BEFORE_JOB_LAUNCH event
    Launcher->>SP: add_launcher(self, fl_ctx)
    SP->>Launcher: launch_job(job_meta, fl_ctx)
    Launcher->>Launcher: read image from launcher_spec[site][docker]
    Launcher->>Launcher: override PARENT_URL → SP/CP container name
    Launcher->>Docker: containers.run(job_image, command, network, mounts, ...)
    Docker->>SJ: start container
    SJ->>SP: connect via PARENT_URL (Docker DNS)
    SJ->>SJ: run FL training
    SJ-->>Docker: exit (rc=0 or rc=1)
    Launcher->>Docker: poll container status
    Docker-->>Launcher: exited (rc=0)
    Launcher->>Launcher: set terminal_state = SUCCESS
    Launcher->>Docker: remove container
    SP->>SP: job complete, results in workspace
```

---

## End-to-End Operation

### Prerequisites — Build Docker images (site admin responsibility)

Building images is not part of the NVFlare deployment workflow. The site admin must build two categories of images independently before deploying:

**SP/CP image** (runs the NVFlare process, needs Docker SDK to launch job containers):
```dockerfile
FROM python:3.12
RUN pip install nvflare docker
# add site-specific dependencies if needed
```

**Job image** (runs the actual FL training, needs ML frameworks but not Docker SDK):
```dockerfile
FROM python:3.12
RUN pip install nvflare torch torchvision
# add job-specific ML dependencies
```

Deploy preparation writes the SP/CP image name into `start_docker.sh`. It does not build, tag, or push any images.

### Step 1 — Prepare the Docker Runtime Kit

Provision normally to create a runtime-neutral server or client startup kit. Then create a site-local Docker runtime config:

```yaml
runtime: docker
parent:
  docker_image: nvflare-site:latest
  network: nvflare-network
job_launcher:
  default_python_path: /usr/local/bin/python
```

Prepare the kit:

```bash
nvflare deploy prepare prod_00/site-1 --config docker.yaml --output prepared/site-1
```

The prepared kit gets:
- `startup/start_docker.sh` - Docker mode startup script
- `local/resources.json.default` - process-mode launcher replaced with `DockerJobLauncher`
- `local/comm_config.json` - internal parent communication adjusted for Docker networking
- `local/study_data.yaml` - template used by job containers for study data mounts

### Step 2 — Persist job storage (server only)

By default, NVFlare stores job zips, results, and snapshots under `/tmp/nvflare/` inside the container. This directory is ephemeral — it is lost when the SP container stops. To survive container restarts, redirect these paths to the workspace bind mount by creating `workspace/server/local/resources.json`:

```json
{
  "server": {
    "job_manager": {
      "uri_root": "/var/tmp/nvflare/workspace/jobs-storage"
    },
    "snapshot_persistor": {
      "storage": {
        "root_dir": "/var/tmp/nvflare/workspace/snapshot-storage"
      }
    }
  }
}
```

These paths resolve to `workspace/jobs-storage/` and `workspace/snapshot-storage/` on the host — both inside the bind mount and therefore persistent across container restarts.

For SJ containers launched by `DockerJobLauncher`, the same paths exist under the isolated tmpfs workspace root.
They are writable so server-job startup can initialize, but they are ephemeral and are not connected to the parent
server's host-backed storage. The tmpfs uses sticky `1777` permissions so the non-root job-container user can
initialize these directories even when Docker owns the tmpfs root as root.

### Step 3 — Start SP/CP in Docker mode

On the server machine:
```bash
cd workspace/server/startup
nohup ./start_docker.sh > server.log 2>&1 &
# → creates nvflare-network if it doesn't exist
# → docker run --name server \
#              --user "$(id -u):$(id -g)" \
#              --group-add <docker-socket-gid> (if non-zero) \
#              --network nvflare-network \
#              -v $HOST_WORKSPACE:/var/tmp/nvflare/workspace \
#              -v /var/run/docker.sock:/var/run/docker.sock \
#              -e NVFL_DOCKER_WORKSPACE=$HOST_WORKSPACE \
#              -p 8002:8002 \
#              --rm nvflare-site:latest \
#              /var/tmp/nvflare/workspace/startup/sub_start.sh
```

On each client machine:
```bash
cd workspace/site-1/startup
nohup ./start_docker.sh > site-1.log 2>&1 &
```

To use a different image without re-provisioning:
```bash
NVFL_P_IMAGE=nvflare-site:2.7.2 ./start_docker.sh
```

### Step 4 — Configure site data (optional)

If jobs need access to study data, create `workspace/local/study_data.yaml`. No change to `resources.json` is needed — `DockerJobLauncher` reads this file fresh on every job launch, so entries can be added or updated at any time without restarting SP/CP.

```yaml
study_a:
  training:
    source: /host/data/study_a/training
    mode: ro
```

Top-level keys are study names from `meta.json`; nested keys are dataset names. Each dataset entry defines the host `source` path and mount `mode`. Dataset names appear in the container path as `/data/<study>/<dataset>`. If the file is absent or the job's study has no entry, no data volume is added, the launcher logs a warning, and the job runs without a data mount. Legacy flat `study -> path` entries are invalid in this schema.

### Step 5 — Submit a job

See [Job Configuration Reference](#job-configuration-reference) for all available fields. Example:

```bash
nvflare job submit -j /path/to/job
```

### Step 6 — Results

Job output is written to `/var/tmp/nvflare/workspace/{job_id}/` inside the container. That job directory is a read-write bind mount from the host, while other job directories are not present in the job container's workspace view:

```
workspace/server/job_001/      ← server-side artifacts
workspace/site-1/job_001/      ← client-side artifacts
```

---

## Class Design

Docker launcher follows the same pattern as `K8sJobLauncher` — not `ProcessJobLauncher` — because Docker containers, like K8s pods, are named execution units that persist after launch and require explicit cleanup.

```
JobHandleSpec (ABC)
  ├── ProcessHandle
  ├── K8sJobHandle
  └── DockerJobHandle        ← models terminal_state pattern from K8sJobHandle

JobLauncherSpec (FLComponent, ABC)
  ├── ProcessJobLauncher
  │     ├── ServerProcessJobLauncher
  │     └── ClientProcessJobLauncher
  ├── K8sJobLauncher
  │     ├── ServerK8sJobLauncher
  │     └── ClientK8sJobLauncher
  └── DockerJobLauncher       ← abstract: get_module_args()
        ├── ServerDockerJobLauncher
        └── ClientDockerJobLauncher
```

### DockerJobHandle state mapping

Consistent with process and K8s launchers — `UNKNOWN` means "not terminal yet, keep waiting":

| Docker state | JobReturnCode | Notes |
|---|---|---|
| `created` | `UNKNOWN` | Not running yet |
| `running` | `UNKNOWN` | In progress |
| `paused` | `UNKNOWN` | In progress |
| `restarting` | `UNKNOWN` | In progress |
| `exited` (code 0) | `SUCCESS` | |
| `exited` (code != 0) | `EXECUTION_ERROR` | Read from `container.attrs["State"]["ExitCode"]` |
| `dead` | `ABORTED` | Killed externally |

Note: Docker exposes the actual exit code (same as subprocess), so `EXECUTION_ERROR` is distinguishable from `ABORTED` — unlike K8s which only has pod phase.

### DockerJobHandle `terminal_state` pattern

Mirrors `K8sJobHandle`. Once a container exits or is removed, `terminal_state` is set and all subsequent `poll()` / `wait()` calls return immediately without querying Docker.

### Known Issue: Job Container May Linger After Job Completion

There is a pre-existing NVFlare shutdown issue where non-daemon threads can keep the SJ/CJ process alive after the job has logically finished. In Docker mode this is especially visible because the job process is PID 1 in the container: if PID 1 does not exit, the container stays running.

This PR does **not** add a `shutdown_timeout` workaround because force-terminating the container on a timer is not a reliable fix and can mask the real upstream shutdown problem. If the issue occurs, the job container may remain present after job completion until it is removed manually.

Operationally, the site admin can recover with `docker rm -f <container_name>`. See `docs/design/docker_container_shutdown_analysis.md` for the root-cause analysis and suggested upstream fixes.

### PARENT_URL

In process mode `PARENT_URL = tcp://localhost:port`. In Docker mode it must be the SP/CP **container name** on the Docker network (e.g. `tcp://server:8004`) so SJ/CJ can connect back via Docker DNS.

`DockerJobLauncher` derives the correct `parent_url` at runtime: it takes the port from `PARENT_URL` in `JOB_PROCESS_ARGS` and combines it with the site name (which equals the container name). No provisioning-time baking needed.

---

## Key Implementation Files

| File | Role |
|---|---|
| `nvflare/app_opt/job_launcher/docker_launcher.py` | `DockerJobHandle`, `DockerJobLauncher`, `ClientDockerJobLauncher`, `ServerDockerJobLauncher` |
| `nvflare/tool/deploy/deploy_cli.py` | CLI entry point for `nvflare deploy prepare` |
| `nvflare/tool/deploy/deploy_commands.py` | Docker runtime validation, resource mutation, and `start_docker.sh` generation |

### Docker SDK import

`docker` and `docker.errors` are imported at module level inside a `try/except ImportError` block:

```python
try:
    import docker
    import docker.errors
    _DOCKER_AVAILABLE = True
except ImportError:
    _DOCKER_AVAILABLE = False
```

**Why:** `DockerJobLauncher` is injected into `local/resources.json` and loaded by all processes on a site — including SJ/CJ job containers. Job containers have no reason to call Docker APIs and do not need the `docker` SDK installed. The try/except means the module loads cleanly on containers without the SDK; `_DOCKER_AVAILABLE` is only checked in `_get_docker_client()`, which is only called from SP/CP via `launch_job()`.

This is a temporary solution. The root cause is that `resources.json` is shared across SP/CP and job containers — a future refactor could avoid loading SP/CP-only components in job containers entirely.

---

## Other Container Runtimes

`DockerJobLauncher` is Docker-specific (uses the `docker` Python SDK). Other runtimes:

| Runtime | Compatibility | Notes |
|---|---|---|
| **Podman** | Near drop-in | Emulates Docker socket API; `docker.from_env()` works if Podman socket is configured. No code change needed. |
| **Singularity/Apptainer** | Requires new launcher | No daemon, no SDK — CLI only (`singularity exec`). Runs in host network so `PARENT_URL = localhost:port` (no Docker DNS needed). Common in HPC where Docker is banned. |
| **rootless Docker** | Works as-is | Reduced privilege vs. standard Docker; setup is non-default. See security section. |

When a second runtime needs support, the right refactor is to extract a `ContainerJobLauncher` base class with `_run_container()` / `_get_status()` / `_stop_container()` as the runtime-specific interface. Everything above that (PARENT_URL override, PYTHONPATH, GPU, event handling) is runtime-agnostic and stays in the base.

---

## Design Considerations

The following decisions were made during the design review of this PR and inform the
recommended follow-up work.

### Decision 1: Provisioning model — distributed site-local preparation

**Chosen: site-local `nvflare deploy prepare`.**

Provisioning generates the same startup kit as today, without Docker or K8s
awareness. Each site admin runs `nvflare deploy prepare` as a second step to opt
that site into Docker or K8s mode. The tool generates:
- `start_docker.sh` (Docker mode) or Helm charts (K8s mode) for that site
- Updates `resources.json` with the right launcher and resource manager

**Why:** the project admin does not need to know each site's runtime environment
at provision time. Site admins are independent — one site may use Docker,
another K8s, another plain process. Provisioning is decoupled from launch mode,
which is a site-local concern.

### Decision 2: Launchers are mutually exclusive; parent and job always use the same runtime

Each site runs in exactly one launcher mode. The launcher determines how SJ/CJ are started
relative to SP/CP.

**Process launcher** (existing): SJ/CJ are subprocesses of SP/CP, sharing the same execution
environment. This works on bare metal, inside a Docker container, or inside a K8s pod —
the process launcher doesn't care about the host environment.

```
Process launcher (bare metal):  SP/CP (host process) ↔ SJ/CJ (subprocess, same host)
Process launcher (in Docker):   SP/CP (container)    ↔ SJ/CJ (subprocess, same container)
Process launcher (in K8s pod):  SP/CP (pod)          ↔ SJ/CJ (subprocess, same pod)
```

**Docker launcher** (new): SJ/CJ are started as **separate containers** per job, each with
their own image. SP/CP must itself be a container to reach SJ/CJ via Docker DNS.

```
Docker launcher:  SP/CP (container) ↔ SJ/CJ (new container per job, own image)
```

**K8s launcher** (existing): SJ/CJ are started as **separate pods** per job.

```
K8s launcher:  SP/CP (pod) ↔ SJ/CJ (new pod per job, own image)
```

The value of Docker/K8s launchers over process launcher is **per-job isolation**: each job
runs in its own environment (image), independent of SP/CP and other jobs. Process launcher
inside Docker/K8s gives container isolation for SP/CP but all jobs still share that same
container/pod environment.

Mixed launcher mode within a site is not supported. The `mode` flag (see Decision 3) enforces
this — a site is committed to one launcher.

### Decision 3: `mode` flag in nvflare tool (follow-up PR)

The nvflare tool (Decision 1) accepts a `--mode docker|k8s|process` flag per site. This
single flag sets everything needed for that mode:

| `mode` | Launcher | Resource manager | Resource consumer | Start script |
|---|---|---|---|---|
| `process` | `ProcessJobLauncher` | `GPUResourceManager` | `GPUResourceConsumer` | `start.sh` (unchanged) |
| `docker` | `DockerJobLauncher` | `PassthroughResourceManager` | none | `start_docker.sh` |
| `k8s` | `K8sJobLauncher` | `PassthroughResourceManager` | none | Helm charts |

The tool removes **both** `GPUResourceManager` and `GPUResourceConsumer` for Docker/K8s mode
and replaces them with `PassthroughResourceManager`. This is important: `GPUResourceConsumer` sets
`CUDA_VISIBLE_DEVICES` in the SP/CP process
environment, which is correct for process mode (subprocess inherits it) but wrong for
Docker/K8s mode (SJ/CJ runs in a separate container/pod with its own environment — GPU
assignment is handled by `device_requests` / pod resource limits instead). Site admin cannot
accidentally leave `GPUResourceConsumer` in place when the tool handles both together.

### Decision 4: Parent image is thin; job image is required with no default

**Parent image (SP/CP):** as thin as possible — only needs NVFlare and the Docker/K8s Python
SDK. No ML frameworks. This minimizes the image the site admin has to maintain and keeps
SP/CP lightweight. It can also serve as the job image for simple jobs (e.g. hello-numpy) that
have no ML framework dependencies.

**Job image:** the job author must explicitly specify an image in `meta.json` for Docker/K8s
mode. There is no default. If no image is specified and the site is in Docker/K8s mode, the
job fails with a clear error — no silent fallback to process mode or parent image.

Rationale: the site admin does not know which image a job needs (there could be many: different
CUDA versions, different frameworks). The job author is the right person to declare the
environment. Requiring an explicit image surfaces misconfiguration early rather than silently
running a job in the wrong environment.

The thin parent image is a known-good option for jobs that don't need ML frameworks — job
authors can specify it explicitly if that is sufficient.

### Decision 5: Docker mode vs process mode

Neither is universally better — the right choice depends on the deployment context:

| | Docker mode | Process mode |
|---|---|---|
| Job isolation | Per-job container; each job gets its own image | Shared SP/CP environment; all jobs see the same Python packages |
| Custom environments | Different image (Python version, CUDA, ML framework) per job | Must match SP/CP environment |
| Startup overhead | Container launch (~seconds, longer with image pull) | Subprocess (~instant) |
| GPU access | Via Docker `device_requests`; requires `PassthroughResourceManager` | Direct; `GPUResourceManager` + `GPUResourceConsumer` assign `CUDA_VISIBLE_DEVICES` |
| File permissions | Requires `--user` alignment between SP/CP and SJ/CJ | Native; no extra setup |
| Operational complexity | Higher (Docker network, socket mount, image management, container lifecycle) | Lower |
| Security posture | Elevated (socket access is root-equivalent with standard Docker) | Least privilege (normal OS user) |


**Use Docker mode when:** jobs need different Python/CUDA environments, or dependency
isolation between jobs is important.

**Use process mode when:** the environment is homogeneous, startup speed matters, or
simplicity is preferred.

---

## Out of Scope

- Docker Swarm / multi-host Docker — use K8s launcher instead
- Building or pushing job images — site admin's responsibility
- Production-scale deployments — use K8s launcher instead
- Mixed mode within a site (SP/CP as process, SJ/CJ as container) — not supported
- Orphaned container recovery on SP/CP restart — future work
- Singularity/Podman support — future extension point, not blocking Docker mode
