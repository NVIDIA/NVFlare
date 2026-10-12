# Docker Job Launcher Example

End-to-end example of running NVFlare in Docker mode using `nvflare deploy prepare`.
SP/CP containers are started manually; SJ/CJ containers are launched automatically per job.

## Prerequisites

- Docker with a working daemon
- NVFlare installed on the host for provisioning and the admin CLI (from the
  repo root: `python -m pip install -e .`). PyTorch and tracking dependencies
  are installed in the job image.
- Run all commands from the `examples/docker` directory unless noted otherwise

Download the example without cloning NVFlare:

```bash
nvflare examples get docker-runtime
cd docker-runtime/examples/docker
```

From a source checkout, use `cd examples/docker` instead.

### Apple Silicon Mac

Choose Docker Desktop or Colima below, then install the host CLI.
For `hello-pt-docker`, use the separate CPU job copy in Step 5 when CUDA is
unavailable. The client selects CUDA when available and CPU otherwise; it
does not select MPS. Other Colima GPU backends and MPS execution have not
been validated with this main example.

#### Docker Desktop

Start Docker Desktop. In **Settings > Advanced**, enable
**Allow the default Docker socket to be used**. This creates the macOS
`/var/run/docker.sock` symlink; see
[Docker's Mac permission requirements](https://docs.docker.com/desktop/setup/install/mac-permission-requirements/).
The generated parent startup script mounts that socket from the Linux daemon
into each parent container. Select Docker Desktop's Linux daemon and check
connectivity before provisioning:

```bash
unset DOCKER_HOST DOCKER_CONTEXT
docker context use desktop-linux
docker info
test -S /var/run/docker.sock && echo "Docker socket is available"
```

Continue after `docker info` succeeds and the socket check prints
`Docker socket is available`.

If a build reports that `docker-credential-desktop` cannot be found, add
Docker Desktop's bundled tools to this shell's PATH:

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

#### Colima

Colima supplies a Linux Docker daemon without Docker Desktop. Homebrew's
`docker` package supplies the standalone CLI. Use Colima's VZ VM for CPU
execution on Apple Silicon:

```bash
brew install colima docker
colima start --profile nvflare-docker --activate=false --vm-type vz --cpu 4 --memory 8 --disk 40 \
  --runtime docker --mount "$(cd ../.. && pwd):w"
export DOCKER_CONTEXT=colima-nvflare-docker
unset DOCKER_HOST
docker info
```

The writable mount must cover the source checkout (or downloaded example)
and its workspace. The generated startup script probes the socket inside the daemon VM to obtain its group ID.
If you choose another profile name, update `DOCKER_CONTEXT` accordingly.
#### Host CLI setup

After selecting either runtime, install the core host CLI in a virtual
environment. From a source checkout, run these commands from `examples/docker`:

```bash
brew install python@3.13
cd ../..
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cd examples/docker
```

For a downloaded example, create and activate a virtual environment and install
`nvflare` with `python -m pip install nvflare` before running
`nvflare examples get docker-runtime` as shown above. The image build still
uses the downloaded example's exact revision. Run the remaining steps in the
same shell with the chosen Docker runtime and virtual environment active.
If port 8002 is occupied, copy `project.yml`, choose a free `fed_learn_port`,
and provision with that copy in Step 1.

## Step 0: Build Docker images

```bash
bash build_docker.sh
```

The build script uses the current checkout when one is present. For an example
download, it reads the exact revision from `.nvflare-example.json` and prepares
a temporary shallow checkout automatically. The checkout is deleted when the
build finishes, so the resulting images contain the same NVFlare source that
supplied the example.

If version metadata is unavailable, provide `NVFL_BASE_VERSION` as an advanced
override:

```bash
NVFL_BASE_VERSION=2.9.0 bash build_docker.sh
```

Use only the base release number, such as `2.9.0`; the package build adds its
development suffix when Git metadata is unavailable.

This builds two images:
- `nvflare-site:latest` — used by SP/CP containers (started by `start_docker.sh`),
  built from this example's `Dockerfile`
- `nvflare-job:latest` — used by SJ/CJ containers (launched automatically per job),
  built from this example's `Dockerfile.nvflare-job`

## Step 1: Provision

```bash
nvflare provision -p project.yml
```

This generates a workspace under `workspace/docker_test_project/` relative to the current directory.

## Step 2: Prepare Docker runtime kits

Prepare the server and both client startup kits for Docker mode:

```bash
nvflare deploy prepare \
  workspace/docker_test_project/prod_00/server \
  --config docker.yaml \
  --output workspace/docker_test_project/prepared/server

nvflare deploy prepare \
  workspace/docker_test_project/prod_00/site-1 \
  --config docker.yaml \
  --output workspace/docker_test_project/prepared/site-1

nvflare deploy prepare \
  workspace/docker_test_project/prod_00/site-2 \
  --config docker.yaml \
  --output workspace/docker_test_project/prepared/site-2
```

The prepared kits include `startup/start_docker.sh`, Docker launcher resources,
and a `local/study_runtime.yaml` template. The generated start script creates
the Docker network if needed.

## Step 3: Configure host admin connectivity

Skip this step if `server` already resolves via your DNS. Otherwise, add the following
to `/etc/hosts` so the admin CLI can reach the server container by name:

```
127.0.0.1  server
```

On macOS, the admin kit can instead use the published loopback port without
editing `/etc/hosts` (including when `server` already resolves to another host):

```bash
python - <<'PYADMIN'
import json
from pathlib import Path

path = Path("workspace/docker_test_project/prod_00/admin@nvidia.com/startup/fed_admin.json")
config = json.loads(path.read_text())
config["admin"]["host"] = "127.0.0.1"
path.write_text(json.dumps(config, indent=2) + "\n")
PYADMIN
```

The server's certificate identity remains the provisioned server name. Both
Docker clients use the server's Docker network name; keep their kit addresses.

## Step 4: Start server and clients

The server and both clients run in Docker mode. Their parent containers use
`start_docker.sh`, and each site launches its per-job process in a separate Docker container.

The first `start_docker.sh` command creates `nvflare-network` if it does not
already exist, so no separate `docker network create` command is required.

Start the server from the `examples/docker` directory:

```bash
(
  cd workspace/docker_test_project/prepared/server
  nohup bash startup/start_docker.sh > server.log 2>&1 < /dev/null &
)
```

Wait for the server container to start (`docker inspect --format
'{{.State.Running}}' server` should print `true`), then start both clients:

```bash
(
  cd workspace/docker_test_project/prepared/site-1
  nohup bash startup/start_docker.sh > site-1.log 2>&1 < /dev/null &
)
(
  cd workspace/docker_test_project/prepared/site-2
  nohup bash startup/start_docker.sh > site-2.log 2>&1 < /dev/null &
)
```

You can watch startup logs with:

```bash
tail -f \
  workspace/docker_test_project/prepared/server/server.log \
  workspace/docker_test_project/prepared/site-1/site-1.log \
  workspace/docker_test_project/prepared/site-2/site-2.log
```

Before submitting a job, confirm both clients are connected:

```bash
nvflare system status \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

## Step 5: Submit a job

```bash
nvflare job submit \
  -j jobs/hello-numpy-docker \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

The original `hello-pt-docker` requests one GPU per client. On any platform, create a
copy without that resource requirement to allow CPU execution when CUDA is
unavailable, preserving the client's existing device selection:

```bash
python - <<'PYCPU'
import json
import shutil
from pathlib import Path

shutil.copytree("jobs/hello-pt-docker", "workspace/hello-pt-docker-cpu", dirs_exist_ok=True)
path = Path("workspace/hello-pt-docker-cpu/meta.json")
meta = json.loads(path.read_text())
meta["resource_spec"] = {}
path.write_text(json.dumps(meta, indent=4) + "\n")
PYCPU
nvflare job submit \
  -j workspace/hello-pt-docker-cpu \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

The supplied FedAvg controller samples one client per round; the same site
can be selected in both rounds. Checking both clients confirms availability,
not that both train. The CPU copy changes only `resource_spec` and preserves
that sampling behavior. A separate validation control can set the copied
controller's `num_clients` to 2 to exercise both sites in every round.

The job image provides a writable `/var/tmp/nvflare/data` CIFAR-10 cache for
non-root job users. To override it, set `NVFL_CIFAR10_ROOT` in
`job_launcher.default_job_env` in `docker.yaml` before preparing both client
kits, and choose a directory writable inside their job containers. Exporting
this variable only on the host does not configure the Docker jobs. Data uses
torchvision's standard CIFAR-10 download URL and checksum validation.

The default image cache and the job's temporary workspace root disappear when
the job container exits. For a reusable cache, configure writable study dataset
mounts in both prepared client kits before starting them. This example keeps
each client's cache in its prepared workspace on the host, which the Colima
writable mount above covers:

```bash
python - <<'PYCACHE'
import tempfile
from pathlib import Path

import yaml

root = Path("workspace/docker_test_project/prepared").resolve()
for site in ("site-1", "site-2"):
    cache = root / site / "cifar10-cache"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=cache):
        pass
    path = root / site / "local/study_runtime.yaml"
    config = yaml.safe_load(path.read_text()) or {}
    study = config.setdefault("studies", {}).setdefault("default", {})
    study.setdefault("datasets", {})["cifar10"] = {"source": str(cache), "mode": "rw"}
    study.setdefault("env", {})["NVFL_CIFAR10_ROOT"] = "/data/default/cifar10"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
print("Both persistent caches are writable and configured")
PYCACHE
```

Each job mounts the corresponding host cache at `/data/default/cifar10`.
This study environment overrides the site-wide cache variable. Job kit
`local` and `startup` directories are read-only, so keep download/extraction
caches outside those directories.

Use the returned job ID to check completion:

```bash
nvflare job wait JOB_ID \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

Available jobs:

| Job | Description |
|-----|-------------|
| `hello-numpy-docker` | Basic numpy federated averaging |
| `hello-pt-docker` | PyTorch CIFAR-10 training; requests one GPU by default |
| `pt-ddp-docker` | Multi-GPU DDP training with torchrun |

## Step 6: Stop the example

After jobs finish, shut down this federation:

```bash
nvflare system shutdown all --force --timeout 60 \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

If using Colima, when finished with the dedicated VM, run
`colima stop --profile nvflare-docker`.

## Notes

- Docker launcher settings are specified per-site in `launcher_spec` in
  `meta.json`. Keep resource requests such as `num_of_gpus` in `resource_spec`,
  the same way as process-mode jobs. Example:
  ```json
  "launcher_spec": {
    "site-1": {"docker": {"image": "nvflare-job:latest", "shm_size": "8g"}},
    "site-2": {"docker": {"image": "nvflare-job:latest", "shm_size": "8g"}}
  },
  "resource_spec": {
    "site-1": {"num_of_gpus": 1},
    "site-2": {"num_of_gpus": 1}
  }
  ```
  Every site configured with a Docker job launcher needs either a site-specific `docker`
  entry or a `launcher_spec.default.docker` entry that supplies the job image.
- Site-level Docker defaults (e.g. `shm_size`, `ipc_mode`) can be set via
  `job_launcher.default_job_container_kwargs` in `docker.yaml`; `nvflare deploy prepare`
  writes them to `resources.json`. This example configures `ipc_mode: host` there because
  host-isolation options are site-owned and cannot be supplied by job metadata. Job-level
  `launcher_spec[site][docker]` takes precedence for supported options.
- Some multi-GPU Docker environments may need `NCCL_P2P_DISABLE=1` to avoid NCCL hangs.
  Set this site-wide with `default_job_env` in `resources.json`, for example:
  ```json
  "default_job_env": {"NCCL_P2P_DISABLE": "1"}
  ```
- Parent containers bind-mount the prepared workspace at `/var/tmp/nvflare/workspace`.
  Job containers receive an isolated workspace with read-only kit files and
  their own writable job directory; reusable data uses study dataset mounts.
- Job containers run as the same UID/GID as the SP/CP so all workspace files remain
  readable and writable by the parent process.
- To watch job container logs: `docker logs -f <site>-<job_id>`
