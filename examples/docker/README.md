# Docker Job Launcher Example

End-to-end example of running NVFlare in Docker mode using `nvflare deploy prepare`.
SP/CP containers are started manually; SJ/CJ containers are launched automatically per job.

## Prerequisites

- Docker with a working daemon
- NVFlare, PyTorch, and TensorBoard installed on the host (from the repo root:
  `python -m pip install -e ".[PT]" tensorboard`)
- Run all commands from the `examples/docker` directory unless noted otherwise

### Apple Silicon Mac with Colima

Colima provides the Linux Docker daemon used by the parent and job containers.
Docker Desktop is not required. The Homebrew `docker` package below provides
the Docker CLI.
These steps were tested on Apple Silicon with Colima's `vz` VM and CPU execution.
For `hello-pt-docker`, use the job copy without a GPU requirement described in Step 5.
The PyTorch client selects `cuda:0` when `torch.cuda.is_available()` is true,
and `cpu` otherwise; it does not select `mps`.
[PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html) and
[Colima's krunkit GPU support](https://github.com/abiosoft/colima#ai-models-gpu-accelerated)
have not been tested with this example.

```bash
brew install colima docker python@3.13
colima start --vm-type vz --cpu 4 --memory 8 --disk 40 --runtime docker \
  --mount "$(cd ../.. && pwd):w"
unset DOCKER_HOST
docker context use colima
docker info

# From the NVFlare repo root (Python 3.13 was tested):
cd ../..
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[PT]" tensorboard
cd examples/docker
```

Run the remaining steps in this shell. Colima must be able to mount the repo
directory into its VM. The generated startup script discovers the socket group
inside that VM and gives the parent containers access to launch job containers.
If host port 8002 is already in use, copy `project.yml` to a local file, set
`fed_learn_port` to a free port (for example, 18002), and use that file in
Step 1. The loopback setup in Step 3 reads the chosen port from the kit.

## Step 0: Build Docker images

```bash
bash build_docker.sh
```

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

Prepare the server and site-1 startup kits for Docker mode:

```bash
nvflare deploy prepare \
  workspace/docker_test_project/prod_00/server \
  --config docker.yaml \
  --output workspace/docker_test_project/prepared/server

nvflare deploy prepare \
  workspace/docker_test_project/prod_00/site-1 \
  --config docker.yaml \
  --output workspace/docker_test_project/prepared/site-1
```

The prepared kits include `startup/start_docker.sh`, Docker launcher resources,
and a `local/study_data.yaml` template. The generated start script creates the
Docker network if needed.

## Step 3: Configure connections from the host

On Linux, skip this step if `server` already resolves to your NVFlare server.
Otherwise, add the following to `/etc/hosts` so the admin CLI can reach the
server container by name:

```
127.0.0.1  server
```

On macOS, point the host-side site-2 and admin kits at the published loopback port
using the commands below. This also handles Macs where `server` already names
another host, and requires no `/etc/hosts` edit.
Run this after provisioning and before starting site-2 or submitting a job:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("workspace/docker_test_project/prod_00")
client_path = root / "site-2/startup/fed_client.json"
client = json.loads(client_path.read_text())
target = client["servers"][0]["service"]["target"]
client["servers"][0]["service"]["target"] = "127.0.0.1:" + target.rsplit(":", 1)[1]
client_path.write_text(json.dumps(client, indent=2) + "\n")

admin_path = root / "admin@nvidia.com/startup/fed_admin.json"
admin = json.loads(admin_path.read_text())
admin["admin"]["host"] = "127.0.0.1"
admin_path.write_text(json.dumps(admin, indent=2) + "\n")
PY
```

The server's certificate identity remains the provisioned server name.

## Step 4: Start server and clients

This example runs in **hybrid mode**: site-1 uses Docker job launcher (`start_docker.sh`),
site-2 runs in process mode (`start.sh`). This tests that both modes work together in the
same federation.

The first `start_docker.sh` command creates `nvflare-network` if it does not
already exist, so no separate `docker network create` command is required.

The PyTorch job image provides a writable `/var/tmp/nvflare/data` cache.
On macOS, set the host-side site-2 cache to the workspace directory below
before downloading data or starting site-2. Run these commands as the same
user who will start site-2, in the same shell used for the remaining steps.
On other hosts, use this override if the default cache is not writable.

```bash
export NVFL_CIFAR10_ROOT="$(pwd)/workspace/cifar10-site-2"
mkdir -p "$NVFL_CIFAR10_ROOT"
python - <<'PY'
import os
import tempfile

with tempfile.TemporaryFile(dir=os.environ["NVFL_CIFAR10_ROOT"]):
    pass
print("CIFAR-10 cache is writable")
PY
```

Continue only after the write check succeeds. If it fails, choose a directory
writable by that user and repeat the commands. This host environment variable
sets site-2's cache; site-1's Docker job keeps its writable default cache.

On macOS, download the process client's archive before starting it. The
commands below use curl's retry/resume support and verify the original
CIFAR-10 checksum. They use the download URL provided by torchvision.
Continue only after checksum verification succeeds.
Docker clients download and verify their own data.

```bash
CIFAR10_CACHE="${NVFL_CIFAR10_ROOT:-/var/tmp/nvflare/data}"
CIFAR10_URL="$(python -c 'from torchvision.datasets import CIFAR10; print(CIFAR10.url)')"
mkdir -p "$CIFAR10_CACHE"
curl -fL --retry 5 --retry-all-errors --continue-at - \
  --output "$CIFAR10_CACHE/cifar-10-python.tar.gz" \
  "$CIFAR10_URL"
python - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ.get("NVFL_CIFAR10_ROOT") or "/var/tmp/nvflare/data")
archive = root / "cifar-10-python.tar.gz"
checksum = hashlib.md5(usedforsecurity=False)
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        checksum.update(chunk)
if checksum.hexdigest() != "c58f30108f718f92721af3b95e74349a":
    raise SystemExit("CIFAR-10 checksum mismatch: remove the archive and retry the curl command.")
print("CIFAR-10 checksum verified")
PY
```

Start the server from the `examples/docker` directory:

```bash
(
  cd workspace/docker_test_project/prepared/server
  nohup bash startup/start_docker.sh > server.log 2>&1 < /dev/null &
)
```

Wait until the server container is running before starting the clients.
The following command should print `true`; repeat it if the container is
still starting:

```bash
docker inspect --format '{{.State.Running}}' server
```

Then start both clients:

```bash
(
  cd workspace/docker_test_project/prepared/site-1
  nohup bash startup/start_docker.sh > site-1.log 2>&1 < /dev/null &
)
(
  cd workspace/docker_test_project/prod_00/site-2
  nohup bash startup/start.sh > site-2.log 2>&1 < /dev/null &
)
```

You can watch startup logs with:

```bash
tail -f \
  workspace/docker_test_project/prepared/server/server.log \
  workspace/docker_test_project/prepared/site-1/site-1.log \
  workspace/docker_test_project/prod_00/site-2/site-2.log
```

Before submitting a job, check that both `site-1` and `site-2` are connected:

```bash
nvflare system status \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

Repeat this command if the clients are still starting.

## Step 5: Submit a job

```bash
nvflare job submit \
  -j jobs/hello-numpy-docker \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

The original `hello-pt-docker` job requests one GPU for site-1. To remove that
GPU requirement on any platform, create a copy as shown below. This allows
the job to run on CPU when CUDA is unavailable to the clients, as in the
Colima setup tested above. It preserves the client's existing device selection:
CUDA when available, CPU otherwise.

```bash
python - <<'PY'
import json
import shutil
from pathlib import Path

shutil.copytree("jobs/hello-pt-docker", "workspace/hello-pt-docker-cpu", dirs_exist_ok=True)
path = Path("workspace/hello-pt-docker-cpu/meta.json")
meta = json.loads(path.read_text())
meta["resource_spec"] = {}
path.write_text(json.dumps(meta, indent=4) + "\n")
PY
nvflare job submit \
  -j workspace/hello-pt-docker-cpu \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

When the Docker host exposes an NVIDIA GPU, submit the original job. Use a
CUDA-capable PyTorch job image and a Docker host configured for NVIDIA GPUs:

```bash
nvflare job submit \
  -j jobs/hello-pt-docker \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

This job keeps the original CIFAR-10 cache path, `/var/tmp/nvflare/data`, unless
`NVFL_CIFAR10_ROOT` selects another directory. For Docker clients, set that
variable in `job_launcher.default_job_env` in `docker.yaml` before preparing
their kits; exporting it on the host affects the process client only.
The job uses torchvision's standard
[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) download URL
and checksum validation.
Use the returned job ID to check completion:

```bash
nvflare job wait JOB_ID \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

The CPU training job can take several minutes.

Available jobs:

| Job | Description |
|-----|-------------|
| `hello-numpy-docker` | Basic numpy federated averaging |
| `hello-pt-docker` | PyTorch CIFAR-10 training; requests one NVIDIA GPU by default |
| `pt-ddp-docker` | Multi-GPU DDP training with torchrun |

## Step 6: Stop the example

After the jobs finish, shut down this federation:

```bash
nvflare system shutdown all --force --timeout 60 \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

On macOS, you can also run `colima stop` when you are finished using its containers.

## Notes

- Docker launcher settings are specified per-site in `launcher_spec` in
  `meta.json`. Keep resource requests such as `num_of_gpus` in `resource_spec`,
  the same way as process-mode jobs. Example:
  ```json
  "launcher_spec": {
    "site-1": {"docker": {"image": "nvflare-job:latest", "shm_size": "8g"}}
  },
  "resource_spec": {
    "site-1": {"num_of_gpus": 1}
  }
  ```
  Sites without a `docker` entry (e.g. `site-2` in these examples) run in process mode. Both
  modes can coexist in the same job.
- Site-level Docker defaults (e.g. `shm_size`, `ipc_mode`) can be set via
  `default_job_container_kwargs` in `resources.json` — job-level
  `launcher_spec[site][docker]` takes precedence on conflict.
- Some multi-GPU Docker environments may need `NCCL_P2P_DISABLE=1` to avoid NCCL hangs.
  Set this site-wide with `default_job_env` in `resources.json`, for example:
  ```json
  "default_job_env": {"NCCL_P2P_DISABLE": "1"}
  ```
- Workspace files are bind-mounted at `/var/tmp/nvflare/workspace` inside all containers.
- Job containers run as the same UID/GID as the SP/CP so all workspace files remain
  readable and writable by the parent process.
- To watch job container logs: `docker logs -f <site>-<job_id>`
