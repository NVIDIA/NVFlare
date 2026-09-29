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
The `hello-numpy-docker` and `hello-pt-docker` jobs run on CPU on a Mac;
`pt-ddp-docker` requires multiple NVIDIA GPUs and is not a Mac example.

```bash
brew install colima docker python@3.13
colima start --cpu 4 --memory 8 --disk 40 --runtime docker \
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

## Step 3: Add /etc/hosts entries (if needed)

Skip this step if `server` already resolves via your DNS. Otherwise, add the following
to `/etc/hosts` so the admin CLI can reach the server container by name:

```
127.0.0.1  server
```

On a Mac where `server` already names another host, leave `/etc/hosts` alone.
Point only the host-side site-2 and admin kits at the published loopback port.
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

Start all three parent processes from the `examples/docker` directory:

```bash
(
  cd workspace/docker_test_project/prepared/server
  nohup bash startup/start_docker.sh > server.log 2>&1 < /dev/null &
)
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

## Step 5: Submit a job

```bash
nvflare job submit \
  -j jobs/hello-numpy-docker \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

On a Mac, the PyTorch example also runs without a GPU:

```bash
nvflare job submit \
  -j jobs/hello-pt-docker \
  --startup-kit workspace/docker_test_project/prod_00/admin@nvidia.com
```

This job downloads CIFAR-10 into a writable `data` directory at each site's
working directory. It uses a [faster mirror](https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz)
of the [CIFAR-10 Python archive](https://zenodo.org/records/10089977);
torchvision checks the archive against the original MD5 before extraction.
To use another source, set `NVFL_CIFAR10_URL` in `job_launcher.default_job_env`
in `docker.yaml` before preparing site-1, and export it before starting site-2.
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
| `hello-pt-docker` | PyTorch CIFAR-10 federated training |
| `pt-ddp-docker` | Multi-GPU DDP training with torchrun |

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
