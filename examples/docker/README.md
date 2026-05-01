# Docker Job Launcher Example

End-to-end example of running NVFlare in Docker mode using `nvflare deploy prepare`.
SP/CP containers are started manually; SJ/CJ containers are launched automatically per job.

## Prerequisites

- Docker with a working daemon
- NVFlare installed (development install from repo root: `pip install -e .[dev,PT]`)
- Run all commands from the **repo root** unless noted otherwise

## Step 0: Build Docker images

```bash
bash examples/docker/build_docker.sh
```

This builds two images:
- `nvflare-site:latest` — used by SP/CP containers (started by `start_docker.sh`)
- `nvflare-job:latest` — used by SJ/CJ containers (launched automatically per job)

## Step 1: Provision

```bash
nvflare provision -p examples/docker/project.yml
```

This generates a workspace under `workspace/docker_test_project/` relative to the current directory.

## Step 2: Prepare Docker runtime kits

Create a Docker runtime config:

```bash
cat > /tmp/nvflare-docker.yaml <<'EOF'
runtime: docker
parent:
  docker_image: nvflare-site:latest
  network: nvflare-network
job_launcher:
  default_python_path: /usr/local/bin/python
EOF
```

Prepare the server and one Docker-mode client:

```bash
nvflare deploy prepare workspace/docker_test_project/prod_00/server \
  --output workspace/docker_test_project/docker/server \
  --config /tmp/nvflare-docker.yaml

nvflare deploy prepare workspace/docker_test_project/prod_00/site-1 \
  --output workspace/docker_test_project/docker/site-1 \
  --config /tmp/nvflare-docker.yaml
```

`start_docker.sh` creates the Docker network if needed.

## Step 3: Add /etc/hosts entries (if needed)

Skip this step if `server` already resolves via your DNS. Otherwise, add the following
to `/etc/hosts` so the admin CLI can reach the server container by name:

```
127.0.0.1  server
```

## Step 4: Start server and clients

This example runs in **hybrid mode**: site-1 uses the prepared Docker runtime kit,
site-2 runs in process mode (`start.sh`). This tests that both modes work together in the
same federation.

Run each in a separate terminal:

```bash
# Server (Docker mode)
cd workspace/docker_test_project/docker/server
bash startup/start_docker.sh

# site-1 (Docker mode — job containers launched per job)
cd workspace/docker_test_project/docker/site-1
bash startup/start_docker.sh

# site-2 (process mode — jobs run as subprocesses of CP)
cd workspace/docker_test_project/prod_00/site-2
bash startup/start.sh
```

## Step 5: Submit a job

```bash
nvflare job submit \
  -j examples/docker/jobs/hello-numpy-docker \
  -w workspace/docker_test_project/prod_00/admin@nvidia.com
```

Available jobs:

| Job | Description |
|-----|-------------|
| `hello-numpy-docker` | Basic numpy federated averaging |
| `hello-pt-docker` | PyTorch CIFAR-10 federated training |
| `pt-ddp-docker` | Multi-GPU DDP training with torchrun |

## Notes

- Job images and Docker resource requirements are specified per-site in `meta.json` under
  `resource_spec[site][docker]`. Example:
  ```json
  "resource_spec": {
    "site-1": {"docker": {"image": "nvflare-job:latest", "num_of_gpus": 1, "shm_size": "8g"}}
  }
  ```
  Sites without a `docker` entry (e.g. `site-2` in these examples) run in process mode. Both
  modes can coexist in the same job.
- Site-level Docker defaults (e.g. `shm_size`, `ipc_mode`) can be set via
  `default_job_container_kwargs` in `resources.json` — job-level `resource_spec` takes
  precedence on conflict.
- Some multi-GPU Docker environments may need `NCCL_P2P_DISABLE=1` to avoid NCCL hangs.
  Set this site-wide with `default_job_env` in `resources.json`, for example:
  ```json
  "default_job_env": {"NCCL_P2P_DISABLE": "1"}
  ```
- Workspace files are bind-mounted at `/var/tmp/nvflare/workspace` inside all containers.
- Job containers run as the same UID/GID as the SP/CP so all workspace files remain
  readable and writable by the parent process.
- To watch job container logs: `docker logs -f <site>-<job_id>`
