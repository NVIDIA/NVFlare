# Docker Job Launcher Example

End-to-end example of running NVFlare in Docker mode using `nvflare deploy prepare`.
SP/CP containers are started manually; SJ/CJ containers are launched automatically per job.

## Prerequisites

- Docker with a working daemon
- NVFlare installed (development install from repo root: `pip install -e .[dev,PT]`)
- Run all commands from the `examples/docker` directory unless noted otherwise

## Step 0: Build Docker images

```bash
bash build_docker.sh
```

This builds two images:
- `nvflare-site:latest` — used by SP/CP containers (started by `start_docker.sh`)
- `nvflare-job:latest` — used by SJ/CJ containers (launched automatically per job)

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
  --config examples/docker/docker.yaml \
  --output workspace/docker_test_project/prepared/server

nvflare deploy prepare \
  workspace/docker_test_project/prod_00/site-1 \
  --config examples/docker/docker.yaml \
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
