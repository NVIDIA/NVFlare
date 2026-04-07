# Docker Job Launcher — Open Discussion Points

## 1. `listening_host.port` — new provisioning concept

The Docker launcher requires a dedicated TCP port for job containers (SJ/CJ) to connect back to the SP/CP container (`PARENT_URL`). This is a new field in `project.yml` that process-mode users have never needed.

- Must differ from `fed_learn_port`
- Currently only documented in the design doc — should it also appear in the provisioning guide?
- Should provisioning validate that `listening_host.port != fed_learn_port` and give a clear error?

## 2. `start_docker.sh` is signed — users cannot edit it

`start_docker.sh` is in the `startup/` folder which is signed by `SignatureBuilder`. Users cannot modify it after provisioning without breaking the signature check.

The image name is baked in at provision time. The escape hatch is `NVFL_P_IMAGE` env var override, but this is not obvious.

- Is `NVFL_P_IMAGE` prominent enough in the UX?
- Should we document this more prominently in the startup kit README or as a comment inside `start_docker.sh`?

## 3. No orphan container recovery on SP/CP restart

If SP/CP crashes or is restarted while a job is running, the SJ/CJ containers from the previous session are left running with no parent to report to. They will eventually time out, but NVFlare has no mechanism to detect or clean them up on restart.

- Is this acceptable for the initial release?
- Should SP/CP scan for and terminate orphaned containers on startup?
- Currently listed as out-of-scope in the design doc.

## 4. Single-machine testing requires `/etc/hosts` edit

When testing server and client on the same machine (e.g. local dev), the client needs to resolve the server's hostname (e.g. `server`) to `127.0.0.1`. This requires:

```bash
echo "127.0.0.1 server" | sudo tee -a /etc/hosts
```

This is not great UX for local testing.

- Should the design doc suggest using `localhost` or `127.0.0.1` as the server `name` for single-machine testing?
- Or should `start_docker.sh` add a `--add-host` flag automatically for local testing?

## 5. `admin_port` not published in `start_docker.sh`

`start_docker.sh` currently only publishes `fed_learn_port` and `listening_host.port` to the host. If `admin_port` differs from `fed_learn_port` (which is valid config — `admin_port` defaults to `fed_learn_port` but can be set independently), the admin CLI cannot reach the server's admin service from outside the container.

- Option A: Always publish `admin_port` separately in the template (add `-p {admin_port}:{admin_port}`) and pass it through `DockerLauncherBuilder`
- Option B: Constrain `admin_port == fed_learn_port` for Docker mode and document this limitation
- Option C: Document that if `admin_port != fed_learn_port`, users must add the port mapping manually (but `start_docker.sh` is signed, so they can't edit it)

## 6. Container runs as root — host workspace file ownership

`start_docker.sh` runs the SP/CP container as the image's default user (typically root). Since the host workspace is bind-mounted, any files written by the container (run artifacts, logs, etc.) will be owned by root on the host, making them inaccessible to the site admin without `sudo`.

- Option A: Run with host UID/GID using `-u $(id -u):$(id -g)` and mount `/etc/passwd` and `/etc/group` so the user exists inside the container
- Option B: Add a `chown` step in `sub_start.sh` or a post-job hook
- Option C: Document the issue and leave it to the site admin to handle (e.g. `chmod`/`chown` after job completion)

## 7. No Python API for specifying job image

Currently the job image is specified in `meta.json` under `deploy_map`:

```json
{"sites": ["@ALL"], "image": "nvflare-job:latest"}
```

There is no Python API (e.g. `FedJob`) support for setting the image. Users building jobs programmatically must manually edit `meta.json`.

- Should `FedJob` (or job config API) support `job.set_image("nvflare-job:latest")` or per-site `job.set_image("nvflare-job:latest", sites=["site-1"])`?
- This would be a separate PR but worth noting as a follow-up.

## 8. No way to pass extra docker flags to SP/CP container

`start_docker.sh` is signed and cannot be edited by the site admin. There is currently no mechanism to pass extra `docker run` flags (e.g. `--gpus all`, `--shm-size 8g`, `--ipc host`) to the SP/CP container.

For SJ/CJ job containers this is handled via `extra_container_kwargs` in `resources.json`. For SP/CP there is no equivalent.

- Option A: Support a `NVFL_EXTRA_DOCKER_ARGS` env var in `start_docker.sh`, evaluated at runtime and injected into the `docker run` command. Not blocked by signing since it's a runtime value.
- Option B: Add an `extra_docker_args` field to `DockerLauncherBuilder` in `project.yml`, baked in at provision time.
- Option C: Document that SP/CP extra flags require re-provisioning with a custom template.

Note: GPU for SJ/CJ job containers is already handled via `resource_spec` in `meta.json` (`num_of_gpus`), which maps to `device_requests` in `docker run`. Only SP/CP GPU access is unaddressed.
