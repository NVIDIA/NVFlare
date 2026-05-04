# Site Deployment Preparation Design

## Goal

Allow each site to prepare an already provisioned startup kit for a deployment
runtime such as Docker or Kubernetes.

This design separates two concerns:

- creating a valid startup kit with identity, certificates, topology, and base
  FLARE configuration
- preparing that startup kit for the site's selected runtime/deployment target

The proposed command is:

```bash
nvflare deploy prepare ./site-1
nvflare deploy prepare ./site-1 --output ./site-1-docker --config docker.yaml
nvflare deploy prepare ./site-1 --output ./site-1-k8s --config k8s.yaml
```

`prepare` is intentionally scoped under `deploy`. A plain `nvflare prepare`
would be too broad, and `deploy prepare` leaves room for future deployment
subcommands such as `validate` or runtime-specific helpers.

## Background

Today there are two ways for a site to get a production-style startup kit.

### Centralized Provisioning

The project admin runs:

```bash
nvflare provision -p project.yml
```

This remains the full centralized provisioning path. It can use existing
provisioning builders and continues to support deployment types that require
centralized setup, including HE and CC.

### Distributed Provisioning

The site and project admins use the manual distributed provisioning flow:

```bash
nvflare cert ...
nvflare package ...
```

In this flow each site owns its private key, receives a signed certificate and
`rootCA.pem`, and assembles its local startup kit. This flow is intended for
standard mTLS deployments and currently does not support HE or CC.

Both flows produce the same conceptual input for this design: an existing valid
startup kit.

## Non-Goals

- Do not replace `nvflare provision`.
- Do not replace `nvflare package`.
- Do not create participant identity, certificates, or topology.
- Do not manage private key, certificate, Kubernetes Secret, or secret
  distribution workflows.
- Do not require a site-local `project.yml` just to choose Docker or K8s.
- Do not use deployment preparation as a rollback mechanism to restore process
  mode. Process-mode kits are the default output of `nvflare provision` and
  `nvflare package`.
- Do not add HE or CC support to distributed provisioning in this work.
- Do not create, start, or manage a Kubernetes cluster.
- Do not make `nvflare deploy prepare` submit jobs. Job submission remains the
  role of the admin client, recipe `ProdEnv`, or related APIs.

## Design Rationale: Separate from Provisioning

The earlier approach reused:

```bash
nvflare package --project-file site-project.yml ...
```

with a site-local full project YAML containing one participant and optional
runtime builders.

That approach worked mechanically, but it made runtime preparation depend on
provisioning concepts:

- the site had to write a project-style YAML even though the startup kit already
  existed
- runtime choice was expressed through `Builder` classes
- `nvflare package` had to carry deployment behavior
- centrally provisioned kits needed cert/key material staged back into package
  input form
- output reporting could still point to process-mode scripts such as `start.sh`
- there was no natural validation that the local project YAML matched the
  actual provisioned identity/topology

The cleaner model is to operate directly on the existing startup kit.

## Identity Material Boundary

`nvflare deploy prepare` does not manage private keys, certificates,
Kubernetes Secrets, or secret distribution.

The input startup kit already contains the identity material required to run
the site. Runtime preparation only generates scripts/artifacts and patches
runtime configuration.

For Docker, generated scripts mount the existing startup kit/workspace into the
parent container, for example at `/var/tmp/nvflare/workspace`. The container
uses the mounted kit at runtime. Private keys and certificates must not be
baked into the Docker image.

For K8s parent pods, generated charts assume the startup kit content is already
available at the configured workspace mount path, such as through a PVC prepared
by the site admin. The site admin is responsible for copying startup kit
material into that storage using their organization's approved process.
`nvflare deploy prepare` does not create Kubernetes Secret objects or upload
certificate material to the cluster.

K8s job pods have a different runtime assumption. They are launched
dynamically by `K8sJobLauncher` after the parent process is already running.
The launcher may create or update Kubernetes Secret objects containing the
startup files required by an individual job pod. This is job-launch runtime
behavior, not deployment preparation behavior. If this launcher path is used,
the parent pod's Kubernetes service account must have the permissions required
to manage those runtime job-pod Secrets in the configured namespace.

K8s image registry credentials follow the same boundary. `nvflare deploy
prepare` should not create registry credentials, manage private registry login,
or create image pull Secrets. If private registry access is needed, the site or
cluster should already be configured through its normal Kubernetes/registry
process.

## Proposed Command

```bash
nvflare deploy prepare <startup-kit-dir>
nvflare deploy prepare <startup-kit-dir> \
  --output <prepared-kit-dir> \
  --config <runtime-config.yaml>
```

The runtime is declared in the config file with `runtime: docker` or
`runtime: k8s`.

By convention, `--config` defaults to `<startup-kit-dir>/config.yaml`, and
`--output` defaults to `<startup-kit-dir>/prepared/<runtime>`, such as
`prepared/docker` or `prepared/k8s`. Users can override either path when
needed.

Possible future runtime names can be added under the same command without
changing provisioning or packaging.

The command reads the input startup kit, writes a prepared copy to the output
directory, and applies runtime-specific generated artifacts and config changes
to the output copy. Existing input kit files should remain unchanged.

## CLI Arguments and Config Validation

`nvflare deploy prepare` should validate both CLI arguments and runtime config
files before writing generated artifacts.

Supported arguments:

- `<startup-kit-dir>`
  - Required: yes
  - Description: path to an existing input startup kit directory.

- `--kit`
  - Required: no
  - Description: explicit alias for `<startup-kit-dir>`, kept for callers that
    prefer named options.

- `--output`
  - Required: no
  - Default: `<startup-kit-dir>/prepared/<runtime>`
  - Description: path to write the prepared startup kit directory. The runtime
    preparation changes are applied here, not to the input kit files.

- `--config`
  - Required: no
  - Default: `<startup-kit-dir>/config.yaml`
  - Description: YAML config file. The file must contain a top-level `runtime`
    key. Initial values are `docker` and `k8s`.

Invalid runtime names, malformed YAML, missing `runtime`, unknown required
types, and unsupported kit/runtime combinations should fail before writing the
prepared output.

### Docker Config

Example `docker.yaml`:

```yaml
runtime: docker

parent:
  docker_image: nvflare-site:latest
  network: nvflare-network

job_launcher:
  default_python_path: /usr/local/bin/python
  default_job_env:
    NCCL_P2P_DISABLE: "1"
  default_job_container_kwargs:
    shm_size: 8g
    ipc_mode: host
```

Docker preparation has two separate outputs:

- `startup/start_docker.sh` for the parent server/client container.
- `local/resources.json.default` launcher configuration for dynamically
  launched job containers.

The `docker.yaml` file should keep those two concerns separate.

Supported top-level keys:

- `runtime`
  - Required: yes
  - Default: none
  - Description: must be `docker`.

Supported `parent` keys:

- `docker_image`
  - Required: yes
  - Default: none
  - Description: image used by the parent server/client container started by
    `start_docker.sh`.

- `network`
  - Required: no
  - Default: `nvflare-network`
  - Description: Docker network shared by parent and job containers.

Docker does not expose a separate `parent_port` in `docker.yaml` for the first
version. The parent process keeps the ports already present in the startup kit,
and `start_docker.sh` is responsible for running the parent container on the
configured Docker network. For server kits, externally visible FL ports are
already known from the kit and can be published by the generated script.

Supported `job_launcher` keys:

- `default_python_path`
  - Required: no
  - Default: `/usr/local/bin/python`
  - Description: Default Python executable used in job containers. Individual
    jobs can override it with `launcher_spec[<site>][docker].python_path`.

- `default_job_env`
  - Required: no
  - Default: `{}`
  - Description: environment variables injected into every Docker job
    container.

- `default_job_container_kwargs`
  - Required: no
  - Default: `{}`
  - Description: Docker SDK kwargs applied to every job container. Reserved
    launcher-controlled keys are rejected.

### K8s Config

K8s preparation has two separate outputs:

- Helm chart values/templates for the parent server/client pod.
- `local/resources.json.default` launcher configuration for dynamically
  launched job pods.

The `k8s.yaml` file should keep those two concerns separate.

Example shape:

```yaml
runtime: k8s
namespace: default

parent:
  docker_image: nvflare-site:latest
  parent_port: 8102
  workspace_pvc: nvflws
  workspace_mount_path: /var/tmp/nvflare/workspace
  resources:
    requests:
      cpu: "2"
      memory: 8Gi
  pod_security_context: {}

job_launcher:
  config_file_path: null
  pending_timeout: null
  default_python_path: /usr/local/bin/python
  job_pod_security_context: {}
```

Supported top-level keys:

- `runtime`
  - Required: yes
  - Default: none
  - Description: must be `k8s`.

- `namespace`
  - Required: no
  - Default: `default`
  - Description: Kubernetes namespace used by both the generated parent
    resources and dynamically launched job pods.

Supported `parent` keys:

- `docker_image`
  - Required: yes
  - Default: none
  - Description: image used by the parent server/client pod in the generated
    Helm chart.

- `parent_port`
  - Required: no
  - Default: `8102`
  - Description: port used by job pods to communicate with the parent process.

- `workspace_pvc`
  - Required: no
  - Default: `nvflws`
  - Description: PVC claim name containing the runtime workspace/startup kit
    content. This raw value is rendered only as `claimName`; the pod volume
    name is a fixed Kubernetes DNS label.

- `workspace_mount_path`
  - Required: no
  - Default: `/var/tmp/nvflare/workspace`
  - Description: in-container mount path for the workspace PVC.

- `resources`
  - Required: no
  - Default: chart defaults
  - Description: parent pod resource requests and limits rendered into the Helm
    chart.

- `pod_security_context`
  - Required: no
  - Default: `{}`
  - Description: parent pod security context rendered into the Helm chart.

Supported `job_launcher` keys:

- `config_file_path`
  - Required: no
  - Default: `null`
  - Description: Kube config path used by `K8sJobLauncher`. If omitted, the
    launcher uses in-cluster config.

- `pending_timeout`
  - Required: no
  - Default: launcher default
  - Description: timeout for pending job pods.

- `default_python_path`
  - Required: no
  - Default: `/usr/local/bin/python`
  - Description: Default Python executable used in job pods. Individual jobs
    can override it with `launcher_spec[<site>][k8s].python_path`.

- `job_pod_security_context`
  - Required: no
  - Default: `{}`
  - Description: job pod security context passed to `K8sJobLauncher`.


## Docker Runtime Preparation

Input is an existing server or client startup kit plus a `docker.yaml` config
file matching the Docker config schema above.

Expected behavior:

- generate `startup/start_docker.sh`
- update `local/resources.json.default`
- replace `process_launcher` with `docker_launcher`
- render parent container settings from `docker.yaml` into `start_docker.sh`
- configure `DockerJobLauncher` with `job_launcher` values from `docker.yaml`
- patch runtime communication settings so job containers can reach the parent
  container on the Docker network
- use `local/study_data.yaml` to resolve study data host paths
- switch `resource_manager` to `PassthroughResourceManager`, which replaces
  CPU/GPU resource scheduling with a no-op resource manager for Docker/K8s
  launcher-based execution
- remove `resource_consumer`
- report the correct next step, such as `./startup/start_docker.sh`

The first version can focus on the behavior above. Additional Docker-specific
settings can be added later without changing the command boundary.

## K8s Runtime Preparation

Input is an existing server or client startup kit plus a `k8s.yaml` config file
matching the K8s config schema above.

Expected behavior:

- generate a Helm chart
- update `local/resources.json.default`
- replace `process_launcher` with `k8s_launcher`
- render parent pod settings from `k8s.yaml` into the generated Helm chart
- configure `K8sJobLauncher` with `job_launcher` values from `k8s.yaml`
- patch runtime communication settings so job pods can reach the parent pod
  through the generated Kubernetes Service
- use `local/study_data.yaml` to resolve study data PVC names
- switch `resource_manager` to `PassthroughResourceManager`, which replaces
  CPU/GPU resource scheduling with a no-op resource manager for Docker/K8s
  launcher-based execution
- remove `resource_consumer`
- report the correct next step, such as a Helm install/upgrade command

Existing Helm chart generation code can still be reused, but it should be
invoked from deployment preparation logic rather than requiring a provisioning
builder in a site-local project YAML.

Unlike Docker preparation, K8s preparation does not need to generate a
`startup/start_k8s.sh` script in the first version. NVFlare does not start or
manage the Kubernetes cluster. The generated Helm chart is the deployment
artifact, and the site admin applies it to an existing cluster with standard
K8s/Helm tooling such as `helm install` or `helm upgrade`.

## Runtime Communication Patching

`deploy prepare` must update runtime-managed communication settings so
dynamically launched job containers or pods can reach the parent server/client
process for the selected runtime. Runtime preparation is not only script/chart
generation and launcher replacement.

For Docker:

- `start_docker.sh` creates or uses the configured Docker network.
- the parent container and job containers use the same Docker network name.
- server parent containers are reachable through the logical Docker DNS alias
  `server`, even when the concrete container name differs.
- `local/comm_config.json` is created if missing, then patched for the runtime
  parent-process endpoint.
- the parent process binds its internal listener to `0.0.0.0` so job
  containers can connect over the Docker network.
- `DockerJobLauncher` uses the same network and parent naming assumptions as
  `start_docker.sh`.

For K8s:

- the generated Helm chart creates a Kubernetes Service for the parent pod.
- dynamically launched job pods connect back to the parent through that
  Service.
- the Service name, parent port, launcher config, and internal communication
  config in the prepared kit must be consistent.
- `local/comm_config.json` is created if missing, then patched for the runtime
  parent-process endpoint.
- the internal pod-to-parent communication settings should use the Kubernetes
  Service host, configured parent port, and the scheme/security expected by the
  runtime.

These behaviors already exist in the current Docker launcher and Helm chart
builder paths. Moving runtime preparation out of provisioning means the new
deploy command must preserve or reuse those communication patching behaviors.

## Study Data Mapping

`local/study_data.yaml` is the site-local mapping from a study name to the
storage location that contains that study's data. Docker uses host paths from
this file, while K8s uses PVC names from this file. `deploy prepare` should
create a commented template when the file is absent in the prepared output, and
should preserve an existing file when the input kit already contains one.

This replaces the separate per-runtime files used previously:
`local/study_data.json` for Docker and `local/study_data_pvc.yaml` for K8s.
Both launchers and `workspace_cell_transfer.py` must be updated to use
`local/study_data.yaml`.

The detailed study data schema follows [Study Dataset Mapping](https://docs.google.com/document/d/1JCKHbjQaDto_SBuTB-wSAaEvfIO8NLA_T2I_u8rf2sQ/edit?tab=t.0#heading=h.s3ol8f3q17a5).

## Job Python, GPU, and Resource Handling

Job-image-specific settings for Docker and K8s job containers should continue
to use job launcher metadata instead of deployment-prepare-only configuration
fields.

For Docker job containers, `DockerJobLauncher` uses job metadata for settings
that belong to the selected job image:

- `launcher_spec[<site>][docker].num_of_gpus`
- `launcher_spec[<site>][docker].python_path`
- legacy `resource_spec[<site>][docker].num_of_gpus`
- legacy flat `resource_spec[<site>].num_of_gpus`
- explicit Docker SDK `device_requests` in the Docker launcher spec

`DockerJobLauncher` translates `num_of_gpus` into Docker SDK
`device_requests`, and job-level settings take precedence over site-level
defaults. Site-local defaults can still be supplied through
`default_job_container_kwargs` in `docker.yaml` when needed.
`launcher_spec[<site>][docker].python_path` overrides
`job_launcher.default_python_path` for jobs whose image uses a different Python
location.

For K8s job pods, `K8sJobLauncher` already supports GPU and basic resource
limits through job metadata, and it uses the same metadata path for
image-specific Python overrides:

- `launcher_spec[<site>][k8s].num_of_gpus`
- `launcher_spec[<site>][k8s].python_path`
- legacy `resource_spec[<site>][k8s].num_of_gpus`
- `launcher_spec[<site>][k8s].cpu`
- `launcher_spec[<site>][k8s].memory`

`K8sJobLauncher` maps `num_of_gpus` to the pod container limit
`nvidia.com/gpu`.
`launcher_spec[<site>][k8s].python_path` overrides
`job_launcher.default_python_path` for jobs whose image uses a different Python
location.

`nvflare deploy prepare` does not need to invent a separate top-level GPU
setting for job containers. If the parent server/client container or pod itself
needs GPU access in the future, that can be added as a deployment runtime
setting distinct from per-job GPU allocation.

## Kit Validation

`nvflare deploy prepare` should validate the input kit before writing the
prepared output.

Minimum validation:

- `<startup-kit-dir>` exists and is a directory
- `--output` can be created or updated, using
  `<startup-kit-dir>/prepared/<runtime>` when omitted
- `startup/`, `local/`, and `local/resources.json.default` exist
- `local/resources.json.default` is valid JSON
- the kit role is detected from files in `startup/`. A server kit has
  `startup/fed_server.json`, a client kit has `startup/fed_client.json`, and an
  admin kit has `startup/fed_admin.json`.
- exactly one kit role is detected. If none or multiple role config files are
  present, validation fails.
- the configured runtime supports that kit type

For the first version, only server and client kits are supported. If an admin
kit is detected, the command should exit with an error instead of silently
skipping or partially preparing the kit.

## Output Behavior

The startup kit is read-only input. Runtime preparation changes are written to
the prepared kit under `--output`, or under
`<startup-kit-dir>/prepared/<runtime>` when `--output` is omitted.

The command should print a result that reflects the configured runtime. For
Docker, a successful result should point users to:

```bash
cd <prepared-kit-dir>
./startup/start_docker.sh
```

For K8s, a successful result should point users to the generated Helm chart and
show the expected command shape, for example:

```bash
helm upgrade --install <release-name> <prepared-kit-dir>/helm_chart \
  --namespace <namespace>
```

Repeated runs should converge `--output` to the config-selected runtime. The
command overwrites runtime-managed files and sections without requiring
`--force`, including generated Docker scripts, generated K8s Helm chart
content, and launcher/resource components in `local/resources.json.default`.

For Docker and K8s runtimes, deployment preparation always replaces
process-mode resource scheduling with `PassthroughResourceManager`, a no-op
resource manager used for launcher-based execution, and removes
`resource_consumer`. This is runtime policy, not a user-facing config option.

## Implementation Direction

The runtime mutation logic should live outside the provisioning `Builder`
contract. A possible internal structure:

- `nvflare.tool.deploy.deploy_cli`: CLI parsing and command registration.
- `nvflare.tool.deploy.deploy_commands`: command orchestration, validation,
  output copy handling, and dispatch to the selected runtime preparer.
- `nvflare.tool.deploy.runtime.docker`: Docker-specific script generation and
  resource/config mutation.
- `nvflare.tool.deploy.runtime.k8s`: K8s-specific Helm chart generation and
  resource/config mutation.

Shared helpers can handle:

- loading and writing `resources.json.default`
- replacing job launcher components
- replacing resource manager components
- removing `resource_consumer`
- detecting server/client/admin kit type
- reporting next steps

Runtime preparation should not depend on provisioning `Builder` classes or
`lighter` templates. Docker/K8s deployment preparation should use the deploy
command rather than requiring a site-local `project.yml` builder entry.

## Workflow Summary

### Centralized Provisioning Plus Deployment Preparation

```bash
# Project admin
nvflare provision -p project.yml

# Site admin, from the received startup kit
nvflare deploy prepare ./site-1
cd ./site-1/prepared/docker
./startup/start_docker.sh
```

### Distributed Provisioning Plus Deployment Preparation

```bash
# Site/project admins complete cert + package flow
nvflare cert ...
nvflare package ...

# Site admin
nvflare deploy prepare ./site-1
cd ./site-1/prepared/docker
./startup/start_docker.sh
```

The deployment preparation step is identical regardless of how the startup kit
was created.
