# Site Runtime Packaging Design

## Goal

Keep centralized provisioning unchanged for normal process-based deployments, while letting
each site choose its own runtime packaging during the local `nvflare package` step.

The site-local runtime choices are:

- process
- Docker
- K8s

## Chosen Approach

Reuse the existing `nvflare package --project-file ...` path.

- Central admin uses one central `project.yml` for `nvflare provision`.
- Each site uses one local full project YAML for `nvflare package`.
- The site-local YAML contains one participant and, optionally, one runtime builder.

This keeps the design builder-driven and avoids adding a new package config concept.

## Runtime Mapping

- process: no runtime builder
- Docker: `nvflare.lighter.impl.docker_launcher.DockerLauncherBuilder`
- K8s: `nvflare.lighter.impl.k8s_runtime.K8sRuntimeBuilder`

Runtime builders are for `server` and `client` kits only. Admin kits should remain
process-based.

## Why This Fits Current Code

`nvflare package --project-file ...` already supports full project YAML and already reuses
`builders:` from that YAML. Package also already swaps `CertBuilder` for
`PrebuiltCertBuilder` and injects `WorkspaceBuilder` and `StaticFileBuilder` when needed.

So the package CLI does not need a new runtime-specific option. The main work is in the
runtime builders themselves.

## Builder Responsibilities

### DockerLauncherBuilder

- generate `startup/start_docker.sh`
- replace `process_launcher` with `docker_launcher`
- allow site-local Docker settings such as `docker_image`, `network`,
  `default_job_env`, and `default_job_container_kwargs`
- replace `GPUResourceManager` with `PassthroughResourceManager`
- remove `resource_consumer`

### K8sRuntimeBuilder

- reuse `HelmChartBuilder` to generate the Helm chart
- replace `process_launcher` with `k8s_launcher`
- write `local/study_data_pvc.yaml`
- replace `GPUResourceManager` with `PassthroughResourceManager`
- remove `resource_consumer`

`HelmChartBuilder` alone is not enough because chart generation does not make the startup
kit runtime-consistent for K8s.

## Site YAML Examples

### Process

```yaml
api_version: 3
name: my_project

participants:
  - name: site-1
    type: client
    org: hospital
```

### Docker

```yaml
api_version: 3
name: my_project

participants:
  - name: site-1
    type: client
    org: hospital

builders:
  - path: nvflare.lighter.impl.docker_launcher.DockerLauncherBuilder
    args:
      docker_image: nvflare-site:latest
      network: nvflare-network
      default_job_env:
        NCCL_P2P_DISABLE: "1"
      default_job_container_kwargs:
        shm_size: 8g
        ipc_mode: host
```

### K8s

```yaml
api_version: 3
name: my_project

participants:
  - name: site-1
    type: client
    org: hospital

builders:
  - path: nvflare.lighter.impl.k8s_runtime.K8sRuntimeBuilder
    args:
      docker_image: nvflare-site:latest
      namespace: default
      parent_port: 8102
      workspace_pvc: nvflws
      workspace_mount_path: /var/tmp/nvflare/workspace
      study_data_pvc:
        default: nvfldata
```

## Workflow

### Centralized Provisioning

```bash
nvflare provision -p project.yml
```

After centralized provisioning, a site that wants Docker or K8s local packaging must first
stage a cert-material directory for `nvflare package`. Today, YAML mode expects:

- `<participant>.crt`
- `<participant>.key`
- `rootCA.pem`

So if the site starts from an existing startup kit containing files such as
`startup/client.crt`, `startup/client.key`, and `startup/rootCA.pem`, there is currently an
extra local step to copy or rename them into a raw cert directory such as:

- `site-1.crt`
- `site-1.key`
- `rootCA.pem`

The site then packages locally:

```bash
nvflare package \
  --project-file ./site-1-project.yml \
  --dir ./certs \
  --endpoint grpc://server:8002
```

### Distributed Manual Provisioning

```bash
nvflare cert csr ...
nvflare cert sign ...
nvflare package \
  --project-file ./site-1-project.yml \
  --dir ./certs \
  --endpoint grpc://server:8002
```

In both cases, the central side provides signed identity material and each site selects its
runtime by choosing its local builder.

## Remaining Workflow Gaps

- In YAML mode, `--dir` currently expects raw cert material named
  `<participant>.crt`, `<participant>.key`, and `rootCA.pem`.
- The default output location can look odd when package is run from inside an existing
  site folder.
- Package still reports a generic next step such as `start.sh`, even when Docker or K8s
  packaging changed the real runtime entrypoint.
- The local site project YAML is not currently validated against the centrally provisioned
  project. In practice, the site YAML should only change runtime packaging behavior, not
  participant identity or topology, but the current package flow does not enforce that.
