# OpenShift Deployment Helpers

This directory contains the OpenShift-specific NVFlare deployment guide and helper scripts.

- [index.md](index.md) is the detailed OpenShift deployment guide.
- Repository `docker/Dockerfile.parent` builds the parent image used by server/client and admin pods.
- Repository `docker/Dockerfile.job` builds the workload image used by job pods.
- `scripts/create_openshift_cluster.sh` configures Red Hat OpenShift Local (CRC) and optionally starts it.
- `scripts/start_openshift_cluster.sh` starts CRC, logs in with `oc`, and prepares the target project.
- `scripts/cleanup_openshift_cluster.sh` deletes scripted deployment resources and stops CRC.
- `scripts/k8s_provision.sh` runs `nvflare provision` for the sample server, `site-1`, `site-2`, and admin.
- `scripts/k8s_deploy.sh` prepares K8s startup kits, stages `startup/` and
  `local/` into PVC workspaces, installs Helm charts, and verifies parent pods
  can import the Kubernetes Python client. This is the manual PVC-copy staging
  path; `nvflare deploy k8s stage` can be used instead to stage `local/` as a
  ConfigMap and `startup/` as a Secret before running Helm.
- `scripts/k8s_submit_job.sh` submits `hello-numpy` from an in-cluster admin pod and waits for successful completion.
- `scripts/k8s_watch.sh` shows an in-place live Rich pod table for the created pods.
- `scripts/k8s_watch.py` implements the Rich table used by the shell wrapper.
- `scripts/k8s_e2e.sh` runs provision, deploy, and submit in order. Set
  `WORKSPACE_STAGING_MODE=pvc` for the manual PVC-copy path, or
  `WORKSPACE_STAGING_MODE=configmap-secret` for the `nvflare deploy k8s stage`
  ConfigMap/Secret path.

## Create a Local OpenShift Cluster

Use the CRC helper scripts only when you need a single-node Red Hat OpenShift
Local cluster for development or testing. Production OpenShift clusters are
platform-specific; create those with your organization's approved installer or
cloud service workflow, then use the deployment scripts here against that
cluster.

Before using the local-cluster scripts, install Red Hat OpenShift Local so the
`crc` command is available, download your Red Hat OpenShift pull secret from
`https://console.redhat.com/openshift/create/local`, enable host hardware
virtualization, and make sure the host has enough CPU, memory, and disk for
OpenShift plus the NVFlare test pods. The create script defaults to 6 vCPUs,
24576 MiB memory, and 120 GiB disk. The deployment scripts make that local
configuration practical by requesting `500m` CPU and `1Gi` memory for each of
the three parent pods. Override `PARENT_CPU` and `PARENT_MEMORY` if the parent
workload needs more resources; increase `CRC_CPUS` and `CRC_MEMORY` to match.
If parent resources are omitted from a general `nvflare deploy prepare`
configuration, the generated Helm chart requests `2` CPU and `8Gi` memory per
parent pod, which does not fit this three-parent example on the default CRC
size after OpenShift overhead.

As a verified alternative for parent pods that need the generated `2` CPU and
`8Gi` memory requests, or for heavier workloads, resize an existing CRC cluster
before restarting it:

```bash
crc stop
crc config set cpus 14
crc config set memory 65536
bash examples/devops/openshift/scripts/start_openshift_cluster.sh
bash examples/devops/openshift/scripts/k8s_e2e.sh
```

This `14` vCPU / `65536` MiB configuration was verified with the complete
example: after the restart, rerunning `k8s_e2e.sh` allowed the submitted job to
reach `FINISHED:COMPLETED`. Make sure the host has enough capacity before using
these settings.

Use `scripts/create_openshift_cluster.sh` for first-time local CRC setup. It
validates that `crc` exists, requires `PULL_SECRET_FILE` when the cluster will
be started, writes CRC settings such as resource sizing and shared-directory
behavior, runs `crc setup`, and starts the cluster by delegating to
`scripts/start_openshift_cluster.sh` unless `START_AFTER_CREATE=false` is set.

```bash
export PULL_SECRET_FILE="$HOME/Downloads/pull-secret.txt"
export NAMESPACE=nvflare-e2e

bash examples/devops/openshift/scripts/create_openshift_cluster.sh
```

Use `scripts/start_openshift_cluster.sh` after CRC has already been configured,
or when restarting after `crc stop`. It runs `crc start` when needed, adds the
CRC-provided `oc` to `PATH` if needed, waits for OpenShift to report running,
logs in with `oc`, creates or selects `NAMESPACE`, and prints the console URL
and available StorageClasses.

```bash
PULL_SECRET_FILE="$HOME/Downloads/pull-secret.txt" \
bash examples/devops/openshift/scripts/start_openshift_cluster.sh
```

Run scripts from the repository root. Build the maintained images from `docker/Dockerfile.parent` and `docker/Dockerfile.job`, push them to a registry the cluster can pull from, then set `IMAGE` to the parent image and `JOB_IMAGE` to the workload image. Podman is supported for these build and push steps and is typically available by default on RHEL OpenShift hosts; Docker can be used instead by setting `CONTAINER_TOOL=docker`, and some RHEL installations provide `docker` as a Podman alias. A Docker daemon is not required. `ADMIN_IMAGE` defaults to `IMAGE`, so the parent image can also be used for the temporary admin pod. The parent image needs NVFlare with the `K8S` extra/Kubernetes Python client. A custom `COPY_IMAGE` needs `sh`, `sleep`, and `tar`; `JOB_IMAGE` only needs `tar` when the job workload itself needs it.

```bash
export PARENT_IMAGE=registry.example.com/nvflare-parent:dev
export WORKLOAD_IMAGE=registry.example.com/nvflare-job:dev
export CONTAINER_TOOL="${CONTAINER_TOOL:-podman}"

"$CONTAINER_TOOL" build -t "$PARENT_IMAGE" -f docker/Dockerfile.parent .
"$CONTAINER_TOOL" build -t "$WORKLOAD_IMAGE" -f docker/Dockerfile.job .
"$CONTAINER_TOOL" push "$PARENT_IMAGE"
"$CONTAINER_TOOL" push "$WORKLOAD_IMAGE"
```

After the images are pushed, keep `PARENT_IMAGE` and `WORKLOAD_IMAGE` in the same shell and map them to the variables consumed by `k8s_e2e.sh`:

```bash
export IMAGE="$PARENT_IMAGE"
export JOB_IMAGE="$WORKLOAD_IMAGE"
export NAMESPACE=nvflare-e2e

bash examples/devops/openshift/scripts/k8s_e2e.sh
```

The e2e script defaults to the manual PVC-copy staging path. To verify the
ConfigMap/Secret staging path instead, run the same command with:

```bash
export WORKSPACE_STAGING_MODE=configmap-secret
```

The watch tool requires the Python `rich` package:

```bash
python3 -m pip install rich
```

Clean up generated resources and stop OpenShift Local:

```bash
bash examples/devops/openshift/scripts/cleanup_openshift_cluster.sh
```
