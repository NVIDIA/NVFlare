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
- `scripts/k8s_e2e.sh` runs provision, deploy, and submit in order.

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
24576 MiB memory, and 120 GiB disk.

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

Run scripts from the repository root. Build the maintained images from `docker/Dockerfile.parent` and `docker/Dockerfile.job`, push them to a registry the cluster can pull from, then set `IMAGE` to the parent image and `JOB_IMAGE` to the workload image. `ADMIN_IMAGE` defaults to `IMAGE`, so the parent image can also be used for the temporary admin pod. The parent image needs NVFlare with the `K8S` extra/Kubernetes Python client. A custom `COPY_IMAGE` needs `sh`, `sleep`, and `tar`; `JOB_IMAGE` only needs `tar` when the job workload itself needs it.

```bash
export IMAGE=registry.example.com/nvflare-parent:dev
export JOB_IMAGE=registry.example.com/nvflare-job:dev
export NAMESPACE=nvflare-e2e

bash examples/devops/openshift/scripts/k8s_e2e.sh
```

The watch tool requires the Python `rich` package:

```bash
python3 -m pip install rich
```

Clean up generated resources and stop OpenShift Local:

```bash
bash examples/devops/openshift/scripts/cleanup_openshift_cluster.sh
```
