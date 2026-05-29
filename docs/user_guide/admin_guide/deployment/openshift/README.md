# OpenShift Deployment Helpers

This directory contains the OpenShift-specific NVFlare deployment guide and helper scripts.

- `index.rst` is the user guide page for OpenShift deployment.
- `scripts/Dockerfile` builds a restricted-SCC-compatible NVFlare image.
- `scripts/create_openshift_cluster.sh` configures Red Hat OpenShift Local (CRC) and optionally starts it.
- `scripts/start_openshift_cluster.sh` starts CRC, logs in with `oc`, and prepares the target project.
- `scripts/cleanup_openshift_cluster.sh` deletes scripted deployment resources and stops CRC.
- `scripts/k8s_provision.sh` runs `nvflare provision` for the sample server, `site-1`, `site-2`, and admin.
- `scripts/k8s_deploy.sh` prepares K8s startup kits, stages PVC workspaces, installs Helm charts, and verifies parent pods can import the Kubernetes Python client.
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

bash docs/user_guide/admin_guide/deployment/openshift/scripts/create_openshift_cluster.sh
```

Use `scripts/start_openshift_cluster.sh` after CRC has already been configured,
or when restarting after `crc stop`. It runs `crc start` when needed, adds the
CRC-provided `oc` to `PATH` if needed, waits for OpenShift to report running,
logs in with `oc`, creates or selects `NAMESPACE`, and prints the console URL
and available StorageClasses.

```bash
PULL_SECRET_FILE="$HOME/Downloads/pull-secret.txt" \
bash docs/user_guide/admin_guide/deployment/openshift/scripts/start_openshift_cluster.sh
```

Run scripts from the repository root. At minimum, the deploy and submit phases need `IMAGE` set to a cluster-pullable NVFlare image with the `K8S` extra/Kubernetes Python client, the `nvflare` CLI, `sh`, and `sleep`. The default `ADMIN_IMAGE` and `COPY_IMAGE` both use `IMAGE`; any custom value for those images also needs `tar` because the scripts copy files into pods with `oc cp`/`kubectl cp`. `JOB_IMAGE` only needs `tar` when the job workload itself needs it.

```bash
export IMAGE=registry.example.com/nvflare-openshift:dev
export NAMESPACE=nvflare-e2e

bash docs/user_guide/admin_guide/deployment/openshift/scripts/k8s_e2e.sh
```

The watch tool requires the Python `rich` package:

```bash
python3 -m pip install rich
```

Clean up generated resources and stop OpenShift Local:

```bash
bash docs/user_guide/admin_guide/deployment/openshift/scripts/cleanup_openshift_cluster.sh
```
