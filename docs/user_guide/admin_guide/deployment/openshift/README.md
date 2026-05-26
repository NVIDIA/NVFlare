# OpenShift Deployment Helpers

This directory contains the OpenShift-specific NVFlare deployment guide and helper scripts.

- `index.rst` is the user guide page for OpenShift deployment.
- `scripts/Dockerfile` builds a restricted-SCC-compatible NVFlare image.
- `scripts/create_openshift_cluster.sh` configures Red Hat OpenShift Local (CRC) and optionally starts it.
- `scripts/start_openshift_cluster.sh` starts CRC, logs in with `oc`, and prepares the target project.
- `scripts/openshift_k8s_provision.sh` runs `nvflare provision` for the sample server, `site-1`, `site-2`, and admin.
- `scripts/openshift_k8s_deploy.sh` prepares K8s startup kits, stages PVC workspaces, installs Helm charts, and verifies parent pods can import the Kubernetes Python client.
- `scripts/openshift_k8s_submit_job.sh` submits `hello-numpy` from an in-cluster admin pod and waits for successful completion.
- `scripts/openshift_k8s_watch.sh` shows an in-place live Rich pod table for the created pods.
- `scripts/openshift_k8s_watch.py` implements the Rich table used by the shell wrapper.
- `scripts/openshift_k8s_e2e.sh` runs provision, deploy, and submit in order.

Run scripts from the repository root. At minimum, the deploy and submit phases need `IMAGE` set to a cluster-pullable NVFlare image with the `K8S` extra/Kubernetes Python client:

```bash
export IMAGE=registry.example.com/nvflare-openshift:dev
export NAMESPACE=nvflare-e2e

bash docs/user_guide/admin_guide/deployment/openshift/scripts/openshift_k8s_e2e.sh
```

The watch tool requires the Python `rich` package:

```bash
python3 -m pip install rich
```
