# Azure NVFlare Deployment Example

This directory contains Azure-specific example pieces for testing NVFlare on
AKS. These scripts are for development and smoke testing only; they are not
production quality deployment guidance.

Current supported path:
- create an AKS Automatic cluster with `examples/devops/azure/aks/create_cluster.sh`
- deploy NVFlare with `examples/devops/multicloud/deploy.py`

Start from `examples/devops/multicloud/all-clouds.yaml` and trim it to an Azure-only
topology if you do not want to test multiple clouds.

## Azure-specific Values

The Azure deployment uses the following storage layout:

- `nvflws`
  - storage class: `managed-csi`
  - access mode: `ReadWriteOnce`
  - purpose: per-site workspace PVC
- `nvfldata`
  - storage class: `managed-csi`
  - access mode: `ReadWriteOnce`
  - purpose: study data / job data

An Azure server deployment also requires:
- `clouds.azure.resource_group`
- `clouds.azure.location`

These values are used by `deploy.py` to reserve and release the Azure Public IP
for the server `LoadBalancer` service.

## Required Steps

### 1. Prerequisites

You need:
- `az`
- `kubectl`
- `helm`
- NVFlare development dependencies installed in the repo virtual environment
- `nvflare` available either on `$PATH` or at `.venv/bin/nvflare`

Example setup from the NVFlare repository root:

```bash
az login
az aks install-cli
uv venv --python 3.14 .venv
uv pip install -e .[dev_mac]
```

You also need an NVFlare container image pushed to Azure Container Registry or
another registry that AKS can pull from.

Quick ACR example:

```bash
az acr login --name <acr-name>
docker tag nvflare-job:latest <acr>.azurecr.io/<repo>:<tag>
docker push <acr>.azurecr.io/<repo>:<tag>
```

### 2. Create The AKS Cluster

```bash
RESOURCE_GROUP=myResourceGroup CLUSTER_NAME=myAKSAutomaticCluster LOCATION=westus2 \
  ./examples/devops/azure/aks/create_cluster.sh
```

This script:
- creates the AKS Automatic cluster
- saves kubeconfig to `examples/devops/.tmp/kubeconfigs/azure.yaml`

### 3. Create An Azure-only Config

Copy the shipped config to a local ignored path:

```bash
mkdir -p examples/devops/.tmp/azure
cp examples/devops/multicloud/all-clouds.yaml examples/devops/.tmp/azure/azure.yaml
cp examples/devops/multicloud/project.yml examples/devops/.tmp/azure/project.yml
```

Edit `examples/devops/.tmp/azure/azure.yaml` and `examples/devops/.tmp/azure/project.yml`:
- add `project_file: project.yml`
- keep one Azure server and any Azure clients you want to test
- remove GCP and AWS participants from both files
- set `clouds.azure.prepare.parent.docker_image` to a real image
- set `clouds.azure.resource_group`
- set `clouds.azure.location`

Minimal config shape:

```yaml
name: azure-test
project_file: project.yml

participants:
  - { name: azure-server,   cloud: azure, namespace: nvflare-server,   role: server }
  - { name: azure-client-1, cloud: azure, namespace: nvflare-client-1, role: client }

clouds:
  azure:
    resource_group: myResourceGroup
    location: westus2
    prepare:
      runtime: k8s
      parent:
        docker_image: <acr>.azurecr.io/<repo>:<tag>
        workspace_pvc: nvflws
        workspace_mount_path: /var/tmp/nvflare/workspace
      job_launcher:
        default_python_path: /usr/local/bin/python3
    helm_overrides: []
    pvc_config:
      nvflws:   { sc: managed-csi, access: ReadWriteOnce, size: 10Gi }
      nvfldata: { sc: managed-csi, access: ReadWriteOnce, size: 1Gi  }
```

### 4. Deploy

```bash
python3 examples/devops/multicloud/deploy.py --config examples/devops/.tmp/azure/azure.yaml up
```

The tool is config-driven, not `-cloud azure` driven.

### 5. Inspect Or Tear Down

```bash
python3 examples/devops/multicloud/deploy.py --config examples/devops/.tmp/azure/azure.yaml status
python3 examples/devops/multicloud/deploy.py --config examples/devops/.tmp/azure/azure.yaml down
```

Optional dry run:

```bash
python3 examples/devops/multicloud/deploy.py --config examples/devops/.tmp/azure/azure.yaml --dry-run up
```

## Common Ops

Check parent pod status:

```bash
kubectl get pods -n nvflare-server
kubectl get pods -n nvflare-client-1
kubectl describe pod <pod-name> -n nvflare-server
kubectl logs <pod-name> -n nvflare-server
```

Example `meta.json` launcher settings for AKS job pods:

```json
{
  "name": "hello-numpy-k8s",
  "min_clients": 1,
  "deploy_map": {
    "app": ["@ALL"]
  },
  "resource_spec": {},
  "launcher_spec": {
    "default": {
      "k8s": {
        "image": "mynvflareregistry.azurecr.io/nvflare/nvflare:2.8.0",
        "cpu": "1000m",
        "memory": "4Gi"
      }
    }
  }
}
```

Use `cpu` and `memory` alone for the common case. NVFlare treats them as the
limit values and, when `cpu_request` or `memory_request` is omitted, mirrors
the same values into the pod requests.

## Notes

- The server is exposed through an Azure `LoadBalancer` service backed by a
  reserved Azure Public IP.
- `managed-csi` is used for both the site workspace PVC and the study data PVC
  in the current Azure deployment path.
- K8s-launched job pods must specify `launcher_spec.<site>.k8s.image`.
- For AKS Automatic job pods launched via `K8sJobLauncher`, specify
  `launcher_spec.<site>.k8s.cpu` and `launcher_spec.<site>.k8s.memory` for
  predictable scheduling.
- If `python3 examples/devops/multicloud/deploy.py ...` fails with
  `ModuleNotFoundError: No module named 'yaml'`, install `pyyaml` in the Python
  environment you are using.

## See Also

- [AKS README](aks/README.md)
- [Multicloud README](../multicloud/README.md)
