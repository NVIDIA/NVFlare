# Azure NVFlare Deployment

This directory contains the Azure-specific pieces for running NVFlare on
AKS.

Current supported path:
- create an AKS Automatic cluster with `devops/azure/aks/create_cluster.sh`
- deploy one NVFlare server and one NVFlare client with
  `devops/multicloud/deploy.py`

The current shipped multicloud config is:
- `devops/multicloud/azure-server.yaml`

## Azure-specific values

The Azure deployment uses the following storage layout:

- `nvflws`
  - storage class: `managed-csi`
  - access mode: `ReadWriteOnce`
  - purpose: per-site workspace PVC
- `nvfldata`
  - storage class: `managed-csi`
  - access mode: `ReadWriteOnce`
  - purpose: study data / job data

The Azure server path also requires:
- `clouds.azure.resource_group`
- `clouds.azure.location`

These are used by `deploy.py` to reserve and release the Azure Public IP
for the server `LoadBalancer` service.

## Required steps

### 1. Prerequisites

You need:
- `az`
- `kubectl`
- `helm`
- Python with `pyyaml` installed
- `nvflare` available either on `$PATH` or at `.venv/bin/nvflare`

Example setup:

```bash
az login
az aks install-cli
python3 -m pip install pyyaml
```

You also need an NVFlare container image pushed to Azure Container
Registry or another registry that AKS can pull from.

Quick ACR example:

```bash
az acr login --name <acr-name>
docker tag nvflare-job:latest <acr>.azurecr.io/<repo>:<tag>
docker push <acr>.azurecr.io/<repo>:<tag>
```

### 2. Create the AKS cluster

```bash
RESOURCE_GROUP=myResourceGroup CLUSTER_NAME=myAKSAutomaticCluster LOCATION=westus2 \
  ./devops/azure/aks/create_cluster.sh
```

This script:
- creates the AKS Automatic cluster
- saves kubeconfig to `.tmp/kubeconfigs/azure.yaml`

### 3. Edit the Azure multicloud config

Update azure-server.yaml:
- replace `image: <acr>.azurecr.io/<repo>:<tag>` with a real image
- confirm `resource_group`
- confirm `location`

Current config shape:

```yaml
name: azure-server

clouds:
  azure:
    kubeconfig: ../../.tmp/kubeconfigs/azure.yaml
    image: <acr>.azurecr.io/<repo>:<tag>
    resource_group: myResourceGroup
    location: westus2
    helm_overrides: []
    pvc_config:
      nvflws:   { sc: managed-csi,   access: ReadWriteOnce, size: 10Gi }
      nvfldata: { sc: managed-csi,   access: ReadWriteOnce, size: 1Gi  }

participants:
  - { name: azure-server,   cloud: azure, namespace: nvflare-server,   role: server }
  - { name: azure-client-1, cloud: azure, namespace: nvflare-client-1, role: client }
```

### 4. Deploy

Run the deploy tool with the Azure config:

```bash
python3 devops/multicloud/deploy.py --config devops/multicloud/azure-server.yaml up
```

This is the important part: the tool is config-driven, not
`-cloud azure` driven.

### 5. Inspect or tear down

```bash
python3 devops/multicloud/deploy.py --config devops/multicloud/azure-server.yaml status
python3 devops/multicloud/deploy.py --config devops/multicloud/azure-server.yaml down
```

Optional dry run:

```bash
python3 devops/multicloud/deploy.py --config devops/multicloud/azure-server.yaml --dry-run up
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
        "image": "mynvflareregistry.azurecr.io/nvflare/nvflare:2.7.2",
        "cpu": "1000m",
        "memory": "4Gi"
      }
    }
  }
}
```

Use `cpu` and `memory` alone for the common case. NVFlare treats them as
the limit values and, when `cpu_request` or `memory_request` is omitted,
mirrors the same values into the pod requests. Only add
`cpu_request`/`memory_request` when you explicitly want requests smaller
than limits.

## Notes

- The server is exposed through an Azure `LoadBalancer` service backed by
  a reserved Azure Public IP.
- `managed-csi` is used for both the site workspace PVC and the study
  data PVC in the current Azure deployment path.
- K8s-launched job pods must specify `launcher_spec.<site>.k8s.image`.
- For AKS Automatic job pods launched via `K8sJobLauncher`, specify
  `launcher_spec.<site>.k8s.cpu` and `launcher_spec.<site>.k8s.memory`
  for predictable scheduling. If `cpu_request` or `memory_request` is
  omitted, NVFlare mirrors the corresponding limit value into the
  request.
- If `python3 devops/multicloud/deploy.py ...` fails with
  `ModuleNotFoundError: No module named 'yaml'`, install `pyyaml` in the
  Python environment you are using.

## See also

- [AKS README](aks/README.md)
- [Multicloud README](../multicloud/README.md)
