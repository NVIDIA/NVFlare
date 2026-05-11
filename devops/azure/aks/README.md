# Azure AKS Automatic

Cluster lifecycle for the Azure multicloud NVFlare path.

For the end-to-end Azure deployment flow, config shape, and required
NVFlare values, start with [../README.md](../README.md).

## Prereqs

`az` (authenticated), `kubectl`.

```bash
az login                     # or: az login --use-device-code
az aks install-cli           # if kubectl missing
```

## Create

```bash
./create_cluster.sh
```

Defaults: `RESOURCE_GROUP=myResourceGroup`, `CLUSTER_NAME=myAKSAutomaticCluster`, `LOCATION=westus2`. Override via env vars (use the same ones on `delete_cluster.sh`).

This script:
- creates the AKS Automatic cluster
- saves kubeconfig to `.tmp/kubeconfigs/azure.yaml`

NVFlare uses the AKS default class `managed-csi` for its RWO PVCs.

## Verify + smoke test

```bash
kubectl get nodes
kubectl apply -f inflate.yaml && kubectl get pods -w
kubectl delete -f inflate.yaml
```

## Delete

```bash
./delete_cluster.sh
```

## Notes

- AKS Automatic is Azure's closest equivalent to EKS Auto Mode.
- First workload can take a few minutes while node auto-provisioning
  allocates VMs. Subscription D-series quota can block provisioning.
- NVFlare deploy and teardown commands live in [../README.md](../README.md).
