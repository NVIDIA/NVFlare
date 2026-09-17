# Azure AKS Automatic

Cluster lifecycle for the Azure multicloud NVFlare path.

## Working Directory

Download this example and enter the preserved provider directory:

```bash
nvflare examples get devops-azure-aks
cd devops-azure-aks/azure/aks
```

From an NVFlare source checkout, use `cd examples/devops/azure/aks` instead.
The scripts keep generated kubeconfig files under the downloaded example root.

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
- does not modify `~/.kube/config`

NVFlare uses the AKS default class `managed-csi` for its RWO PVCs.

## Verify + smoke test

```bash
export KUBECONFIG="$(pwd)/../../.tmp/kubeconfigs/azure.yaml"
kubectl get nodes
kubectl apply -f inflate.yaml
kubectl get pods -w
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
