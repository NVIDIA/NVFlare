# Azure AKS Automatic

Cluster lifecycle. **NVFlare deployment on AKS is not yet supported** by
`devops/multicloud/scripts/deploy.py` (Azure IP reservation raises
`NotImplementedError`).

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
