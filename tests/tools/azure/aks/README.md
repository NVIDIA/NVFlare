# Azure AKS Automatic Test Tooling

This directory contains:

- `create_cluster.sh` and `delete_cluster.sh` for a minimal AKS Automatic lifecycle
- `inflate.yaml` for a tiny non-FLARE workload to verify an AKS Automatic cluster can schedule pods

## Prerequisites

- `az` installed and authenticated
- `kubectl` installed

If you need `kubectl`, install it with:

```bash
az aks install-cli
```

## Quick Start

Run these commands from the `tests/tools/azure/aks` directory.

Create the AKS Automatic cluster and fetch credentials:

```bash
./create_cluster.sh
```

The script defaults to:

```bash
RESOURCE_GROUP=myResourceGroup
CLUSTER_NAME=myAKSAutomaticCluster
LOCATION=westus2
```

Verify the cluster:

```bash
kubectl get nodes
```

Deploy the sample workload:

```bash
kubectl apply -f inflate.yaml
kubectl get pods -w
```

In another terminal, verify the pod and node:

```bash
kubectl get pods -o wide
kubectl get nodes
```

Delete the sample workload:

```bash
kubectl delete -f inflate.yaml
```

Delete the cluster when you are done:

```bash
./delete_cluster.sh
```

If you want to override the defaults:

```bash
RESOURCE_GROUP=myOtherGroup CLUSTER_NAME=myOtherCluster LOCATION=eastus2 ./create_cluster.sh
```

Use the same environment variables with `./delete_cluster.sh` if you created the cluster with non-default names.

## Notes

- AKS Automatic is the closest Azure equivalent to EKS Auto Mode.
- This example uses `westus2` because it had better total regional quota than `eastus` in our subscription checks.
- If this is the first workload you deploy, AKS node auto provisioning can take a few minutes to create capacity for the pod.
- AKS Automatic can still fail if the subscription does not have enough per-family D-series quota for the VM sizes it selects.
