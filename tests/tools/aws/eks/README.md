# AWS EKS Auto Mode Test Tooling

This directory contains:

- `cluster.yaml` for a minimal EKS Auto Mode cluster
- `inflate.yaml` for a tiny non-FLARE workload to verify the cluster can schedule pods

## Prerequisites

- `aws` CLI installed and authenticated
- `eksctl` installed
- `kubectl` installed

## Quick Start

Run these commands from the `tests/tools/aws/eks` directory.

Edit `cluster.yaml` if you want a different cluster name or region.

Create the cluster:

```bash
eksctl create cluster -f cluster.yaml
```

Check the built-in Auto Mode node pools:

```bash
kubectl get nodepools
```

New Auto Mode clusters can start with zero nodes. Deploy the sample workload to force node provisioning:

```bash
kubectl apply -f inflate.yaml
kubectl get events -w --sort-by '.lastTimestamp'
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
eksctl delete cluster -f cluster.yaml --wait
```

## Notes

- The sample workload is based on the AWS EKS Auto Mode `inflate` example.
- The `eks.amazonaws.com/compute-type: auto` selector makes this workload land on Auto Mode nodes.
