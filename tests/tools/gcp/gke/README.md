# Google Cloud GKE Autopilot Test Tooling

This directory contains:

- `create_cluster.sh` and `delete_cluster.sh` for a minimal GKE Autopilot lifecycle
- `inflate.yaml` for a tiny non-FLARE workload to verify a GKE Autopilot cluster can schedule pods

## Prerequisites

- `gcloud` installed and authenticated
- a Google Cloud project selected
- `kubectl` installed
- `gke-gcloud-auth-plugin` installed

If you need to log in:

```bash
gcloud auth login
```

If you need to select a project:

```bash
gcloud config set project your-gcp-project-id
```

If you need to enable the GKE API:

```bash
gcloud services enable container.googleapis.com
```

If you need the GKE auth plugin for `kubectl`:

```bash
gcloud components install gke-gcloud-auth-plugin
gke-gcloud-auth-plugin --version
```

## Quick Start

Run these commands from the `tests/tools/gcp/gke` directory.

Create the GKE Autopilot cluster and fetch credentials:

```bash
./create_cluster.sh
```

The script defaults to:

```bash
CLUSTER_NAME=gke-auto-test
LOCATION=us-central1
PROJECT_ID=your-gcp-project-id
NETWORK_NAME=gke-test
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
PROJECT_ID=your-gcp-project-id CLUSTER_NAME=my-gke-test LOCATION=us-east1 NETWORK_NAME=my-gke-test ./create_cluster.sh
```

Use the same environment variables with `./delete_cluster.sh` if you created the cluster with non-default names.

## Notes

- GKE Autopilot is the closest Google Cloud equivalent to EKS Auto Mode.
- GKE manages the node infrastructure for you. A simple deployment is enough to verify the cluster can schedule work.
- The create script creates a dedicated VPC network if it does not already exist, and the delete script removes that network after the cluster is deleted.
