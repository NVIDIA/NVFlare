# Google Cloud GKE Autopilot Test Tooling

This directory contains:

- `create_cluster.sh` and `delete_cluster.sh` for a GKE Autopilot lifecycle plus NVFlare bootstrap helpers
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

Run these commands from the `devops/gcp/gke` directory.

Create the GKE Autopilot cluster and fetch credentials:

```bash
./create_cluster.sh
```

The create script also:

- saves a dedicated kubeconfig to `.tmp/kubeconfigs/gcp.yaml`
- enables `file.googleapis.com`
- creates a custom `filestore-rwx` `StorageClass` for the cluster VPC

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

## NVFlare Deployment Setup

After `./create_cluster.sh`, the script has already done the cluster-side storage bootstrap:

- created `.tmp/kubeconfigs/gcp.yaml`
- enabled `file.googleapis.com`
- created the `filestore-rwx` `StorageClass`

The generated `filestore-rwx` class uses the cluster VPC automatically. If you hit Filestore zone-capacity issues, rerun cluster creation with a pinned zone:

```bash
FILESTORE_ZONE=us-central1-a ./create_cluster.sh
```

Notes:
- Filestore has a minimum 1 TiB instance size (roughly $200/month)
- provisioning typically takes 3-5 minutes
- the CSI driver does not retry other zones on capacity failure, which is why `FILESTORE_ZONE` exists as an override

What is still manual before pushing NVFlare images:

- enable `artifactregistry.googleapis.com`
- create the Artifact Registry repository
- grant the node service account pull access

### Container registry (Artifact Registry)

```bash
gcloud artifacts repositories create nvflare \
  --repository-format=docker --location=us-central1 --project=<project>
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# Grant GKE nodes pull access
PROJECT_NUM=$(gcloud projects describe <project> --format="value(projectNumber)")
gcloud artifacts repositories add-iam-policy-binding nvflare \
  --project=<project> --location=us-central1 \
  --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

### Static IP (for server LoadBalancer)

```bash
gcloud compute addresses create <name> --region=<region> --project=<project>
gcloud compute addresses describe <name> --region=<region> --format="value(address)"
```

If you use `devops/multicloud/scripts/deploy.py up`, it can auto-reserve the server IP for you. This step is only needed for manual Helm installs.

### Helm install overrides for Autopilot

```bash
helm install <name> <chart> -n <namespace> --create-namespace \
  --set hostPortEnabled=false \
  --set tcpConfigMapEnabled=false \
  --set service.type=LoadBalancer \
  --set service.loadBalancerIP=<static-ip>
```

## Notes

- GKE Autopilot is the closest Google Cloud equivalent to EKS Auto Mode.
- GKE manages the node infrastructure for you. A simple deployment is enough to verify the cluster can schedule work.
- The create script creates a dedicated VPC network if it does not already exist, and the delete script removes it only when the script created it.
