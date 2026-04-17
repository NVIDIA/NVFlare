# Multicloud Deployment for NVFlare on Kubernetes

Deploy NVFlare (1 server + 2 clients) across existing k8s clusters.

- GCP GKE Autopilot: server + 1 client
- AWS EKS Auto Mode: 1 client

## Prerequisites

- Clusters created via `devops/gcp/gke/create_cluster.sh` and
  `devops/aws/eks/create_cluster.sh` (sets up StorageClasses, EFS)
- Docker image built from current branch and pushed to both registries
- Kubeconfigs in `.tmp/kubeconfigs/`

## Deploy

```bash
python devops/multicloud/scripts/deploy.py up \
  --gcp-image us-central1-docker.pkg.dev/<project>/nvflare/nvflare:<tag> \
  --aws-image <account>.dkr.ecr.<region>.amazonaws.com/nvflare/nvflare:<tag>
```

Or with env vars:

```bash
GCP_IMAGE=<image> AWS_IMAGE=<image> python devops/multicloud/scripts/deploy.py up
```

The script is **resumable** — re-run after a failure and it skips
already-completed steps (namespaces, PVCs, helm releases).

## Status

```bash
python devops/multicloud/scripts/deploy.py status
```

## Destroy

```bash
python devops/multicloud/scripts/deploy.py down
```

## Options

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `--gcp-image` | `GCP_IMAGE` | (required) | GCP container image |
| `--aws-image` | `AWS_IMAGE` | (optional) | AWS container image (ECR) |
| `--server-ip` | `GCP_SERVER_IP` | auto-reserve | Static IP for server |
| `--gcp-project` | `GCP_PROJECT` | auto-detect | GCP project ID |
| `--gcp-region` | `GCP_REGION` | us-central1 | GCP region |
| `--force-provision` | | false | Re-provision even if output exists |

## Documentation

- `devops/gcp/gke/README.md` — GKE cluster setup + NVFlare prerequisites
- `devops/aws/eks/README.md` — EKS cluster setup + NVFlare prerequisites
- `BUGS.md` — issues found and resolution status
- `DEPLOY_NOTES.md` — operational notes per cloud
- `RECOMMENDATIONS.md` — what to fix in core vs deploy tooling
