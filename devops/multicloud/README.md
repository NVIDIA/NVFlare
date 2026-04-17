# Multicloud NVFlare Deployment

YAML-driven deploy of one NVFlare server + N clients across existing
Kubernetes clusters.

Shipped configs:
- `gcp-server.yaml` — GCP server, GCP client, AWS client
- `aws-server.yaml` — AWS server, GCP client, AWS client

Cluster topology is defined in the YAML (`clouds:` + `participants:`);
everything else comes from that file. Region / project / EKS cluster
name are autoderived from each kubeconfig's current-context, so the
YAML stays minimal.

## Prerequisites

- Clusters created via `devops/gcp/gke/create_cluster.sh` and
  `devops/aws/eks/create_cluster.sh` — each saves a kubeconfig under
  `.tmp/kubeconfigs/<cloud>.yaml`.
- NVFlare image pushed to both registries (GAR + ECR).
- `helm`, `kubectl`, and the cloud CLIs on `$PATH`.

## Deploy

```bash
python devops/multicloud/deploy.py up
```

Default config is `gcp-server.yaml`; pass `--config devops/multicloud/aws-server.yaml` to switch topology. Image URLs come from the config (`clouds.<x>.image`).

Re-run after a failure — idempotent. Skips namespaces / PVCs / helm
releases that already exist; always re-runs `nvflare provision` (fast).

## Inspect / tear down

```bash
python devops/multicloud/deploy.py status
python devops/multicloud/deploy.py down
python devops/multicloud/deploy.py --dry-run up       # print commands, don't execute
```

## Flags

| Flag | Default | Applies to |
|------|---------|-----------|
| `--config <path>` | `devops/multicloud/gcp-server.yaml` | all subcommands |
| `--dry-run` | false | all subcommands |

## Live dashboard

`k8sview.py` renders a live pod / PVC / LB / events view across every
participant in a config:

```bash
python devops/multicloud/k8sview.py --config devops/multicloud/gcp-server.yaml
```

Requires `rich` and `kubernetes` (`uv pip install rich kubernetes`).

## See also

- `devops/gcp/gke/README.md` — GKE cluster setup
- `devops/aws/eks/README.md` — EKS cluster setup + SELinux/ECR notes
