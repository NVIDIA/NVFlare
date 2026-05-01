# Multicloud NVFlare Deployment

YAML-driven deploy of one NVFlare server + N clients across existing
Kubernetes clusters.

Shipped configs:
- `gcp-server.yaml` — GCP server, GCP client, AWS client
- `aws-server.yaml` — AWS server, GCP client, AWS client
- `azure-server.yaml` — Azure server, Azure client

Cluster topology is defined in the YAML (`clouds:` + `participants:`);
everything else comes from that file. Region / project / EKS cluster
name are autoderived from each kubeconfig's current-context where
possible, so the YAML stays minimal. Azure uses explicit
`resource_group` and `location` values in the config.

## Prerequisites

- Clusters created via `devops/gcp/gke/create_cluster.sh` and
  `devops/aws/eks/create_cluster.sh` — each saves a kubeconfig under
  `.tmp/kubeconfigs/<cloud>.yaml`.
- Azure clusters can be created via `devops/azure/aks/create_cluster.sh`
  which saves `.tmp/kubeconfigs/azure.yaml`.
- NVFlare image pushed to the registry used by each cloud config.
- `helm`, `kubectl`, and the cloud CLIs on `$PATH`.
- Python with `pyyaml` installed.
- `nvflare` available on `$PATH` or at `.venv/bin/nvflare`.

## Deploy

```bash
python devops/multicloud/deploy.py up
```

Default config is `gcp-server.yaml`; pass `--config devops/multicloud/aws-server.yaml` to switch topology. Image URLs come from the config (`clouds.<x>.image`).

Azure example:

```bash
python3 devops/multicloud/deploy.py --config devops/multicloud/azure-server.yaml up
```

Before running the Azure config:
- update `devops/multicloud/azure-server.yaml` with a real image
- confirm `clouds.azure.resource_group`
- confirm `clouds.azure.location`

Re-run after a failure — idempotent. Skips namespaces / PVCs / helm
releases that already exist; always re-runs `nvflare provision` (fast).

## Inspect / tear down

```bash
python devops/multicloud/deploy.py status
python devops/multicloud/deploy.py down
python devops/multicloud/deploy.py --dry-run up       # print commands, don't execute
```

Azure example:

```bash
python3 devops/multicloud/deploy.py --config devops/multicloud/azure-server.yaml status
python3 devops/multicloud/deploy.py --config devops/multicloud/azure-server.yaml down
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
- `devops/azure/README.md` — Azure run steps and required values
- `devops/azure/aks/README.md` — AKS setup for the Azure deployment path
