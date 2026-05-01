# Multicloud NVFlare Deployment

YAML-driven deploy of one NVFlare server + N clients across existing
Kubernetes clusters.

Shipped config:
- `all-clouds.yaml` — GCP server with GCP, AWS, and Azure clients

Cluster topology is defined in the YAML (`clouds:` + `participants:`).
The deploy tool reads kubeconfigs from `.tmp/kubeconfigs/<cloud>.yaml`.
`fetch_kubeconfigs.py` discovers clusters from the active cloud CLI contexts
and supports environment variable overrides when discovery is ambiguous.

The deploy tool still supports GCP, AWS, or Azure as the server cloud.
Cloud-specific behavior lives in provider modules under
`devops/multicloud/clouds/`; `deploy.py` owns the common flow.

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

## Fetch kubeconfigs

If the clusters already exist, fetch kubeconfigs from the active cloud CLI
contexts:

```bash
python devops/multicloud/fetch_kubeconfigs.py --config devops/multicloud/all-clouds.yaml
```

This writes `.tmp/kubeconfigs/gcp.yaml`, `.tmp/kubeconfigs/aws.yaml`, and
`.tmp/kubeconfigs/azure.yaml`. The script uses:

- GCP: active `gcloud` project, and the only matching GKE cluster unless `GCP_CLUSTER`/`GCP_LOCATION` are set.
- AWS: `AWS_REGION`, `AWS_DEFAULT_REGION`, or `aws configure get region`, and the only EKS cluster unless `AWS_CLUSTER` is set.
- Azure: active `az` subscription, and the only AKS cluster unless `AZURE_CLUSTER`/`AZURE_RESOURCE_GROUP` are set.

Preview the cloud CLI commands without writing kubeconfigs:

```bash
python devops/multicloud/fetch_kubeconfigs.py --dry-run
```

## Deploy

Default config is `all-clouds.yaml`. Image URLs come from the per-cloud
`prepare.parent.docker_image` values.

Build and push the image tags referenced by a config:

```bash
python devops/multicloud/build_and_push.py --config devops/multicloud/all-clouds.yaml
```

Before running the all-clouds config:
- confirm `clouds.gcp.prepare.parent.docker_image`
- confirm `clouds.aws.prepare.parent.docker_image`
- confirm `clouds.azure.prepare.parent.docker_image`
- run `python devops/multicloud/fetch_kubeconfigs.py` if `.tmp/kubeconfigs/*.yaml`
  does not exist yet

Deploy:

```bash
python devops/multicloud/deploy.py --config devops/multicloud/all-clouds.yaml up
```

`up` always re-runs `nvflare provision`, runs `nvflare deploy prepare` for
each participant, refreshes the staged startup/local kit files in each
participant workspace PVC, uninstalls any existing Helm release for each
participant, then installs the prepared chart. The cloud IP name is
deterministic from the config `name`, for example `all-clouds` uses
`nvflare-all-clouds`. If an IP with that name already exists, `up` reuses it.

The server is deployed first and must become available before clients are
started. Clients are deployed in parallel. `down` tears down clients in
parallel first, then tears down the server and releases the deterministic
cloud IP.

## Dev Cluster Runbook

For an isolated dev cluster, use `setting-up-dev-cluster.md`. It covers
creating a temporary config, choosing unique names/namespaces, setting image
tags, building/pushing images, fetching kubeconfigs, deploying, validating,
and tearing down.

For Codex-driven setup, prompt Codex with:

```text
Use devops/multicloud/setting-up-dev-cluster.md to create a multicloud dev cluster.
```

## Inspect / tear down

```bash
python devops/multicloud/deploy.py status
python devops/multicloud/deploy.py down
python devops/multicloud/deploy.py --dry-run up       # print commands, don't execute
```

## Flags

| Flag | Default | Applies to |
|------|---------|-----------|
| `--config <path>` | `devops/multicloud/all-clouds.yaml` | all subcommands |
| `--dry-run` | false | all subcommands |

## Live dashboard

`k8sview.py` renders a live all-cloud dashboard by default. It reads
`.tmp/kubeconfigs/*.yaml`, discovers NVFlare namespaces, groups them into
systems, and shows cluster node/pod totals plus server/client sites, pod IPs,
service IPs, external server IPs, PVCs, and job pod counts:

```bash
python devops/multicloud/k8sview.py
```

Pass `--config` for a focused pod / PVC / LB / events view across every
participant in one deploy config:

```bash
python devops/multicloud/k8sview.py --config devops/multicloud/all-clouds.yaml
```

`k8sview.py` automatically deletes terminal pods (`Succeeded` or `Failed`) after
5 minutes.

Requires `rich` and `kubernetes` (`uv pip install rich kubernetes`).

## Dry-run regression checks

The unit tests include golden dry-run output for representative GCP-server,
AWS-server, Azure-server, and all-clouds topologies. After changing
provider logic or deploy command generation, run:

```bash
python -m pytest tests/unit_test/devops/multicloud_deploy_test.py::TestDryRunGoldenOutput -q
```

## See also

- `devops/gcp/gke/README.md` — GKE cluster setup
- `devops/aws/eks/README.md` — EKS cluster setup + SELinux/ECR notes
- `devops/azure/README.md` — Azure run steps and required values
- `devops/azure/aks/README.md` — AKS setup for the Azure deployment path
