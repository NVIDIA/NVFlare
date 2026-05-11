# Multicloud NVFlare Deployment

YAML-driven deploy of one NVFlare server + N clients across existing
Kubernetes clusters.

## Audience And Scope

This example is for NVFlare developers and users who want a simple Kubernetes
deployment for multicloud testing, experimentation, and learning. It is not a
production deployment blueprint. Real production deployments usually have
site-specific cluster ownership, security policy, networking, registry,
storage, monitoring, and operational requirements that are outside the scope of
this example.

Shipped config:
- `all-clouds.yaml` — GCP server with GCP, AWS, and Azure clients

Cluster topology is defined in the YAML (`clouds:` + `participants:`).
The deploy tool reads kubeconfigs from `.tmp/kubeconfigs/<cloud>.yaml`.
`fetch_kubeconfigs.py` discovers clusters from the active cloud CLI contexts
and supports environment variable overrides when discovery is ambiguous.

The deploy tool supports GCP, AWS, or Azure as the server cloud.
Cloud-specific behavior lives in provider modules under
`devops/multicloud/clouds/`; `deploy.py` owns the common flow.

## Topology Selection

The `participants:` section is the primary control for what is deployed. The
default `all-clouds.yaml` topology creates one server and three clients:

```yaml
participants:
  - { name: gcp-server,     cloud: gcp,   namespace: nvflare-server,   role: server }
  - { name: gcp-client-1,   cloud: gcp,   namespace: nvflare-client-1, role: client }
  - { name: aws-client-2,   cloud: aws,   namespace: nvflare-client-2, role: client }
  - { name: azure-client-3, cloud: azure, namespace: nvflare-client-3, role: client }
```

Edit that list to choose a smaller topology. For example, to deploy only on
GCP, keep the GCP server and GCP client entries and remove the AWS/Azure client
entries. To deploy two clouds, keep only participants for those clouds. Exactly
one participant must have `role: server`; all other participants are clients.

`fetch_kubeconfigs.py`, `build_and_push.py`, and `deploy.py` operate on the
clouds referenced by `participants:`.

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

## Roles And Permissions

Cluster creation and FLARE site deployment can be separate roles:

- Cluster creator: permission to create and manage the target GKE, EKS, or AKS
  clusters, node pools/autoscaling settings, storage classes, and container
  registries.
- FLARE site deployer: permission to use the kubeconfig for each target site
  cluster and create/delete namespaces, PVCs, pods, services, deployments, and
  Helm release resources for that site.
- FLARE server site deployer: the FLARE site deployer permissions plus
  permission to reserve and release the public/static IP used by the server
  service.

The server cloud means the cloud where the FLARE server participant is
deployed. For GCP, the server IP is a regional compute address; for AWS, it is
an Elastic IP; for Azure, it is a Public IP in the configured resource group.
When AWS is the server cloud, the deployer also needs permission to describe
the EKS cluster and public subnets. When Azure is the server cloud,
`clouds.azure.resource_group` and `clouds.azure.location` must identify where
the Public IP is managed.

## Hardware And Sizing

This flow is intended for auto/autoscaling Kubernetes clusters. The FLARE
server and client parent pods are lightweight; the real CPU, memory, GPU, and
storage needs are driven by the jobs you submit. Job pod resources come from
the job/runtime configuration. Workspace and study-data PVC sizes are configured
under each cloud's `pvc_config`.

GPU jobs have not been validated by this example yet. GPU deployments need
cluster GPU capacity, device plugin/runtime support, and job resource requests
that match the target cloud and cluster setup.

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

## What Gets Created

`up` creates or refreshes:

- Local provision output under `devops/multicloud/.work/provision`.
- A prepared K8s runtime kit per participant from `nvflare deploy prepare`.
- A deterministic public/static server IP named from config `name`, for example
  `nvflare-all-clouds`.
- One Kubernetes namespace per participant.
- The configured PVCs in each participant namespace, including `nvflws` for the
  workspace and `nvfldata` for study data in the shipped config.
- A temporary `kit-copy-*` pod per participant to stage `startup/` and `local/`
  files into the workspace PVC.
- One Helm release per participant, which installs the parent Deployment and
  Service.

Job pods are not created by deploy. They are created later by the FLARE runtime
when jobs are submitted.

## Dev Cluster Runbook

For an isolated dev cluster, use `setting-up-dev-cluster.md`. It covers
creating a temporary config, choosing unique names/namespaces, setting image
tags, building/pushing images, fetching kubeconfigs, deploying, validating,
and tearing down.

For AI-agent-assisted setup, prompt the agent with:

```text
Use devops/multicloud/setting-up-dev-cluster.md to create a multicloud dev cluster.
```

## Inspect / tear down

```bash
python devops/multicloud/deploy.py status
python devops/multicloud/deploy.py down
python devops/multicloud/deploy.py --dry-run up       # print commands, don't execute
```

## Optional System Monitoring

Multicloud monitoring uses the FLARE connection instead of cross-cloud StatsD
networking. Clients collect FLARE system lifecycle metrics and stream them to
the server through CellNet. Only the server sends metrics to `statsd-exporter`
in the server cluster:

```text
client SysMetricsCollector
  -> ConvertToFedEvent
  -> FLARE/CellNet
  -> server RemoteMetricsReceiver
  -> server StatsDReporter
  -> statsd-exporter -> Prometheus -> Grafana
```

Enable this in a local config:

```yaml
monitoring:
  enabled: true
```

The runtime image must include the `MONITORING` extra because
`StatsDReporter` requires the `datadog` package. For example, build a runtime
image that installs `.[K8S,MONITORING]` before enabling monitoring.

When monitoring is enabled, `deploy.py up` applies
`devops/multicloud/monitoring-stack.yaml` in the server cluster before starting
FLARE. The stack runs in a namespace derived from the config name, for example
`all-clouds` uses `nvflare-all-clouds-monitoring`; `deploy.py down` removes
that namespace.

Open Grafana with the server cloud kubeconfig. For the default GCP-server
config:

```bash
MONITORING_NS=nvflare-all-clouds-monitoring
kubectl --kubeconfig .tmp/kubeconfigs/gcp.yaml \
  -n "$MONITORING_NS" port-forward svc/grafana 3000:3000
```

Log in with `admin` / `admin`, then use the Prometheus datasource in Grafana
Explore. Useful starter queries:

```promql
_system_start_count
_client_register_received_count
_client_heartbeat_received_count
_client_heartbeat_time_taken
{env="all-clouds"}
```

These are FLARE lifecycle metrics, not CPU or memory metrics.

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
