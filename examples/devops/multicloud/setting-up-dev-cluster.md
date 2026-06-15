# Setting Up A Multicloud Dev Cluster

Use this file when creating an NVFlare multicloud dev cluster from the
repository root. Keep private project, cluster, account, and registry values
out of checked-in files. Use a generated config under `examples/devops/.tmp/` for local
edits.

These steps are for testing only. They are not production deployment guidance.

## Inputs To Confirm

- The target config shape is `examples/devops/multicloud/all-clouds.yaml`.
- The default topology is GCP server plus GCP, AWS, and Azure clients.
- The project YAML controls FLARE participants; the deploy YAML maps those
  participant names to clouds and namespaces.
- Real image tags are required for `clouds.<cloud>.prepare.parent.docker_image`.
- Kubeconfig discovery uses active cloud CLI contexts plus env overrides, not
  local YAML override files.

## Create A Temporary Config

```bash
RUN_ID=$(date +%Y%m%d-%H%M%S)
CONFIG=examples/devops/.tmp/multicloud/${RUN_ID}/all-clouds.yaml
mkdir -p "$(dirname "$CONFIG")"
cp examples/devops/multicloud/all-clouds.yaml "$CONFIG"
cp examples/devops/multicloud/project.yml "$(dirname "$CONFIG")/project.yml"
```

Edit `$CONFIG`:

- Set top-level `name` to a unique value, for example `all-clouds-${RUN_ID}`.
- Add `project_file: project.yml`.
- Set each participant `name` in both `$CONFIG` and the copied project YAML to
  a unique value if multiple dev clusters may coexist.
- Set each participant `namespace` to a unique value if multiple dev clusters
  may coexist.
- Replace all placeholder `clouds.<cloud>.prepare.parent.docker_image` values
  with real registry tags.
- Do not add project names, cluster names, subscription IDs, or account IDs to
  the YAML.
- Do not add kubeconfig paths to the YAML; deploy reads
  `examples/devops/.tmp/kubeconfigs/<cloud>.yaml`.

Optional: enable FLARE system monitoring in the temporary config:

```yaml
monitoring:
  enabled: true
```

If monitoring is enabled, use a runtime image that includes `datadog`, for
example an image built with `.[K8S,MONITORING]`.

To select clouds, edit `participants:` in both `$CONFIG` and the copied project
YAML. The config must have exactly one server. For a GCP-only smoke setup:

```yaml
participants:
  - { name: gcp-server,   cloud: gcp, namespace: nvflare-server,   role: server }
  - { name: gcp-client-1, cloud: gcp, namespace: nvflare-client-1, role: client }
```

For a two-cloud setup with a GCP server and AWS client:

```yaml
participants:
  - { name: gcp-server,   cloud: gcp, namespace: nvflare-server,   role: server }
  - { name: aws-client-2, cloud: aws, namespace: nvflare-client-2, role: client }
```

If all clouds can pull the same image, use a YAML anchor so the tag is edited in
one place:

```yaml
x-dev-image: &dev_image registry.example.com/nvflare/nvflare:dev

clouds:
  gcp:
    prepare:
      parent:
        docker_image: *dev_image
```

Repeat `docker_image: *dev_image` for the other clouds. YAML resolves the
anchor before the deploy scripts read the config.

## Preflight

Confirm the deployer has access before starting:

```bash
gcloud auth print-access-token >/dev/null
aws sts get-caller-identity
az account show
kubectl --kubeconfig examples/devops/.tmp/kubeconfigs/gcp.yaml auth can-i create namespaces
helm version
```

For AWS, also confirm the configured region:

```bash
aws configure get region
```

## Build And Push The Image

```bash
python examples/devops/multicloud/build_and_push.py --config "$CONFIG"
```

This reads the image tags from `$CONFIG`, authenticates to recognized
registries, builds `docker/Dockerfile.parent` once from the NVFlare repository
root, tags the same parent image for every used cloud, and pushes all tags.

If monitoring is enabled and the config points to images that still need to be
built, pass a Dockerfile that installs `.[K8S,MONITORING]`:

```bash
python examples/devops/multicloud/build_and_push.py --config "$CONFIG" --dockerfile /path/to/Dockerfile.monitoring
```

Use dry-run first when changing the config:

```bash
python examples/devops/multicloud/build_and_push.py --config "$CONFIG" --dry-run
```

## Fetch Kubeconfigs

Use env vars only when CLI discovery is ambiguous:

```bash
# Optional examples:
# export GCP_CLUSTER=<cluster>
# export GCP_LOCATION=<location>
# export AWS_CLUSTER=<cluster>
# export AWS_REGION=<region>
# export AZURE_CLUSTER=<cluster>
# export AZURE_RESOURCE_GROUP=<resource-group>

python examples/devops/multicloud/fetch_kubeconfigs.py --config "$CONFIG"
```

The script writes `examples/devops/.tmp/kubeconfigs/gcp.yaml`,
`examples/devops/.tmp/kubeconfigs/aws.yaml`, and
`examples/devops/.tmp/kubeconfigs/azure.yaml`.

Preview discovery commands:

```bash
python examples/devops/multicloud/fetch_kubeconfigs.py --config "$CONFIG" --dry-run
```

## Deploy

```bash
python examples/devops/multicloud/deploy.py --config "$CONFIG" up
```

Expected behavior:

- `up` provisions fresh.
- `nvflare deploy prepare` creates each runtime kit.
- Prepared `local/` files are staged as a ConfigMap and `startup/` files are
  staged as a Secret.
- The server is deployed first.
- Clients are deployed in parallel.
- The deterministic cloud IP name comes from the config `name`.
- One namespace and Helm release are created per participant.
- Job pods are created only after a job is submitted.
- If `monitoring.enabled` is true, the monitoring stack is deployed in the
  server cluster.

## Validate

```bash
python examples/devops/multicloud/k8sview.py --config "$CONFIG"
```

If a known sample job is available, run a small smoke job after all
participants are ready. For example, use the prepared `hello-numpy` job if it
exists in the local environment:

```bash
ADMIN_KIT=examples/devops/multicloud/.work/provision/nvflare_multicloud/prod_00/admin@example.com/startup
JOB=/tmp/nvflare/jobs/job_config/hello-numpy
./.venv/bin/nvflare job submit -j "$JOB" --startup-kit "$ADMIN_KIT"
```

Use `nvflare job list` and `nvflare job monitor` to confirm the job completes.

If monitoring is enabled, open Grafana from the server cluster. This default
runbook uses a GCP server; use the server cloud kubeconfig if you changed the
server cloud:

```bash
MONITORING_NS=nvflare-all-clouds-${RUN_ID}-monitoring
kubectl --kubeconfig examples/devops/.tmp/kubeconfigs/gcp.yaml \
  -n "$MONITORING_NS" port-forward svc/grafana 3000:3000
```

Use the Grafana development login and query the Prometheus datasource for
FLARE system metrics such as `_system_start_count`.

## Tear Down

```bash
python examples/devops/multicloud/deploy.py --config "$CONFIG" down
```

`down` tears down clients in parallel first, then the server, then the
deterministic cloud IP.
