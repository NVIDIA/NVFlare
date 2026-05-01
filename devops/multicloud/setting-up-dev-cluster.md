# Setting Up A Multicloud Dev Cluster

Use this file when prompted to create an NVFlare multicloud dev cluster. Keep all private project, cluster, and registry values out of checked-in files. Use a generated config under `.tmp/` for local edits.

## Inputs To Confirm

- The target config shape is `devops/multicloud/all-clouds.yaml`.
- The default topology is GCP server plus GCP, AWS, and Azure clients.
- The `participants:` section controls which clouds and sites are deployed.
- Real image tags are required for `clouds.<cloud>.prepare.parent.docker_image`.
- Kubeconfig discovery uses active cloud CLI contexts plus env overrides, not local YAML override files.

## Create A Temporary Config

```bash
RUN_ID=$(date +%Y%m%d-%H%M%S)
CONFIG=.tmp/multicloud/${RUN_ID}/all-clouds.yaml
mkdir -p "$(dirname "$CONFIG")"
cp devops/multicloud/all-clouds.yaml "$CONFIG"
```

Edit `$CONFIG`:

- Set top-level `name` to a unique value, for example `all-clouds-${RUN_ID}`.
- Set each participant `name` to a unique value if multiple dev clusters may coexist.
- Set each participant `namespace` to a unique value if multiple dev clusters may coexist.
- Replace all placeholder `clouds.<cloud>.prepare.parent.docker_image` values with real registry tags.
- Do not add project names, cluster names, subscription IDs, or account IDs to the YAML.
- Do not add kubeconfig paths to the YAML; deploy reads `.tmp/kubeconfigs/<cloud>.yaml`.

To select clouds, edit `participants:`. The config must have exactly one server.
For a GCP-only smoke setup:

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
x-dev-image: &dev_image pcnudde03194/nvflare-dev-pcnudde:pcnudde-dev

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
kubectl --kubeconfig .tmp/kubeconfigs/gcp.yaml auth can-i create namespaces
helm version
```

For AWS, also confirm the configured region:

```bash
aws configure get region
```

## Build And Push The Image

```bash
python devops/multicloud/build_and_push.py --config "$CONFIG"
```

This reads the image tags from `$CONFIG`, authenticates to recognized registries, builds `docker/Dockerfile` once, tags the same image for every used cloud, and pushes all tags.

Use dry-run first when changing the config:

```bash
python devops/multicloud/build_and_push.py --config "$CONFIG" --dry-run
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

python devops/multicloud/fetch_kubeconfigs.py --config "$CONFIG"
```

The script writes `.tmp/kubeconfigs/gcp.yaml`, `.tmp/kubeconfigs/aws.yaml`, and `.tmp/kubeconfigs/azure.yaml`.

Preview discovery commands:

```bash
python devops/multicloud/fetch_kubeconfigs.py --config "$CONFIG" --dry-run
```

## Deploy

```bash
python devops/multicloud/deploy.py --config "$CONFIG" up
```

Expected behavior:

- `up` provisions fresh.
- `nvflare deploy prepare` creates each runtime kit.
- Startup/local kit files are staged into the workspace PVC.
- The server is deployed first.
- Clients are deployed in parallel.
- The deterministic cloud IP name comes from the config `name`.
- One namespace and Helm release are created per participant.
- Job pods are created only after a job is submitted.

## Validate

```bash
python devops/multicloud/k8sview.py --config "$CONFIG"
```

If a known sample job is available, run a small smoke job after all participants are ready. For example, use the prepared `hello-numpy` job if it exists in the local environment:

```bash
ADMIN_KIT=devops/multicloud/.work/provision/nvflare_multicloud/prod_00/admin@nvidia.com/startup
JOB=/tmp/nvflare/jobs/job_config/hello-numpy
./.venv/bin/nvflare job submit -j "$JOB" --startup-kit "$ADMIN_KIT"
```

Use `nvflare job list` and `nvflare job monitor` to confirm the job completes.

## Tear Down

```bash
python devops/multicloud/deploy.py --config "$CONFIG" down
```

`down` tears down clients in parallel first, then the server, then the deterministic cloud IP.
