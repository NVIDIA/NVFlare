# Submit a CIFAR-10 Recipe Job to Two Kubernetes Clients

> **Development Branch Notice**
> This example uses `set_recipe_meta`, part of the NVFlare 2.9 Recipe API. The
> `requirements.txt` file pins the first upcoming NVFlare release that supports
> it. Until that package is published on PyPI, install NVFlare from this
> repository instead of changing the pin to an older release.

This example submits a PyTorch FedAvg job that trains a small CIFAR-10 image
classifier to an existing production NVFlare system with one server and two
clients. The clients, `site-1` and `site-2`, run in separate Kubernetes
clusters and use `ClientK8sJobLauncher` to create a job pod in their respective
cluster. Each client is assigned a disjoint strided partition of the CIFAR-10
training set; the bounded default sample counts are described below.

```text
workstation (job.py + ProdEnv)
             |
        NVFlare server
          /       \
 site-1 parent   site-2 parent
 K8s cluster A  K8s cluster B
      |               |
 site-1 job pod  site-2 job pod
```

The client-only metadata assumes that the server site uses a process job
launcher. If the server instead uses `ServerK8sJobLauncher`, pass
`--server-image` so its job pod has an image specification too. Launcher
selection itself remains a site runtime policy.

## What the Example Demonstrates

[`job.py`](job.py) uses `set_recipe_meta` twice:

- `JobMetaKey.RESOURCE_SPEC` supplies scheduler-facing resource requirements
  for each client. The CPU demo requests zero GPUs by default; use
  `--site-1-gpus` and `--site-2-gpus` for GPU jobs.
- `JobMetaKey.JOB_LAUNCHER_SPEC` supplies each client's Kubernetes job image,
  Python path, CPU, memory, and ephemeral storage. This enum value is written
  to generated `meta.json` as `launcher_spec`, not `job_launcher_spec`.

The Recipe uses the PyTorch `FedAvgRecipe`, bundles [`client.py`](client.py)
and [`model.py`](model.py), and gives the two clients different data-partition
indices through `per_site_config`.

The client site names are provisioned NVFlare participant identities, not
Kubernetes cluster names. No kubeconfig, namespace, or cluster endpoint belongs
in the job metadata. Each long-running client parent already has the launcher
configuration and credentials for its own cluster. The participant name
`default` cannot be used here because it is reserved for shared launcher
settings.

In a production team, the platform team typically supplies approved image
URLs, Python paths, and resource defaults, often through an organization-owned
wrapper around this script. The data scientist then supplies the training code
and job-specific choices.

The recipe also uses `per_site_config` to target the two client apps explicitly
and pass partition index `0` to `site-1` and index `1` to `site-2`. Resource
and launcher metadata alone do not change a recipe's deployment map.

## Prerequisites

- A running production NVFlare system with one server and two provisioned
  clients whose names match `--site-1-name` and `--site-2-name`.
- Each client configured with `ClientK8sJobLauncher`, including namespace,
  workspace storage, ServiceAccount, and RBAC for creating job pods and startup
  Secrets. NVFlare allows exactly one job launcher per site, so the Kubernetes
  launcher must replace the default process launcher rather than be added
  alongside it; `nvflare deploy prepare` handles this. See the
  [Kubernetes Helm deployment guide](../../../docs/user_guide/admin_guide/deployment/helm_chart.rst).
- An admin startup kit on the workstation running `job.py`.
- Site authorization policies that permit submitted custom code (BYOC), since
  the recipe bundles `client.py` and `model.py` with the job.
- A job image for each cluster. Each image must contain a compatible NVFlare
  installation with the `PT` extra (PyTorch and torchvision), and must be
  pullable from that cluster. Use `--image` when both clusters pull the same
  image, or the two site-specific arguments for different registries.
- Outbound access from each client job pod to download CIFAR-10, or the dataset
  already available at the path passed with `--data-dir` and selected with
  `--no-download-data`.

The server runtime also needs the NVFlare `PT` dependencies for model
persistence. If the server uses `ServerK8sJobLauncher`, its `--server-image`
must contain them.

The [Brev scripted deployment quickstart](../../../docs/user_guide/admin_guide/deployment/brev_scripted_deployment.rst)
creates this exact `server`, `site-1`, and `site-2` topology across three
independent Kubernetes environments.

## Install

From the repository root, install the current source while NVFlare 2.9 is not
yet published:

```bash
python3 -m pip install -e ".[PT]"
```

After the pinned release is available, the example dependencies can instead be
installed normally:

```bash
python3 -m pip install -r examples/advanced/recipe-k8s/requirements.txt
```

The same compatible NVFlare version must be present in the server and client
job images.

## Submit and Monitor the Job

Run from the example directory so the recipe can bundle the relative training
code with paths that are also portable inside the job pods:

```bash
cd examples/advanced/recipe-k8s

python3 job.py \
  --startup-kit /path/to/admin/startup \
  --image registry.example.com/nvflare-pt-job:2.9
```

`ProdEnv` submits the generated job through the admin startup kit. The script
prints the job ID, monitors the job until completion, downloads the result, and
prints the final status and result path. A site-specific image takes precedence
over `--image`; the shared option applies only to clients and never implies a
server image.

By default, each ephemeral client pod downloads CIFAR-10 under `/tmp`, uses a
disjoint strided partition, and caps its partition at 5,000 training and 1,000
test samples for a bounded demo runtime. Set both sample limits to `0` to use
the complete partitions. For a platform-managed dataset volume, use
`--no-download-data --data-dir <mounted-path>`; dataset mounting remains site
runtime policy rather than Recipe metadata.

This example uses at most one device per client, so each `--site-*-gpus` value
must be `0` or `1`. When set to `1`, the client requires CUDA and fails clearly
instead of silently falling back to CPU in a GPU-requesting pod.

For clients with different participant names or resource needs:

```bash
python3 job.py \
  --startup-kit /path/to/admin/startup \
  --site-1-name hospital-a \
  --site-2-name hospital-b \
  --site-1-image registry-a.example.com/nvflare-pt-job:2.9 \
  --site-2-image registry-b.example.com/nvflare-pt-job:2.9 \
  --site-1-gpus 1 \
  --site-2-gpus 1 \
  --site-1-cpu 4 \
  --site-2-cpu 8 \
  --site-1-memory 16Gi \
  --site-2-memory 32Gi \
  --local-epochs 2 \
  --batch-size 128
```

If the server is also prepared with `ServerK8sJobLauncher`, add an image that
its cluster can pull. The server pod defaults to one CPU and `2Gi` of memory;
override those values with `--server-cpu` and `--server-memory`:

```bash
  --server-image registry-server.example.com/nvflare-pt-job:2.9 \
  --server-cpu 4 \
  --server-memory 8Gi
```

### Expected Output and Troubleshooting

The submission process may print monitoring details between these lines:

```text
Job ID: <job-id>
Waiting for job to complete...
Job status: FINISHED:COMPLETED
Result: <downloaded-result-path>
```

While the job is running, run (or ask the platform team to run) `kubectl get
pods` against each client cluster and launcher namespace to see the job pods.
For common failures:

- `ErrImagePull` or `ImagePullBackOff`: verify the registry URL and pull
  Secret, then inspect the pod with `kubectl --context <cluster> -n <namespace>
  describe pod <pod>`.
- No job pod, or a `Forbidden` event: verify the launcher's namespace,
  ServiceAccount, and RBAC, then inspect the long-running client parent logs.
- Submission or scheduling does not progress: verify the admin startup kit and
  username, exact provisioned client names, connected clients, and requested
  GPU availability. A CIFAR-10 download error instead indicates that the job
  pod needs outbound access or a pre-populated `--data-dir`.

## Inspect Without Submitting

`--export` and `--export-dir` are standard Recipe API flags. The
`nvflare.recipe` package consumes them before `job.py` parses its
example-specific arguments, and `recipe.execute()` exports instead of calling
`ProdEnv.deploy()`. They therefore do not appear as arguments in
`define_parser()`.

Recipe export uses the same metadata-building path without connecting to the
NVFlare server. The startup-kit argument must still name an existing directory
because `ProdEnv` validates it during construction:

```bash
python3 job.py \
  --startup-kit /path/to/admin/startup \
  --image registry.example.com/nvflare-pt-job:2.9 \
  --export --export-dir /tmp/nvflare-recipe-job

python3 -m json.tool /tmp/nvflare-recipe-job/cifar10-k8s/meta.json
```

The exported metadata contains flat per-site GPU requests under
`resource_spec` and per-site Kubernetes container settings under
`launcher_spec`.
