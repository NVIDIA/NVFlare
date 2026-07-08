# Submit a Recipe Job to Two Kubernetes Clients

> **Development Branch Notice**
> This example uses `set_recipe_meta`, part of the NVFlare 2.9 Recipe API. The
> `requirements.txt` file pins the first upcoming NVFlare release that supports
> it. Until that package is published on PyPI, install NVFlare from this
> repository instead of changing the pin to an older release.

This example submits one NumPy FedAvg job to an existing production NVFlare
system with one server and two clients. The clients, `site-1` and `site-2`, run
in separate Kubernetes clusters and use `ClientK8sJobLauncher` to create a job
pod in their respective cluster.

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

The client site names are provisioned NVFlare participant identities, not
Kubernetes cluster names. No kubeconfig, namespace, or cluster endpoint belongs
in the job metadata. Each long-running client parent already has the launcher
configuration and credentials for its own cluster. The participant name
`default` cannot be used here because it is reserved for shared launcher
settings.

The recipe also passes
`per_site_config={"site-1": {}, "site-2": {}}`. Those entries target the two
client apps explicitly. Resource and launcher metadata alone do not change a
recipe's deployment map.

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
  the recipe bundles `client.py` with the job.
- A job image for each cluster. Each image must contain a compatible NVFlare
  installation and NumPy, and must be pullable from that cluster. The two image
  arguments may point to different registries. The recipe bundles `client.py`
  into the submitted job, so the training script need not be baked into the
  images.

The [Brev scripted deployment quickstart](../../../docs/user_guide/admin_guide/deployment/brev_scripted_deployment.rst)
creates this exact `server`, `site-1`, and `site-2` topology across three
independent Kubernetes environments.

## Install

From the repository root, install the current source while NVFlare 2.9 is not
yet published:

```bash
python3 -m pip install -e .
```

After the pinned release is available, the example dependencies can instead be
installed normally:

```bash
python3 -m pip install -r examples/advanced/recipe-k8s/requirements.txt
```

The same compatible NVFlare version must be present in the server and client
job images.

## Submit and Monitor the Job

Run from the example directory so the recipe can bundle the relative
`client.py` training script with a path that is also portable inside the job
pods:

```bash
cd examples/advanced/recipe-k8s

python3 job.py \
  --startup-kit /path/to/admin/startup \
  --site-1-image registry-a.example.com/nvflare-job:2.9 \
  --site-2-image registry-b.example.com/nvflare-job:2.9
```

`ProdEnv` submits the generated job through the admin startup kit. The script
prints the job ID, monitors the job until completion, downloads the result, and
prints the final status and result path.

For clients with different participant names or resource needs:

```bash
python3 job.py \
  --startup-kit /path/to/admin/startup \
  --site-1-name hospital-a \
  --site-2-name hospital-b \
  --site-1-image registry-a.example.com/nvflare-job:2.9 \
  --site-2-image registry-b.example.com/nvflare-job:2.9 \
  --site-1-gpus 1 \
  --site-2-gpus 2 \
  --site-1-cpu 4 \
  --site-2-cpu 8 \
  --site-1-memory 16Gi \
  --site-2-memory 32Gi
```

If the server is also prepared with `ServerK8sJobLauncher`, add an image that
its cluster can pull. The server pod defaults to one CPU and `2Gi` of memory;
override those values with `--server-cpu` and `--server-memory`:

```bash
  --server-image registry-server.example.com/nvflare-job:2.9 \
  --server-cpu 4 \
  --server-memory 8Gi
```

## Inspect Without Submitting

Recipe export uses the same metadata-building path without connecting to the
NVFlare server. The startup-kit argument must still name an existing directory
because `ProdEnv` validates it during construction:

`--export` and `--export-dir` are standard Recipe API flags. The
`nvflare.recipe` package consumes them before `job.py` parses its
example-specific arguments, and `recipe.execute()` exports instead of calling
`ProdEnv.deploy()`. They therefore do not appear as arguments in
`define_parser()`.

```bash
python3 job.py \
  --startup-kit /path/to/admin/startup \
  --site-1-image registry-a.example.com/nvflare-job:2.9 \
  --site-2-image registry-b.example.com/nvflare-job:2.9 \
  --export --export-dir /tmp/nvflare-recipe-job

python3 -m json.tool /tmp/nvflare-recipe-job/hello-numpy-k8s/meta.json
```

The exported metadata contains flat per-site GPU requests under
`resource_spec` and per-site Kubernetes container settings under
`launcher_spec`.
