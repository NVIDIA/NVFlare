# Job resources and launcher configuration

Job metadata describes required resources and backend runtime settings. Site
configuration chooses the backend and owns its authority: job metadata cannot
select Kubernetes credentials, Docker sockets, Slurm accounts, namespaces or
storage mounts. This contract covers both resident job execution and fresh task
workers. A Client Job (CJ) coordinates one federated job at a site; a Client Parent
(CP) starts CJs. In task mode, each assignment runs in a separate compute worker.

## Resource roles

| Field | Job execution mode | Task execution mode |
| --- | --- | --- |
| `resource_spec` | Resident CJ resources, including compute. | Per-worker compute resources. |
| `cj_resource_spec` | Not accepted for resident client apps. | Independent coordinator resource budget. |
| `launcher_spec` | Backend-specific runtime settings. | Backend-specific compute/runtime settings, separate from CJ resources. |

Portable resources are `num_of_cpus`, `memory` and `num_of_gpus`. CPU requests
are positive integers for compute; coordinator CPU budgets may be positive
fractions. Native Kubernetes CPU quantities can also express fractions. Memory
uses positive integer `Mi`,
`Gi` or `Ti` quantities. GPU count is a nonnegative integer; zero is an explicit
request and must override a GPU site default. A backend must enforce a supplied
requirement or reject it, rather than silently discarding it.

For a task client, this example requests a small CPU coordinator and a GPU worker:

```json
{
  "deploy_map": {"app": ["@ALL"]},
  "resource_spec": {
    "@default": {"num_of_cpus": 2, "memory": "4Gi", "num_of_gpus": 1}
  },
  "cj_resource_spec": {
    "@default": {"num_of_cpus": 0.25, "memory": "512Mi", "num_of_gpus": 0}
  },
  "launcher_spec": {
    "default": {"k8s": {"image": "registry.example/nvflare-training:2.10"}}
  }
}
```

The client app must declare `execution_lifetime: "task"`. In Kubernetes,
`launcher_spec...k8s.image` still selects the CJ image. The task worker image is
fixed separately by trusted `task_execution.launcher.args.image`; a job cannot
select its worker image. Export and submission
derive an empty `cj_resource_spec` entry for each receiving task site, such as
`"@ALL": {}`; authors need not add that marker themselves. Without coordinator
overrides, the empty entry selects trusted site defaults. Compute CPU, memory and
GPU fields never fill missing coordinator fields. Kubernetes coordinator defaults
are 0.25 CPU, 512Mi and zero GPUs; sites may configure `cj_resources` on
ClientK8sJobLauncher. An explicit CJ GPU budget is independent of its workers.
Process has no implicit Kubernetes coordinator defaults or OS resource limiter.

Resident job apps keep their existing resource admission and may coexist with
named task apps at other sites. A coordinator budget for a resident app is rejected.
The Server Job keeps its existing resource semantics.

## Resolution and compatibility

Portable `resource_spec` and `cj_resource_spec` use `@default`. Matching `@ALL`
and named-site entries overlay coordinator defaults, with the named site winning.
Portable resource site overrides are field-based; omitted fields inherit defaults.
`launcher_spec` uses the different reserved key `default`, followed by mode:

```json
{
  "launcher_spec": {
    "default": {"k8s": {"image": "registry.example/training:2.10"}},
    "site-1": {"k8s": {"python_path": "/usr/local/bin/python"}}
  }
}
```

Resolution shallow-merges `launcher_spec.default[mode]` with
`launcher_spec[site][mode]`. The site block wins; nested dictionaries are not
recursively merged. An installed launcher reads only its own mode's settings.
When a matching launcher block is absent, the existing nested
`resource_spec[site][mode]` compatibility fallback remains available. New jobs
should use portable resource fields and `launcher_spec` rather than mixing images
or runtime flags into `resource_spec`.

Submission preparation preserves the reserved `@default` and `@ALL` keys
through JSON/YAML-to-ConfigTree conversion, CLI overrides and serialization.
Saving uses the full indexed configuration root, including when the first field
is a nested resource block. `default` and `@default` remain distinct keys.

Portable and native declarations of the same resource are rejected as conflicts.
For example, do not combine portable `num_of_cpus` with Kubernetes `cpu` or
`cpu_request`, nor portable memory with Kubernetes `memory` or `memory_request`.
Legacy Process GPU requirements and a native GPU count must remain consistent.
Explicit zero GPU counts are preserved when resolving worker defaults.

Native settings remain backend-specific. Kubernetes supports CPU/memory request
and limit quantities, image, Python path, pending timeout and ephemeral storage.
Docker has its own image, entrypoint and shared-memory settings. Slurm has job
topology, CPU/memory/GPU quantities and time settings. Each backend's validator
defines its allowed fields. Site routing, accounts, setup commands and arbitrary
runtime authority remain outside job metadata. Job-supplied images retain the
existing bring-your-own-code authorization requirement.

## Admission and enforcement

Submission derives coordinator entries from the client app declaration, without
resolving unrelated application arguments on the server. Client-only environment
substitutions and HOCON includes remain client concerns. Task lifetime should be
declared literally in the submitted root client config. Before launch, CP checks
the deployed resolved mode; CJ checks it again before constructing application
components. A task admission marker cannot excuse a resident CJ from compute
resource reservation if local configuration resolves to job lifetime.

The scheduler's resource-manager request is obtained through
`get_resource_manager_spec`. For a task site this uses only the coordinator
budget; `get_compute_resource_manager_spec` resolves compute admission fields
separately. Native worker launchers translate those compute fields into their
runtime's resource requests. CPU and memory are enforced by a capable backend,
rather than implicitly reserved in the legacy NVFlare GPU resource manager.

Custom resources, licenses and per-GPU memory requirements need explicit backend
enforcement. The current Kubernetes and Process task paths reject unsupported
compute resource-manager claims. Process validates its compute CPU/memory and
custom resource claims before starting a normalized task CJ and again before
executing a task, so unsupported sites fail once at launch. Removing the compute GPU from whole-job admission
must never silently drop these other requirements. Unsupported Process CPU/memory
limits or coordinator GPU requirements likewise fail explicitly.

With one available GPU, two Kubernetes task CJs can remain resident while their
worker Pods use that GPU successively. Kubernetes owns scheduling and fairness;
NVFlare does not guarantee round-robin alternation. A queued worker holds its
attempt identity until settlement, cancellation or a configured deadline. Worker
resource queue waiting and container startup use separate timeouts. A resident
job continues to use the K8s job launcher's existing pending timeout.

## Related contracts

- [Task execution architecture and lifecycle](task_worker_lifecycle.md): compute,
  coordination, settlement, checkpoint promotion and cancellation.
- [JobLauncher and JobHandle](job_launcher_and_job_handle.md): resident runtime
  creation, early handle publication and job termination.
- [Kubernetes operator guide](../user_guide/admin_guide/deployment/kubernetes_task_launcher.rst):
  site configuration, two-PVC storage, namespace choices and qualification checks.
