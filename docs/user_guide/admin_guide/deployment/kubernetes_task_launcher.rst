Kubernetes task execution: deployment and operations
====================================================

This guide configures a Kubernetes site to run each assignment in a fresh CPU or
GPU worker Pod. The Client Parent (CP) starts and monitors Client Jobs (CJs).
Each task-mode CJ keeps federation communication, filters and result submission
resident; CP creates and cleans up its compute workers. Existing job mode keeps
compute inside CJ and remains the default.

The site chooses one matching Kubernetes CP/CJ/worker backend and owns images,
accounts, namespace and storage. Jobs supply separate resource budgets:

* ``resource_spec`` and native compute CPU/memory/GPU fields apply to workers in
  task mode and to CJ in job mode.
* ``cj_resource_spec`` applies only to task-mode coordinators. Site defaults are
  0.25 CPU, 512Mi and zero GPUs; explicit coordinator GPUs are independent.
* The worker image comes from trusted ``task_execution`` configuration. A job's
  Kubernetes image setting selects its CJ image.

Workers use their CJ's configured namespace. CP can share that namespace or use
a separate control namespace with explicit permissions and compatible storage.
There is no automatic namespace per job. The recommended storage profile uses a
private CP workspace PVC and a separate jobs PVC, with one job/site subPath per
CJ and its workers. Task mode requires durable storage; resident jobs stay
ephemeral unless ``persist_resident_workspaces`` is enabled explicitly.

CJ and workers have no Kubernetes API token. CJ sends bounded commands to CP
using a per-job capability token; CP fixes the Pod manifest and admits one
unsettled attempt per job. Pod deletion alone does not prove settlement: CP uses
UID preconditions, a finalizer and container/sandbox shutdown evidence. API errors,
ambiguous creates or missing evidence retain ownership and prevent unsafe retry.
CP continues cleanup after CJ exits; CP restart adoption is not provided.

The `task execution design <../../../design/task_worker_lifecycle.md>`_ explains
the architecture and lifecycle. The configuration, storage assumptions, operational
limits and qualification steps needed to deploy this profile are included below.

Control-plane limits
--------------------

The normal job handle observes CJ lifecycle. A separate ten-second CJ observer
backs it up without using the shorter worker polling interval. CJ shutdown makes
bounded cleanup calls; CP keeps responsibility for unresolved workers.

CP retains attempt identities to fence delayed requests and lost replies. The
trusted task launcher ``max_attempts`` setting (positive integer, default 10,000)
bounds all retained identities per job, including locally rejected launches.
Unknown cancel-before-launch identities also have a limit of 128. Exceeding either
quota closes admission and cancels owned execution without evicting old fences.
Existing poll/cancel operations and CP cleanup remain available. Size the history
quota for the expected number of assignments; it does not provide automatic retry.

Do not clear an unresolved finalizer solely to remove a stuck Pod. Verify runtime
shutdown before operator reconciliation after CP loss, node loss or missing
settlement evidence. No timeout or subsequent 404 substitutes for that proof.

Site prerequisites
------------------

The recommended site profile uses two pre-provisioned claims: a private CP
workspace PVC and a separate jobs PVC shared by this site's CJs and task workers.
CP mounts the jobs PVC once at ``job_workspace_pvc_root``; each CJ/worker mounts
only its own job/site subtree. Never mount CP's private workspace claim in CJ or
worker Pods. This keeps the PVC count fixed as jobs and attempts increase.

Use an existing namespace and preapproved worker image. For pods on
different nodes the jobs volume must support the required concurrent access
(commonly RWX); an access-mode declaration alone does not establish filesystem
semantics. All bootstrap paths are absolute paths in the same explicitly
configured mount in both pods. Configure CP's ``job_workspace_pvc_root`` as its
explicit mount of ``job_workspace_pvc``. CP creates
``nvflare-tasks/<sha256(job,site)>`` with mode 0700 and verifies its numeric owner;
CJ and workers mount that subtree at the CJ's ``workspace_mount_path`` / worker's
``workspace_root``. Existing subtrees fail rather than being reset or adopted.
CP, CJ and workers must share the configured nonroot UID; configure matching
numeric groups and volume permissions. No chown or privileged helper is used.
The storage driver must permit these directory and fsGroup operations.

The shared UID is a site trust assumption: mode 0700 does not separate jobs
owned by that same UID. Cross-job filesystem isolation depends on kubelet's
subPath mount boundary and the storage driver. This profile does not allocate
per-job filesystem identities or claims. Keep Kubernetes nodes and storage
components maintained, and grant applications no path to the full jobs claim.
The persistent profile rejects duplicate jobs-claim volumes and extra workspace
mounts. Trusted Pod templates retain security settings such as seccomp profiles,
supplemental groups and a read-only container root filesystem, while the launcher
enforces nonroot IDs, no privilege escalation and dropped capabilities. The
persistent CJ profile requires one container, no init/ephemeral containers, and
no host PID/IPC/network sharing; a template with sidecars is rejected.

The resident workspace download stages ``local/`` and this job's files into the
subtree. The worker mounts ``local/`` read-only from that same subtree, protecting
canonical site policy against task writes. Both CJ and worker mount the existing
site startup Secret read-only at ``startup/``; the resident workspace download
excludes startup. The Secret construction already withholds site private keys.
Set optional ``startup_secret_name`` on ``ClientK8sJobLauncher`` (and the server
job launcher when sharing its namespace) to a DNS subdomain of at most 253
characters. CJ and task workers use that exact Secret; do not configure a second
name on the task launcher. The default remains
``nvflare-startup-<normalized-participant>``. Names must be unique per federation
and participant within a namespace; for example, use ``federation-a-site-1`` and
``federation-b-site-1`` when two federations both have ``site-1``.

Startup Secrets carry the label ``nvflare.nvidia.com/startup-federation``, a
base32-encoded SHA-256 fingerprint of ``rootCA.pem``. CP refuses to replace a
Secret whose federation identity differs, and pins replacement to the version
it read. Identical kits are reused, and concurrent updates re-read and revalidate
ownership up to three times. Legacy unlabeled Secrets are upgraded only after
their existing CA matches. For unsecured kits without a CA, identity is a digest of the public kit
payload. Trust-root rotation or an unsecured kit change requires a new Secret
name, or an intentional operator replacement after stopping existing consumers.
A conflict fails launch with a request to configure a unique name; it never
automatically deletes another kit. CP needs ``get`` on ``secrets`` to check
ownership when a name already exists.

Task pods receive no CJ bootstrap environment, Kubernetes API token or kubeconfig.
Authorized workers still share the job's filesystem identity and can read files
already present there; this profile does not create a separate filesystem trust
boundary for job credentials. Do not distribute a cluster-admin kubeconfig.

Task workers require a direct Cell connection to their CJ for live metrics. CP
injects ``NVFLARE_K8S_CJ_POD_IP`` from Kubernetes ``status.podIP``; the task launcher
advertises it with a dynamically selected CJ listener port. A job/template value
cannot override this downward-API field. Allow worker-to-CJ traffic on that port
in cluster routing and NetworkPolicy. This CellNet path requires IPv4 Pod
addresses; missing, loopback or unsupported addresses fail CJ startup.

The listener uses the site's internal scheme and connection security, including
mTLS when configured. The worker bootstrap supplies the per-job identity and
both client/server certificate-key pairs with their CA. Those files must be
readable at the same absolute paths in CJ and worker mounts; the shared jobs
workspace and startup Secret provide this layout. No site private key is needed.
Metrics are delivered while compute runs, with a 30-second reply timeout; a
rejected metric fails its attempt. Tracking-receiver exceptions are logged as in
resident job mode. Analytics no longer wait for settlement or replay from disk.

The CP JobLauncher needs its documented Pod/event/Secret permissions plus
``patch`` on ``pods`` and ``get`` on ``pods/log`` in the configured CJ/worker
namespace. The client Helm Role includes these permissions. CP may run in a
different namespace if the operator grants its account access to the target
namespace and supplies connectivity and storage. This does not create a new
namespace per job; all CJs and workers use the launcher's configured namespace.

CJ and worker service accounts need no Kubernetes API permissions. No CJ launch
Role or RoleBinding is needed. The task path creates no namespaces, claims,
accounts or Secrets; CP retains the resident launcher's existing startup and
per-job bootstrap Secrets. Pod construction is enforced by trusted CP code,
independently of admission policy. Do not mount a kubeconfig or an explicit
service-account token into CJ or worker Pods.

Configure worker ``image``, ``python_path``, ``environment``, ``run_as_user`` and
``run_as_group`` in trusted site policy. Worker resources default to the site's
``cpu_cores``, ``memory_mb`` and ``gpu_count``. The bound job overrides these with
its receiving-site portable resources or native ``launcher_spec[site]["k8s"]``
CPU/memory/GPU fields, using the existing default/site and legacy resolution
rules. Native CPU and memory quantities and distinct requests/limits are
preserved. Explicit ``num_of_gpus=0`` overrides a GPU site default. Nonzero GPU
counts become equal ``nvidia.com/gpu`` requests and limits; manual GPU visibility
settings remain forbidden. Namespace quotas/admission enforce site policy.
Private-image pull access belongs to the existing worker service account.

CJ and compute resources have separate lifetimes. In job mode the existing
``resource_spec`` and native launcher resource fields describe CJ, which runs
the computation. In task mode those same fields describe each compute worker;
``cj_resource_spec`` describes the persistent coordinator independently. A GPU
worker still needs CPU and memory in addition to its GPU. Native request/limit
fields keep their existing conflict rules and apply only to compute in task mode.

``cj_resource_spec`` is keyed by client site, with an optional ``@default``
block. It accepts ``num_of_cpus`` (including positive fractional shares),
``memory`` (Mi/Gi/Ti), ``num_of_gpus`` and ``mem_per_gpu_in_GiB``. Overrides are
merged per field. Missing CJ fields use the site's ``ClientK8sJobLauncher``
``cj_resources`` defaults: 0.25 CPU, 512Mi memory and zero GPUs. Configure these
for the site's model sizes, filters and result processing. CJ requests and
limits are equal in this profile. A positive explicit CJ GPU budget remains
independent of worker GPUs; zero-GPU CJs mask GPU visibility.

Export and submission derive task sites from the actual client configurations,
including mixed job/task deployments and ``@ALL``. They add an empty CJ entry
when no override was supplied; that entry selects site defaults. A supplied CJ
budget cannot mark a resident client as task based. Server admission reserves
only the CJ budget for those sites, leaving compute GPUs available to workers.
Kubernetes admits each worker's resource request and releases it only after
confirmed shutdown; the finalizer settlement rules are unchanged. CPU/memory
are enforced by Pod requests/limits, rather than by the FLARE GPU manager.
Kubernetes workers reject custom FLARE resource-manager requirements and
positive per-GPU memory reservations, which this Pod profile cannot enforce;
these requirements are never silently dropped when compute leaves CJ.
The separate Process backend retains its own capability and static GPU-device
assignment policy; Kubernetes defaults do not impose Process resource limits.

Queue waiting and startup use separate site-owned limits. ``queue_timeout``
defaults to ``null``: an unassigned task Pod may wait for CPU, memory or GPU
capacity while its CJ remains alive. A finite nonnegative value bounds that
queue wait; ``0`` fails immediately if the Pod is still unassigned. After a node
assignment is first observed, ``startup_timeout`` defaults to 120 seconds and
bounds startup until the Pod runs. Queue time does not consume that startup
budget. Invalid scheduling constraints and known image/container failures still
fail promptly. A timeout follows the existing cancellation and confirmed
settlement rules; it never authorizes bypassing the finalizer.

The common ``worker_timeout`` remains an overall physical-attempt budget,
including queue waiting, startup and execution. Give it enough time for expected
contention, or set it to ``null`` and rely on the job/task deadlines. Explicit
abort remains effective in every phase. Configure settlement bounds appropriate
to the termination grace period and storage driver. These task settings do not
change the job launcher's existing ``pending_timeout`` policy. A created CJ's
handle is handed to CP before the launcher waits for Running, so aborting an
unscheduled CJ deletes its Pod without waiting for that resource timeout.

For example, two CPU CJs can keep their jobs alive while their task Pods request
one GPU each from a cluster with one allocatable GPU. One worker runs and the
other queues; the GPU becomes reusable between task attempts, before CJ result
upload finishes. Kubernetes selects the next eligible Pod; this launcher does
not enforce alternating jobs. CPU/memory requests must also fit the CJs, parent
and worker. The two-job GPU-contention scenario requires cluster qualification.

Both portable ``resource_spec`` GPU requests and native ``launcher_spec`` GPU
requests are supported for workers. Do not specify both for the same resource.
Resubmit older persisted task jobs with portable GPUs so submission derives
their CJ admission entries. Old metadata that would still reserve compute GPUs
for the whole job is rejected before a CJ/task Pod is launched.

Configure the generated client chart's existing ``persistence.workspace`` claim
for private CP state, then enable its optional ``persistence.jobs`` mount before
installing/upgrading the CP Deployment. For site-1, merge these settings into
``helm_chart/values.yaml`` after ``nvflare deploy prepare``::

   persistence:
     workspace:
       claimName: site-1-cp
     jobs:
       claimName: site-1-jobs
       mountPath: /mnt/nvflare-jobs
   securityContext:
     runAsUser: 1000
     runAsGroup: 1000
     runAsNonRoot: true
     fsGroup: 1000

Keep the generated workspace ``volumeName`` and ``mountPath`` entries. Both
claims must already exist and permit the configured nonroot identity. The chart
rejects identical CP/jobs claim names and overlapping CP/jobs mount paths. It
does not create PVCs. Leave ``persistence.jobs.claimName`` null for deployments
without this optional storage profile. Give each client site its own pair of
claims. A PVC is namespaced: CP and its CJs/workers must be able to reference the
same jobs claim in the same namespace; a cross-namespace layout requires a
separately configured shared-filesystem arrangement.

Keep the existing ``ClientK8sJobLauncher`` component and add the job storage and
identity arguments shown below. Configure ``K8sTaskLauncher``
under ``task_execution.launcher`` and preserve the site's other components.
After ``nvflare deploy prepare``, put this task profile in the canonical
``local/resources.json`` override. Keep the generated
``local/resources.json.default`` when using the existing ``deploy k8 stage``
command, which identifies the prepared kit through that generated default.
These site-resource and chart-value settings are not additional ``prepare``
YAML keys. The launcher ``job_workspace_pvc`` and worker ``workspace_pvc`` must
both match ``persistence.jobs.claimName``; ``job_workspace_pvc_root`` must match
``persistence.jobs.mountPath``. Apply the override after preparation, then stage
both local files::

   {
     "components": [{
       "id": "k8s_job_launcher",
       "path": "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
       "args": {
         "config_file_path": null,
         "namespace": "site",
         "startup_secret_name": "federation-a-site-1",
         "job_workspace_pvc": "site-1-jobs",
         "job_workspace_pvc_root": "/mnt/nvflare-jobs",
         "persist_resident_workspaces": true,
         "workspace_mount_path": "/var/tmp/nvflare/workspace",
         "run_as_user": 1000,
         "run_as_group": 1000,
         "cj_resources": {"num_of_cpus": 0.25, "memory": "512Mi", "num_of_gpus": 0}
       }
     }],
     "task_execution": {
       "launcher": {
         "path": "nvflare.app_opt.task_launcher.k8s_launcher.K8sTaskLauncher",
         "args": {
           "namespace": "site",
           "image": "site-approved-worker:reviewed-source",
           "workspace_root": "/var/tmp/nvflare/workspace",
           "workspace_pvc": "site-1-jobs",
           "worker_service_account": "task-no-api",
           "run_as_user": 1000,
           "run_as_group": 1000,
           "cpu_cores": 1,
           "memory_mb": 512,
           "queue_timeout": null,
           "startup_timeout": 120
         }
       },
       "worker_timeout": 1800,
       "settlement_timeout": 60
     }
   }

The job retains its existing ``execution_lifetime="task"`` application flag.
Its metadata specifies the CJ image and independent resource budgets, for example::

   {
     "resource_spec": {
       "@default": {"num_of_cpus": 2, "memory": "4Gi", "num_of_gpus": 1}
     },
     "cj_resource_spec": {
       "@default": {"num_of_cpus": 0.5, "memory": "1Gi", "num_of_gpus": 0}
     },
     "launcher_spec": {
       "default": {"k8s": {"image": "site-approved-cj:reviewed-source"}}
     }
   }

In task mode this requests 0.5 CPU and 1Gi for CJ, plus 2 CPUs, 4Gi and one GPU
while a compute worker runs. Use ``num_of_gpus=0`` in ``resource_spec`` for a CPU
worker. Native ``launcher_spec`` CPU/memory/GPU fields remain an alternative to
portable compute fields, including distinct native requests and limits. Jobs
can set the new field through ``meta_props`` or
``set_recipe_meta(recipe, JobMetaKey.CJ_RESOURCE_SPEC, ...)``. The same metadata
without a task-lifetime client must omit ``cj_resource_spec``; job mode keeps
its existing resource semantics. Images must contain the same reviewed
framework source and the required application dependencies.

The cluster must expose kubelet ``PodReadyToStartContainers`` teardown evidence
and terminal container statuses for this profile. Pods lacking that evidence
remain unresolved. Disable admission-injected long-lived containers for the
qualified worker profile; the handle nevertheless checks every admitted main,
init and ephemeral container before cleanup. No force deletion, automatic
attempt restart, Job controller or TTL deletion is used. See Kubernetes
`Pod lifecycle <https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/>`_
and `finalizers <https://kubernetes.io/docs/concepts/overview/working-with-objects/finalizers/>`_.

Deployment readiness runbook
----------------------------

Complete these checks before submitting the first application job. Keep their
commands, rendered manifests, versions, image digests and results together as
one setup record. A running parent Pod alone does not establish readiness for
launching CJ and task Pods. These checks establish setup readiness; the full
application and resource-release qualification below remains separate.
Set the examples' ``NVFLARE_TEST_*`` variables to the actual namespace, parent
Pod, CP service account, workspace root, site name and worker image digest for
the participant being checked. Use the intended Kubernetes context throughout.

1. **Record the environment and exact source.** Confirm the node is ready and
   record the Kubernetes/container-runtime, NVIDIA driver, Container Toolkit,
   device-plugin and PyTorch/CUDA versions. Build the site and worker images
   from the same reviewed source, publish them, and record their registry
   digests. On a CPU-only host, GPU checks are inapplicable; CPU execution and
   the remaining readiness checks still apply. Useful inventory commands are::

      kubectl version -o yaml
      kubectl get nodes -o wide
      kubectl get nodes -o json

   For a GPU profile, the node's ``status.allocatable`` must advertise the
   requested ``nvidia.com/gpu`` capacity. Driver visibility on the host does
   not establish device-plugin registration or CUDA access inside a Pod.

2. **Verify the runtime used by the generated worker.** This adapter does not
   set ``runtimeClassName``. GPU support must therefore work with the Pod's
   actual admitted runtime, including any site admission changes. A smoke Pod
   that explicitly sets ``runtimeClassName: nvidia`` does not test an unmodified
   task Pod that omits it.

   In a K3s environment using the NVIDIA container runtime, the operator can
   configure the relevant GPU nodes' default runtime using ``default-runtime:
   nvidia`` in K3s configuration. Follow the platform's
   `NVIDIA runtime instructions <https://docs.k3s.io/advanced#nvidia-container-runtime>`_
   and retain the existing node configuration. After any runtime change,
   recheck node readiness and run a fresh Pod with the task profile's actual
   runtime selection. Record the admitted Pod specification. The CUDA check
   must allocate GPU memory, execute a kernel and synchronize successfully.
   Also check that the CPU profile observes zero CUDA devices in the same
   GPU-capable image.

3. **Prove image pulls through the Kubernetes runtime.** The shared manifest
   builder sets ``imagePullPolicy: Always`` for generated task Pods and default
   CJ/SJ Pods. Building an image with Docker or importing it into containerd
   does not establish that a fresh generated Pod can pull it. Pin the registry
   digest in the job and site worker configuration; verify the exact reference
   from every eligible node, then launch a fresh smoke Pod using ``Always``.
   A smoke using ``Never`` or ``IfNotPresent`` does not check this path. See
   Kubernetes `image pull policy <https://kubernetes.io/docs/concepts/containers/images/#image-pull-policy>`_.

   For K3s, a node-side diagnostic is::

      sudo k3s crictl pull "$NVFLARE_TEST_IMAGE"

   Set ``NVFLARE_TEST_IMAGE`` to the exact registry digest. Configure any
   required node-side registry access using the platform's
   `private-registry configuration <https://docs.k3s.io/installation/private-registry>`_.
   Validate worker service-account pull references as well as parent/CJ pull
   references. A loopback registry is a single-node lab arrangement: on another
   node, ``127.0.0.1`` names that other node. A multi-node profile needs image
   access from every node on which its Pods can schedule.

   **Check all three image configuration locations.** Parent images come from
   the Deployment/Helm configuration. Job metadata
   ``launcher_spec[site]["k8s"]["image"]`` selects the SJ/CJ image. The worker
   image comes independently from canonical site
   ``task_execution.launcher.args.image``. Updating one location does not
   update the others. Record the intended digest for each role. When a lab
   uses one image for all roles, generate all three settings from that one
   image input. Include reset/redeployment templates in that update; a reset
   must not restore a stale local tag. Inspect the actual parent references::

      kubectl -n "$NVFLARE_TEST_NAMESPACE" get deployments \
        -o 'custom-columns=NAME:.metadata.name,IMAGES:.spec.template.spec.containers[*].image'

   Compare the job metadata and actual mounted policy in every CP and fresh CJ;
   editing a host-side file alone does not update an already mounted kit.
   This diagnostic prints the selected policy file and requires its worker
   image to match ``NVFLARE_TEST_IMAGE``::

      kubectl -n "$NVFLARE_TEST_NAMESPACE" exec -i "$NVFLARE_TEST_CP_POD" -- \
        python - "$NVFLARE_TEST_WORKSPACE" "$NVFLARE_TEST_SITE" "$NVFLARE_TEST_IMAGE" <<'PY'
      import sys
      from nvflare.apis.workspace import Workspace
      from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
      workspace = Workspace(sys.argv[1], site_name=sys.argv[2])
      policy = TaskExecutionPolicy.from_workspace(workspace)
      print(workspace.get_resources_file_path(), policy.launcher.image, sep="\n")
      assert policy.launcher.image == sys.argv[3]
      PY

   Set the workspace and site variables to that participant's actual values;
   run the same check in the new CJ. Verify the active ConfigMap contents and
   rollout/mount behavior when replacing site policy. A successful SJ/CJ pull
   does not establish that task Pods use the intended image.

4. **Prepare Pod-reachable addresses and verify registration.** Start from
   Kubernetes-prepared runtime kits, for example::

      nvflare deploy prepare ./site-1 --output ./site-1-k8s --config k8s-site1.yaml

   Supply the site's normal ``runtime: k8s`` configuration file. Preparation
   configures internal parent communication for Kubernetes; inspect the
   federation startup addresses separately. A POC kit's ``localhost`` server
   address does not reach a server in another Pod. Use the actual Service
   address reachable from each participant's namespace, preserving the
   provisioned TLS identity and verification settings. Kubernetes documents
   `Service DNS names <https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/>`_.

   For example, from a client parent Pod in the same namespace as the server
   Service, verify name resolution and the federation port::

      kubectl -n "$NVFLARE_TEST_NAMESPACE" exec "$NVFLARE_TEST_CP_POD" -- \
        python -c 'import socket; c=socket.create_connection(("nvflare-server",8002),timeout=5); print(c.getpeername()); c.close()'

   Substitute the configured Service and port. TCP connectivity is only a
   diagnostic: require successful authenticated client registration and an
   admin connection through its intended endpoint before job submission. A
   host-side admin may use a lab port-forward while in-cluster clients use
   Service DNS; those addresses serve different network locations.

5. **Verify storage, ownership and launch permissions.** Apply the matching
   task launcher pair shown above to each client's ``local/resources.json``
   override before staging the prepared kit. Preserve the generated default
   file and verify which file the running workspace selects. Confirm that the
   CP actually mounts its
   declared task PVC and that CP/CJ/workers use matching numeric identities.
   Exercise ``check_posix_storage`` on that mount as the configured nonroot
   user, then validate the CJ and worker mounts and the storage semantics below.
   Verify the existing service accounts' required access, including::

      kubectl -n "$NVFLARE_TEST_NAMESPACE" auth can-i create pods \
        --as="system:serviceaccount:$NVFLARE_TEST_NAMESPACE:$NVFLARE_TEST_CP_ACCOUNT"
      kubectl -n "$NVFLARE_TEST_NAMESPACE" auth can-i get pods/log \
        --as="system:serviceaccount:$NVFLARE_TEST_NAMESPACE:$NVFLARE_TEST_CP_ACCOUNT"

   These diagnostics require appropriate impersonation permission; otherwise
   have the operator verify access using the intended account. Check the full
   CP get/delete/patch permissions described above, plus its existing bootstrap
   permissions. Verify that CJ and worker accounts cannot create Pods and that
   their admitted manifests have no token mounts. An RBAC answer alone does not establish admission-policy or
   volume access.

6. **Run the CPU and GPU application baselines.** Inspect fresh smoke Pods for
   the same image reference, pull policy, runtime selection, service account,
   UID/GID, security context, mounts and resource requests used by the target
   execution profile. Record any differences. Require the CPU application to
   complete before the CUDA row, and compare task mode with ordinary resident
   mode. Verify the compute probe's real CPU/CUDA result, then exercise the
   cancellation, artifact/receipt and capacity-reuse cases below. Report a
   failed readiness check by its phase and exact command; preserve application
   failures against their exact candidate before testing a corrected candidate.

Common setup failures
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Diagnose setup before changing application code
   :header-rows: 1
   :widths: 28 38 34

   * - Symptom
     - Check
     - Required evidence
   * - Host GPU works, worker CUDA fails
     - Compare actual runtime selection and device-plugin allocation
     - Fresh Pod with the target runtime executes CUDA
   * - Local image works, generated Pod cannot start
     - Registry access, exact digest, pull policy and pull credentials
     - Fresh ``Always`` Pod starts from the intended image
   * - SJ/CJ start, task Pods hit ``ImagePullBackOff``
     - Compare job image metadata with active CP/CJ worker-image policy
     - Task policy uses its intended published digest
   * - Reset parents cannot restart after a working smoke
     - Check Deployment/Helm and reset templates for a stale image reference
     - New parent Pods pull their intended published image
   * - Clients cannot register
     - Startup endpoint, Service DNS, port and TLS identity
     - Authenticated registration and admin connectivity
   * - Worker cannot read or publish artifacts
     - PVC mount, numeric UID/GID, subtree ownership and storage operations
     - Nonroot storage probe and CJ/worker artifact visibility
   * - Pod creation or cleanup is forbidden
     - Actual service account, RBAC and namespace admission policy
     - Required operations work under the intended identity

Storage and retention verification
----------------------------------

``check_posix_storage`` checks local hard links, rename publication, file fsync,
directory fsync and reading an inode after unlinking its other name. The task
CP job launcher caches a successful probe of its trusted PVC root before CJ
starts; failed probes are retried on the next launch. CP does not run a storage
write probe inside a live job-controlled subtree. Per-attempt workspace, filesystem and identity checks still
run on every launch. The probe is a preflight, not certification of a storage
driver. Before deployment qualification, run it in both CJ and task images on
the chosen volume, and exercise cross-pod
visibility, exclusive creation, no-follow directory operations, file fsync,
atomic small-manifest replacement and crash durability. Attempt and checkpoint
directories must reside on the same filesystem. Do not disable task artifact validation
when any required operation is unsupported.

The backend deletes only its Pod. Immutable results, native checkpoint candidates
and worker file logs live on the PVC after removal. API logs are a bounded,
best-effort tail cached in the CP handle; they do not replace durable worker logs.
The existing result reader decodes the FOBS record and closes that record's
source file before returning to CJ. PyTorch tensor payloads may remain mapped
from adjacent files; the existing sender serializes those tensors individually.
Attempt tensor files remain until terminal receipt, and active POSIX mappings
survive unlink. Shared saved-result replay pins payloads while a reader decodes
or maps them; a concurrent terminal receipt defers cleanup until those readers
finish. A failed read retains the payload and diagnostics. Keep attempt payloads
while receipt is unknown or ``RETRY``. Finish any additional readers that reopen
storage paths before forwarding terminal ``RECEIVED``/``TASK_CLOSED`` into the
existing supervisor's cleanup path. Such lazy readers require explicit ownership;
the supervisor has no general reader-lease API. Failures and unresolved
settlement retain diagnostics and payloads. Checkpoint promotion continues to
use an opaque application file, immutable hard links and an atomic local manifest,
independently of server receipt or algorithm use.

Task workspaces and explicitly enabled persistent resident workspaces are
retained after CJ shutdown, including successful completion; there is no
automatic subtree garbage collector. Enabling resident persistence is therefore
a storage-retention decision. Budget capacity for accumulated checkpoints,
artifacts and workspaces, set storage alerts and define archival/deletion policy.
The resident default remains ephemeral even when a task jobs PVC is configured.
The operator owns archival and deletion under the site's retention policy. Before removing a subtree, confirm that the job is
closed, all CJ/task Pods and in-flight create requests are settled, and no
result/checkpoint readers or uploads still use its files. Retain failure and
unresolved-settlement evidence. Re-launching the same job/site while its subtree
exists raises ``ValueError`` and preserves the files; use a new job identity or
complete operator cleanup before reusing that identity.

A failed pre-create preparation or definitive API rejection (for example quota
403 or invalid manifest 422) releases only the freshly reserved empty directory,
after checking that its inode has not changed. Nonempty directories and replaced
paths remain untouched. A transport error, server error or name conflict can
hide an admitted Pod, so its subtree stays fenced for operator reconciliation.
A subsequent 404 alone does not authorize deleting that fence.

Persistent files and CJ resume
------------------------------

The shared jobs PVC preserves workspace files, committed checkpoints and task
artifacts after a task CJ Pod is removed, and after a resident CJ Pod is removed
when ``persist_resident_workspaces`` is enabled. Scratch files and container
logs outside that mount remain ephemeral. The existing CJ ephemeral-storage
request/limit (default 1Gi) continues to cover Pod-local storage; it does not
limit files on the PVC. Task workers currently set no explicit ephemeral-storage
request/limit, so namespace defaults and node eviction policy apply.

The Kubernetes job handle does not opt into main's automatic CJ restart policy;
CP continues to fence old workers and refuses retained-workspace adoption.
Persistence does not itself implement CJ resume. Applications must write their
required recovery state, and a future restart path must restore committed state,
coordinate with SJ and fence every earlier CJ/worker attempt. The launcher rejects an existing job/site directory rather than overwrite or automatically
adopt it. The current shutdown upload also remains in place for CP's result
archive. Abrupt CJ loss preserves the jobs-PVC files without depending on that
upload running.

Local reproduction and deployment qualification
--------------------------------------------------

Run from the source checkout with test dependencies and the Kubernetes Python
client installed::

   python -m pytest -q tests/unit_test/app_opt/task_launcher
   python -m pytest -q tests/unit_test/app_opt/job_launcher/k8s_launcher_test.py
   python -m pytest -q tests/unit_test/private/fed/task_worker
   ./runtest.sh -s --skip-install

The adapter tests use a deterministic API model and real on-disk artifacts.
Helm-rendering tests verify separate CP/jobs claims, reject private-claim reuse
and overlapping mount paths, and preserve the default chart without a jobs
claim. Both client lifetimes test the shared per-job mount, with resident GPU
resources preserved and task-mode GPU admission delegated to workers. The
worker tests also run the unmodified NPTrainer in a real CPU interpreter and
check authorization before custom imports. They do not launch Kubernetes pods.

The initial worker mount profile contains the job subtree and read-only site
configuration/startup only. Additional dataset volumes mounted into a resident
CJ are not automatically mounted into workers; extending those bindings requires
its own explicit site profile and qualification.

With separately authorized infrastructure, qualify matching Kubernetes CP ->
CPU CJ -> disposable task pods using that
simple application and the separate CP/jobs PVC profile. Inspect actual mounts
and verify that deleting CJ preserves its committed checkpoint on the jobs PVC
while CJ/workers cannot see the private CP workspace. Record exact source/image digests and rendered manifests;
confirm each pod UID and native allocation. Exercise success, pending/running
cancellation, CJ death, lost Cell replies, cross-job capability rejection,
image/admission failure, eviction, API outage, missing shutdown
evidence and incomplete/corrupt artifacts. Verify validated output after removal,
checkpoint continuity, lost-receipt resubmission without compute, active-reader
retention and ordinary resident JobLauncher behavior. For GPU qualification,
queue another owned worker and record it obtaining the same released scheduler
capacity while the previous result upload remains deliberately blocked. Preserve
timestamps, container/sandbox observations, pod-removal evidence and GPU workload
identities. A submitted pod, visible GPU, successful delete response or unit-test
pass cannot substitute for these observations.
