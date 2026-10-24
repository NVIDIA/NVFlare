Kubernetes task launcher
========================

Backend contract
----------------

* Launch authority: the Client Parent (CP) creates, observes and cancels task
  Pods through its existing Kubernetes client. The CPU Client Job (CJ) uses a
  job-scoped Cell command interface. CJ and task Pods use the configured
  unprivileged worker service account with automatic token mounting disabled.
  Applications cannot choose Pod manifests, accounts, namespace or storage.
* Identity: every physical attempt has a unique namespace/name/UID, labels for
  job/site/task/attempt, and full identities in annotations. Deletes use UID
  preconditions; replacement pods are never adopted or deleted.
* Storage: CP's private workspace uses its own PVC. A second, shared jobs PVC
  holds per-job workspaces for both resident and task CJs. CJ and its task pods
  mount only their job/site subtree at the same absolute workspace root. That
  root contains bootstrap, application code, input, immutable output and
  checkpoints. Pod-local
  ``emptyDir`` is insufficient. CP explicitly mounts that PVC and creates an
  owned per-job/site subtree. Paths outside the configured root are rejected.
* Resources: both CPU and GPU task workers are required. Kubernetes admits
  their native CPU/memory/GPU requests; the CPU CJ has no GPU request. GPU
  workers obtain devices through the cluster's NVIDIA device plugin. CPU
  workers mask GPU visibility even when using a GPU-capable image. No Process
  GPU allocator, physical retry scheduler or distributed trainer is introduced.
* Settlement: graceful deletion alone is insufficient. A finalizer preserves
  the owned pod until shutdown evidence is observed. Scheduled pods require
  terminal status, termination of every admitted container, and kubelet sandbox
  teardown evidence. An unscheduled pod requires a deletion timestamp that
  fences scheduling. Only after this evidence and confirmed pod removal does
  the handle report settlement. Missing/replaced pods without prior evidence,
  API errors, node loss, and timeouts retain unresolved ownership. Transient
  observation errors preserve the last-known execution phase and report a
  diagnostic; they do not start terminal settlement of a healthy Running Pod.
  An ambiguous create remains fenced even after a 404 with an unknown UID:
  the server may still admit the outstanding request later.
* Results: pod cleanup never deletes PVC files. CJ validates the existing
  immutable result/checkpoint contract after settlement. Receipt, algorithm
  acceptance and local checkpoint promotion remain separate. Payload release
  requires the existing supervisor receipt/closure and reader-quiescence rules.

Trusted site selection
-----------------------

The current profile uses one backend for the full CP/CJ/task-worker chain:
Process launches all three as processes; Kubernetes runs all three in Pods.
Mixed chains are unsupported. CPU versus GPU is a resource choice within that
backend, so the CPU CJ can request either CPU or GPU task Pods from CP.

The shared task policy selects the exact site-configured task launcher and
validates its canonical ``local/resources.json`` configuration. Kubernetes
requires one matching ``ClientK8sJobLauncher`` / ``K8sTaskLauncher`` pair,
matching CJ/worker namespace, PVC, workspace path and numeric filesystem identity. Mixed modes fail before task launch.
Job configuration cannot choose the backend or launch authority. Resource
export checks shape; CPU/GPU capability is checked at the trusted site boundary.

Before starting CJ, CP loads canonical site policy and binds the task backend to
its admitted job/site metadata. CJ's ``START_RUN`` binds a tokenless proxy to the
existing CP Cell endpoint using the per-job workspace-transfer capability. CP
checks both that capability and the authenticated CJ origin on every request.
Only bounded JSON commands and staged paths are accepted; job-controlled data
cannot supply a Pod manifest or override the site's runtime authority. CP fences
attempt identities, admits one unsettled task per job, and owns a launch even
when its reply is lost. CJ cannot launch or inspect another job's attempts. Native workers receive
only the site-configured worker environment. Process-style environment
passthrough is rejected for native backends. These shared selection and resource
rules delegate to an installed native backend's site validator. A backend whose
implementation is absent is rejected as unsupported. This change supplies the
Kubernetes implementation; Docker and Slurm implementations are delivered
separately. Use the full ``path`` for ``ClientK8sJobLauncher``; its short ``name``
is not registered in the component registry.

The Kubernetes adapter accepts only the common CJ's strict staged-bootstrap
request, bound to its owning job/site by ``START_RUN``. The request's local
interpreter and parent PID are never used in the Pod command and do not select
a Process backend. The Pod command uses the configured image Python and site
worker environment, with resource defaults overridden by the bound job's resource
metadata. Standalone native requests and per-request environment/resource
overrides are rejected. Both backends invoke the existing
``task_worker_process`` entry point. The Pod launcher selects
``--execution_scope native`` because CJ is not the worker's OS parent. Process
launches require an explicit ``--parent_pid`` greater than 1 and retain
expected-parent and owned-process-group checks. A missing parent flag or an
init-adopted Process worker is rejected before compute. Native workers support
PID 1 without Process parent/group checks. ``TaskLaunchRequest``,
``TaskSupervisor`` and artifact formats are unchanged.

Job and task launchers share stateless Pod construction and startup diagnostics.
The CP task backend creates ``K8sTaskHandle``; CJ receives a remote task handle.
Credential/workspace-transfer cleanup stays with the job handle. CP retains
physical task ownership independently of those job credentials.

The existing ``ClientK8sJobLauncher`` handles both execution lifetimes. Its
optional ``job_workspace_pvc``, ``job_workspace_pvc_root`` and numeric identity
settings enable persistent workspaces for both CJ lifetimes. With this profile,
it replaces the generated ``workspace-job`` emptyDir with the job's subtree on
the shared jobs PVC for resident and task CJs. Resident CJs retain their native
GPU resources; only task-mode CJs have their GPU resources removed. Deployments
without this optional storage profile retain the existing resident emptyDir
behavior.
The task profile removes CJ GPU requests/limits and visibility, selects the
unprivileged worker account with API token mounting disabled, and uses the
durable workspace. It preserves the existing workspace download and credential-Secret lifecycle. A pod-template PVC override
alone is insufficient because generated workspace volumes take precedence.

CP seals task admission before terminating CJ. A Pending CJ that never starts
can be deleted using the normal job launcher path. CP continues observing and
cancelling workers after CJ exits, including a create still in flight when CJ
was terminated. The CJ end-run handler makes bounded cleanup attempts; it does
not have to remain alive indefinitely to preserve Kubernetes cleanup authority.
An independent CP monitor also detects a terminal, deleting, replaced or missing
CJ and closes its task admission. API observation errors preserve ownership.

Missing/replaced task Pods and lost settlement evidence remain explicitly fenced
in CP, even after CJ exits. Retained ownership does not authorize a replacement
attempt or result publication. CP must remain available to perform this cleanup;
a CP crash/restart or permanently lost node evidence requires operator
reconciliation of stranded Pods/finalizers after verifying runtime shutdown.
Never remove a finalizer solely to clear a stuck Pod.

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

The resident workspace download stages ``local/`` and this job's files into the
subtree. The worker mounts ``local/`` read-only from that same subtree, protecting
canonical site policy against task writes. Both CJ and worker mount the existing
site startup Secret read-only at ``startup/``; the resident workspace download
excludes startup. The Secret construction already withholds site private keys.
Task pods receive no CJ bootstrap environment, Kubernetes API token or kubeconfig.
Authorized workers still share the job's filesystem identity and can read files
already present there; this profile does not create a separate filesystem trust
boundary for job credentials. Do not distribute a cluster-admin kubeconfig.

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
change the job launcher's existing ``pending_timeout`` policy.

For example, two CPU CJs can keep their jobs alive while their task Pods request
one GPU each from a cluster with one allocatable GPU. One worker runs and the
other queues; the GPU becomes reusable between task attempts, before CJ result
upload finishes. Kubernetes selects the next eligible Pod; this launcher does
not enforce alternating jobs. CPU/memory requests must also fit the CJs, parent
and worker. The two-job GPU-contention scenario requires cluster qualification.

Use native ``launcher_spec`` GPU requests. Positive portable or legacy
``resource_spec`` CP GPU reservations are rejected by this task profile: they
reserve GPUs for the entire job independently of the task Pod. Portable
CPU/memory mapping remains supported, subject to the existing native/portable
conflict rules. Correct task delegation of whole-job GPU reservations is a
separate prerequisite for portable GPU support.

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
         "job_workspace_pvc": "site-1-jobs",
         "job_workspace_pvc_root": "/mnt/nvflare-jobs",
         "workspace_mount_path": "/var/tmp/nvflare/workspace",
         "run_as_user": 1000,
         "run_as_group": 1000
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
Its normal native metadata supplies the CJ image and task resources, for example::

   {
     "launcher_spec": {
       "default": {
         "k8s": {
           "image": "site-approved-cj:reviewed-source",
           "num_of_gpus": 1,
           "cpu": "1",
           "memory": "1Gi"
         }
       }
     }
   }

Use ``num_of_gpus=0`` for a CPU worker. The task CJ profile removes GPU resources
and visibility from CJ's manifest for either worker profile. Images must contain
the same reviewed framework source and the required application dependencies.

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
directories must reside on the same filesystem. Do not disable P4 validation
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
use P4's opaque file, immutable hard links and atomic local manifest,
independently of server receipt or algorithm use.

Per-job/site PVC subtrees are retained after CJ shutdown; there is no automatic
subtree garbage collector. The operator owns archival and deletion under the
site's retention policy. Before removing a subtree, confirm that the job is
closed, all CJ/task Pods and in-flight create requests are settled, and no
result/checkpoint readers or uploads still use its files. Retain failure and
unresolved-settlement evidence. Re-launching the same job/site while its subtree
exists raises ``ValueError`` and preserves the files; use a new job identity or
complete operator cleanup before reusing that identity.

Persistent files and CJ resume
------------------------------

The shared jobs PVC preserves workspace files, committed checkpoints and task
artifacts after either kind of CJ Pod is removed. Scratch files and container
logs outside that mount remain ephemeral. The existing CJ ephemeral-storage
request/limit (default 1Gi) continues to cover Pod-local storage; it does not
limit files on the PVC. Task workers currently set no explicit ephemeral-storage
request/limit, so namespace defaults and node eviction policy apply.

Persistence does not itself implement CJ resume. Applications must write their
required recovery state, and a future restart path must restore committed state,
coordinate with SJ and fence every earlier CJ/worker attempt. This PR continues
to reject an existing job/site directory rather than overwrite or automatically
adopt it. The current shutdown upload also remains in place for CP's result
archive. Abrupt CJ loss preserves the jobs-PVC files without depending on that
upload running.

For profiles from earlier unreleased revisions, rename ``task_workspace_pvc`` /
``task_workspace_pvc_root`` to ``job_workspace_pvc`` /
``job_workspace_pvc_root``. The existing hashed ``nvflare-tasks`` directory format
is retained so this configuration change cannot bypass the retained-data fence.

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
