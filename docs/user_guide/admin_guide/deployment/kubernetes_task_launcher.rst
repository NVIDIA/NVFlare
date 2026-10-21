Kubernetes task launcher (experimental adapter)
==================================================

Backend contract
----------------

* Launch authority: the CPU Client Job (CJ) uses a site-owned service account
  confined to one namespace. Task pods have a separate service account with no
  API permissions and automatic token mounting disabled. Applications select
  only ``execution_lifetime="task"``; they cannot choose these accounts or storage.
* Identity: every physical attempt has a unique namespace/name/UID, labels for
  job/site/task/attempt, and full identities in annotations. Deletes use UID
  preconditions; replacement pods are never adopted or deleted.
* Storage: an existing durable POSIX PVC is explicitly mounted at the same
  absolute workspace root in CJ and task pods. That root contains bootstrap,
  application code, input, immutable output and checkpoints. Pod-local
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
  API errors, node loss, and timeouts retain unresolved ownership.
* Results: pod cleanup never deletes PVC files. CJ validates the existing
  immutable result/checkpoint contract after settlement. Receipt, algorithm
  acceptance and local checkpoint promotion remain separate. Payload release
  requires the existing supervisor receipt/closure and reader-quiescence rules.

Trusted site selection
-----------------------

The shared task policy selects the exact site-configured task launcher and
validates its canonical ``local/resources.json`` configuration. Kubernetes
requires one matching ``ClientK8sTaskJobLauncher`` / ``K8sTaskLauncher`` pair,
matching namespace, PVC, workspace path and numeric filesystem identity, and
distinct CJ/worker service accounts. Mixed modes fail before task launch.
Job configuration cannot choose the backend or launch authority. Resource
export checks shape; CPU/GPU capability is checked at the trusted site boundary.

The CJ forwards its existing ``START_RUN`` event to the selected launcher. The
Kubernetes hook binds its owning job/site and resolves that site's job metadata
before launch; missing or changed bindings fail closed. Native workers receive
only the site-configured worker environment. Process-style environment
passthrough is rejected for native backends. These shared selection and resource
rules also apply to Docker and Slurm, with their own site validators.

The Kubernetes adapter materializes P4's strict staged-bootstrap request into a
site-image command. It ignores the CJ interpreter, parent PID and Process
environment and uses only its configured image Python/environment and resource
defaults, overridden by the bound job's resource metadata. Explicit image-native
requests remain available to trusted callers.
Neither ``TaskLaunchRequest``, ``TaskSupervisor``, artifact formats nor the
Process worker's parent/group guards are changed.

``ClientK8sTaskJobLauncher`` supplies the Kubernetes-specific CJ preparation. A
small manifest hook in the resident JobLauncher replaces the generated
``workspace-job`` emptyDir only for applications whose existing flag is
``execution_lifetime="task"``. Job-lifetime manifests retain resident behavior.
The task profile removes CJ GPU requests/limits and visibility, selects the
site-owned CJ account, and uses the durable workspace. It preserves the existing
workspace download and credential-Secret lifecycle. A pod-template PVC override
alone is insufficient because generated workspace volumes take precedence.

Independent cleanup after abrupt CJ death remains outside this profile; the
site operator owns reconciliation of stranded pods/finalizers after verifying
node/runtime shutdown. Never remove a finalizer solely to clear a stuck pod.

Site prerequisites
------------------

Use an existing namespace, preapproved worker image and durable PVC. For pods on
different nodes the selected volume must support the required concurrent access
(commonly RWX); an access-mode declaration alone does not establish filesystem
semantics. All bootstrap paths are absolute paths in the same explicitly
configured mount in both pods. Configure CP's ``task_workspace_pvc_root`` as its
explicit mount of ``task_workspace_pvc``. CP creates
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

The existing CP JobLauncher needs its documented pod/event/Secret permissions
for CJ bootstrap and credential lifecycle. The CJ launch account needs only
``create``, ``get``, ``delete`` and ``patch`` on ``pods``, plus ``get`` on
``pods/log`` in the selected namespace. It needs no node, Secret, Role, PVC or
cluster-role writes. The site operator must preconfigure an appropriate
namespace boundary and admission policy: pod creation can otherwise select any
service account or mount any Secret in that namespace. Kubernetes RBAC alone
cannot restrict create to particular pod labels or service accounts. Do not
give the task account any API permissions. No Roles, bindings, accounts,
namespaces or claims are created. The task adapter creates no Secrets; the CP
profile retains the resident launcher's existing startup/bootstrap Secrets.

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
Use a finite worker timeout and settlement bounds appropriate to the configured
termination grace period and storage driver.

Use native ``launcher_spec`` GPU requests. Positive portable or legacy
``resource_spec`` CP GPU reservations are rejected by this task profile: they
reserve GPUs for the entire job independently of the task Pod. Portable
CPU/memory mapping remains supported, subject to the existing native/portable
conflict rules. Correct task delegation of whole-job GPU reservations is a
separate prerequisite for portable GPU support.

Configure the site's matching launcher pair in canonical resources, replacing
its existing client JobLauncher component and preserving other components.
After ``nvflare deploy prepare``, put this task profile in the canonical
``local/resources.json`` override. Keep the generated
``local/resources.json.default`` when using the existing ``deploy k8 stage``
command, which identifies the prepared kit through that generated default.
Task-profile settings are not additional ``prepare`` YAML keys. Apply the
override after preparation, then stage both local files::

   {
     "components": [{
       "id": "k8s_job_launcher",
       "path": "nvflare.app_opt.task_launcher.k8s_job_launcher.ClientK8sTaskJobLauncher",
       "args": {
         "config_file_path": null,
         "namespace": "site",
         "task_workspace_pvc": "task-storage",
         "task_workspace_pvc_root": "/mnt/task-storage",
         "workspace_mount_path": "/var/tmp/nvflare/workspace",
         "task_cj_service_account": "cj-launch-authority",
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
           "workspace_pvc": "task-storage",
           "worker_service_account": "task-no-api",
           "run_as_user": 1000,
           "run_as_group": 1000,
           "cpu_cores": 1,
           "memory_mb": 512
         }
       },
       "worker_timeout": 120,
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
Pod, CJ service account, workspace root, site name and worker image digest for
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
        --as="system:serviceaccount:$NVFLARE_TEST_NAMESPACE:$NVFLARE_TEST_CJ_ACCOUNT"
      kubectl -n "$NVFLARE_TEST_NAMESPACE" auth can-i get pods/log \
        --as="system:serviceaccount:$NVFLARE_TEST_NAMESPACE:$NVFLARE_TEST_CJ_ACCOUNT"

   These diagnostics require appropriate impersonation permission; otherwise
   have the operator verify access using the intended account. Check the full
   get/delete/patch permissions described above, plus CP's existing bootstrap
   permissions. An RBAC answer alone does not establish admission-policy or
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
directory fsync and reading an inode after unlinking its other name. It is a
preflight, not certification of a storage driver. Before deployment qualification,
run it in both CJ and task images on the chosen volume, and exercise cross-pod
visibility, exclusive creation, no-follow directory operations, file fsync,
atomic small-manifest replacement and crash durability. Attempt and checkpoint
directories must reside on the same filesystem. Do not disable P4 validation
when any required operation is unsupported.

The backend deletes only its Pod. Immutable results, native checkpoint candidates
and worker file logs live on the PVC after removal. API logs are a bounded,
best-effort tail cached in the handle; they do not replace durable worker logs.
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

Local reproduction and deployment qualification
--------------------------------------------------

Run from the source checkout with test dependencies and the Kubernetes Python
client installed::

   python -m pytest -q tests/unit_test/app_opt/task_launcher
   python -m pytest -q tests/unit_test/app_opt/job_launcher/k8s_launcher_test.py
   python -m pytest -q tests/unit_test/private/fed/task_worker
   ./runtest.sh -s --skip-install

The adapter tests use a deterministic API model and real on-disk artifacts. The
worker tests also run the unmodified NPTrainer in a real CPU interpreter and
check authorization before custom imports. They do not launch Kubernetes pods.

The initial worker mount profile contains the job subtree and read-only site
configuration/startup only. Additional dataset volumes mounted into a resident
CJ are not automatically mounted into workers; extending those bindings requires
its own explicit site profile and qualification.

With separately authorized infrastructure, qualify matching Kubernetes CP ->
CPU CJ -> disposable task pods using that
simple application. Record exact source/image digests and rendered manifests;
confirm each pod UID and native allocation. Exercise success, pending/running
cancellation, image/admission failure, eviction, API outage, missing shutdown
evidence and incomplete/corrupt artifacts. Verify validated output after removal,
checkpoint continuity, lost-receipt resubmission without compute, active-reader
retention and ordinary resident JobLauncher behavior. For GPU qualification,
queue another owned worker and record it obtaining the same released scheduler
capacity while the previous result upload remains deliberately blocked. Preserve
timestamps, container/sandbox observations, pod-removal evidence and GPU workload
identities. A submitted pod, visible GPU, successful delete response or unit-test
pass cannot substitute for these observations.
