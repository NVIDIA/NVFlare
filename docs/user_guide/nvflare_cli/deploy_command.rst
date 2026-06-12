.. _deploy_prepare_command:

##############
Deploy Command
##############

``nvflare deploy`` prepares already-created server or client startup kits for a
site deployment runtime. The first supported subcommand is
``nvflare deploy prepare``.

``deploy prepare`` does not create identities, certificates, startup kits, or
Kubernetes clusters. Use ``nvflare provision`` or the distributed
``nvflare cert`` / ``nvflare package`` workflow first, then run
``deploy prepare`` on each server or client kit that should run in Docker or
Kubernetes.

For Kubernetes deployment workflow, see :ref:`helm_chart`. For job-level Docker
and Kubernetes image settings, see :ref:`launcher_spec`.

*****
Usage
*****

.. code-block:: none

   nvflare deploy prepare <startup-kit-dir> [--output <prepared-kit-dir>] [--config <runtime-config.yaml>]

Arguments:

- ``<startup-kit-dir>``: existing server or client startup kit directory.
- ``--kit``: named-option alias for ``<startup-kit-dir>``.
- ``--output``: directory for the prepared kit copy. Defaults to
  ``<startup-kit-dir>/prepared/<runtime>``.
- ``--config``: YAML runtime config. Defaults to
  ``<startup-kit-dir>/config.yaml``.
- ``--schema``: print command schema as JSON and exit.

Default convention:

Put the runtime config in the startup kit as ``config.yaml``, then run
``nvflare deploy prepare <startup-kit-dir>``. The command reads ``runtime`` from
that config and writes the prepared copy to ``<startup-kit-dir>/prepared/docker``
or ``<startup-kit-dir>/prepared/k8s``. Use ``--config`` to read a config file
from another path, and use ``--output`` to write the prepared kit somewhere else.

Admin startup kits are not supported by ``deploy prepare`` because admin kits do
not run parent server or client processes.

The input kit is treated as read-only. Runtime-specific files are written to
the prepared output directory.

*************
Docker Config
*************

Example ``docker.yaml``:

.. code-block:: yaml

   runtime: docker

   parent:
     docker_image: registry.example.com/nvflare-site:2.8
     network: nvflare-network

   job_launcher:
     default_python_path: /usr/local/bin/python
     default_job_env:
       NCCL_P2P_DISABLE: "1"
     default_job_container_kwargs:
       shm_size: 8g
       ipc_mode: host

Top-level keys:

- ``runtime``: required, must be ``docker``.
- ``parent``: required mapping for the parent server/client container.
- ``job_launcher``: optional mapping for per-job Docker container defaults.

``parent`` keys:

- ``docker_image``: required parent image used by ``startup/start_docker.sh``.
- ``network``: Docker network for parent and job containers. Defaults to
  ``nvflare-network``.

``job_launcher`` keys:

- ``default_python_path``: Python executable used in job containers unless a
  job overrides it with ``launcher_spec[site]["docker"]["python_path"]``.
- ``default_job_env``: environment variables injected into every Docker job
  container.
- ``default_job_container_kwargs``: Docker SDK container kwargs applied to
  every job container. Launcher-controlled keys such as ``volumes``,
  ``mounts``, ``network``, ``environment``, ``command``, ``name``, ``detach``,
  ``user``, and ``working_dir`` are rejected.

Prepare and start:

.. code-block:: shell

   nvflare deploy prepare ./site-1 --config docker.yaml --output ./site-1-docker
   cd ./site-1-docker
   ./startup/start_docker.sh

The command writes:

- ``startup/start_docker.sh``
- patched ``local/resources.json.default`` with ``DockerJobLauncher``
- patched ``local/comm_config.json``
- ``local/study_data.yaml`` template when missing

**********
K8s Config
**********

Example ``k8s.yaml``:

.. code-block:: yaml

   runtime: k8s
   namespace: nvflare

   parent:
     docker_image: registry.example.com/nvflare-site:2.8
     image_pull_secrets:
       - registry-credentials
     parent_port: 8102
     workspace_pvc: nvflws
     workspace_mount_path: /var/tmp/nvflare/workspace
     python_path: /usr/local/bin/python3
     resources:
       requests:
         cpu: "2"
         memory: 8Gi
     pod_security_context: {}

   job_launcher:
     config_file_path: null
     pending_timeout: 300
     default_python_path: /usr/local/bin/python3
     image_pull_secrets:
       - job-registry-credentials
     job_pod_security_context: {}

Top-level keys:

- ``runtime``: required, must be ``k8s``.
- ``namespace``: Kubernetes namespace for parent and job pods. Defaults to
  ``default``.
- ``server_service_name``: optional Kubernetes Service name for the FL server.
- ``parent``: required mapping for the generated parent Helm chart.
- ``job_launcher``: optional mapping for dynamically launched job pods.

``parent`` keys:

- ``docker_image``: required parent image used by the Helm chart.
- ``image_pull_secrets``: optional list of existing Kubernetes Secret names to
  render as ``imagePullSecrets`` on the parent server/client pod. Create these
  registry pull Secrets in the target namespace before installing the chart.
  This setting applies to the generated parent pod chart. Use
  ``job_launcher.image_pull_secrets`` for dynamically launched job pods.
- ``parent_port``: port that job pods use to reach the parent pod's FLARE
  process.
  Defaults to ``8102``.
- ``workspace_pvc``: PVC claim containing the runtime workspace. Defaults to
  ``nvflws``.
- ``workspace_mount_path``: parent pod workspace mount path. Defaults to
  ``/var/tmp/nvflare/workspace``. This is also written into the Kubernetes job
  launcher config so job pods use the same in-container workspace path.
- ``python_path``: Python executable used by the parent pod command. Defaults
  to ``/usr/local/bin/python3``.
- ``resources``: parent pod resources rendered into ``values.yaml``.
- ``pod_security_context``: parent pod security context rendered into
  ``values.yaml``.

``job_launcher`` keys:

- ``config_file_path``: kubeconfig path used by ``K8sJobLauncher``. Use
  ``null`` for in-cluster config, where the Kubernetes Python client uses the
  pod's ServiceAccount token.
- ``pending_timeout``: seconds to wait for a job pod to leave ``Pending``.
- ``default_python_path``: Python executable used in job pods unless a job
  overrides it with ``launcher_spec[site]["k8s"]["python_path"]``. Defaults to
  ``/usr/local/bin/python3``.
- ``image_pull_secrets``: optional list of existing Kubernetes Secret names
  attached to every dynamically launched job pod for this prepared site. This
  is configured by the deployment owner and does not require job authors to add
  registry Secret names to ``meta.json``.
- ``job_pod_security_context``: security context passed to dynamically
  launched job pods.
- ``study_job_spec_file_path``: optional YAML mapping from study name to
  Kubernetes Pod template file. Matching studies use the template with
  launcher-owned fields overlaid. If this is set without a configured
  ``study_data_pvc_file_path``, no study-data PVC mounts are added. If both are
  configured and the job study has entries in both files, the template is used
  and the study-data entries are added as extra volume mounts with a warning.
  Template volumes or job-container mounts named ``workspace-job`` or
  ``startup-kit`` are replaced by the launcher-generated workspace and startup
  mounts.

Prepare the parent server or client kit first:

.. code-block:: shell

   nvflare deploy prepare ./site-1 --config k8s.yaml --output ./site-1-k8s

Then choose one of the following two staging methods before starting the parent
pod with Helm.

**Method 1: copy ``startup/`` and ``local/`` into the workspace PVC.**

Copy the prepared kit's ``startup/`` and ``local/`` directories into the root of
the configured workspace PVC. The chart mounts that PVC at
``workspace_mount_path``. The OpenShift helper
:github_nvflare_link:`examples/devops/openshift/scripts/k8s_deploy.sh <examples/devops/openshift/scripts/k8s_deploy.sh>`
is a complete scripted example of this PVC-copy method.

After the copy is complete, install or upgrade the chart:

.. code-block:: shell

   helm upgrade --install site-1 ./site-1-k8s/helm_chart --namespace nvflare

**Method 2: stage ``local/`` as a ConfigMap and ``startup/`` as a Secret.**

Use ``nvflare deploy k8s stage`` to create the Kubernetes resources and patch
the generated Helm values. ``nvflare deploy k8 stage`` is accepted as an alias.

.. code-block:: shell

   nvflare deploy k8s stage ./site-1-k8s --namespace nvflare
   helm upgrade --install site-1 ./site-1-k8s/helm_chart --namespace nvflare

This keeps the workspace PVC mounted at ``workspace_mount_path`` for writable
runtime state, but the parent pod reads ``local/`` from the generated ConfigMap
and ``startup/`` from the generated Secret.

``nvflare deploy prepare`` also patches the prepared kit's internal
communication settings so dynamically launched job pods connect to the generated
parent Kubernetes Service on ``parent_port``. If you customize the chart's
Service name or port, keep that Service endpoint consistent with the prepared
kit.

The command writes:

- ``helm_chart/`` for the parent server or client pod
- patched ``local/resources.json.default`` with ``K8sJobLauncher``
- patched ``local/comm_config.json``
- ``local/study_data.yaml`` template when missing

************
K8s Staging
************

``nvflare deploy k8s stage`` creates Kubernetes resources from a prepared K8s
kit and patches the generated Helm chart values to mount them:

.. code-block:: none

   nvflare deploy k8s stage <prepared-kit-dir> [--namespace <namespace>]

The command requires Kubernetes CLI access to the target cluster. It uses
``kubectl`` by default; set ``--kubectl oc`` or ``KUBECTL=oc`` when staging into
OpenShift with ``oc``. It:

- creates or updates a ConfigMap containing every file under prepared ``local/``
- creates or updates a Secret containing every file under prepared ``startup/``
- patches ``helm_chart/values.yaml`` so the parent pod mounts the ConfigMap at
  ``workspace_mount_path/local`` and the Secret at
  ``workspace_mount_path/startup``

The resource names default to ``nvflare-local-<site>`` and
``nvflare-startup-<site>``. Override them with ``--local-configmap`` and
``--startup-secret``. The namespace defaults to the namespace written into the
prepared kit's ``K8sJobLauncher`` config, or ``default`` when unavailable.

After this staging command succeeds, run the printed ``helm_command`` or the
equivalent ``helm upgrade --install`` command for the prepared chart to start
the parent server or client pod.

The generated Helm chart still mounts the configured workspace PVC at the
workspace root. The ConfigMap and Secret only replace the ``local/`` and
``startup/`` subdirectories.

**********
Job Images
**********

Docker and Kubernetes jobs must specify a job image in ``meta.json``. The
preferred form is ``launcher_spec``:

.. code-block:: json

   {
     "launcher_spec": {
       "default": {
         "docker": {"image": "registry.example.com/nvflare-job:2.8"},
         "k8s": {"image": "registry.example.com/nvflare-job:2.8"}
       },
       "site-1": {
         "docker": {"shm_size": "8g"}
       }
     },
     "resource_spec": {
       "site-1": {
         "num_of_gpus": 1
       }
     }
   }

``launcher_spec["default"][mode]`` applies to every site for that mode.
``launcher_spec[site][mode]`` overrides the default for one site. Keep resource
requests such as ``num_of_gpus`` in ``resource_spec``.

***********
Exit Status
***********

Validation errors exit with code ``4`` and report a structured error. Common
causes include:

- missing or invalid runtime config
- unsupported admin startup kit
- missing ``startup/`` or ``local/`` directory
- invalid ``resources.json.default``
- reserved Docker launcher kwargs
- ``--output`` pointing at or inside the input kit
