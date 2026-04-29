.. _brev_deployment:

###############################
Brev Kubernetes Helm Deployment
###############################

This guide walks through an end-to-end NVIDIA FLARE deployment on two Brev
single-node Kubernetes environments, treated as two Kubernetes clusters:

* one cluster for the FLARE server;
* one cluster for a single FLARE client named ``site-1``.

It covers provisioning, editing ``project.yml``, generating Helm charts for the
server and client, creating the required PersistentVolumeClaims (PVCs), staging
the provisioned folders into the PVCs, and deploying the generated charts.

The Kubernetes environments are created from the Brev web UI. The exact control
labels in Brev can change, but the workflow is the same: create an environment,
select compute, switch the software configuration to ``Single-node Kubernetes``,
open a Brev shell, copy the startup kits to the environment, then deploy with
``kubectl`` and ``helm`` inside each Brev environment.

Assumptions
===========

The examples use:

* one server named ``server1``;
* one client named ``site-1``;
* one Brev Kubernetes environment named ``nvflare-server-k8s``;
* one Brev Kubernetes environment named ``nvflare-site-1-k8s``;
* namespace ``nvflare`` in both clusters;
* PVC names ``nvflws``, ``nvfletc``, and ``nvfldata`` in both clusters;
* an externally reachable DNS name for the server, for example
  ``server1.example.com``;
* a container image in a registry that both clusters can pull, for example
  ``registry.example.com/nvflare:dev``.

Using the same namespace and PVC names in both clusters is safe because each
cluster has its own Kubernetes API and storage backend.

References:

* `NVIDIA Brev console documentation <https://docs.nvidia.com/brev/guides/console-reference>`__
* `Brev connectivity documentation <https://docs.nvidia.com/brev/cli/connectivity>`__
* :ref:`helm_chart`

Create Brev Kubernetes Environments
===================================

Create the server Kubernetes environment first, then repeat the same flow for
the client Kubernetes environment. In the Brev UI, a single-node Kubernetes
environment is created from the same ``GPUs`` page used for GPU and CPU
development environments.

Server Kubernetes Environment
-----------------------------

#. Sign in to the `Brev console <https://brev.nvidia.com>`__.
#. Open ``GPUs`` in the top navigation.
#. Click ``Create Environment``.

   .. figure:: ../../../resources/brev_creating.png
      :alt: Brev GPU Environments page with the Create Environment button.

      Start from the Brev ``GPUs`` page and create a new environment.

#. Select the hardware for the server environment. A CPU instance is enough for
   the FLARE server unless your server-side workflow requires GPU compute.

   .. figure:: ../../../resources/brev_instance.png
      :alt: Brev Create Environment page with CPU selected.

      Select a CPU or GPU instance type. For a basic server deployment, a CPU
      instance type is sufficient.

#. Configure storage and region:

   * ``Name``: ``nvflare-server-k8s``.
   * ``Organization`` or ``Project``: choose the Brev organization that should
     own the environment.
   * ``Provider`` or ``Cloud``: choose the cloud provider where the server
     should run.
   * ``Region``: choose a region reachable by the client cluster and by your
     admin operator.
   * ``Disk Storage``: choose enough space for the container image cache, the
     provisioned workspace PVC, server job storage, snapshots, and logs.

   .. figure:: ../../../resources/brev_config_instance.png
      :alt: Brev hardware, storage, region, and software configuration page.

      Configure disk storage and region before changing the software mode.

#. In ``Software Configuration``, click ``Edit``.
#. Select ``Single-node Kubernetes``.
#. Keep ``Install Kubernetes Dashboard`` enabled if you want browser access to
   the cluster dashboard.
#. Leave ``Run a cluster init script`` disabled unless your organization has a
   required initialization script.
#. Click ``Apply``.

   .. figure:: ../../../resources/brev_select_k8s.png
      :alt: Brev software picker with Single-node Kubernetes selected.

      Choose ``Single-node Kubernetes`` so the environment is created with
      Kubernetes, ``kubectl``, and ``helm`` ready to use.

#. Expand ``Advanced`` only if you need to set custom network or startup
   options.
#. Set ``Name Instance`` to ``nvflare-server-k8s``.
#. Click ``Deploy``.

   .. figure:: ../../../resources/brev_deploy.png
      :alt: Brev deployment page showing Name Instance and Deploy.

      Name the server environment and deploy it.

#. Wait until the environment status is ``Running`` or ``Ready``.

Client Kubernetes Environment
-----------------------------

Repeat the same web UI flow and use these values:

* ``Name``: ``nvflare-site-1-k8s``.
* ``Instance Type``: choose CPU or GPU compute based on the jobs that ``site-1``
  will run.
* ``Networking``: the client cluster needs outbound access to
  ``server1.example.com:8002``.
* ``Disk Storage``: choose enough space for the client workspace, logs, and data
  PVC.
* ``Software Configuration``: choose ``Single-node Kubernetes``.
* ``Ports``: no inbound FLARE port is required for this basic client
  deployment. The client connects outbound to the server on ``8002``.

Enable Server Port Access and SSH
---------------------------------

After both Kubernetes environments are running, open the server environment's
``Access`` page. In the ``Using Ports`` section, expose the FLARE federated
learning port, ``fed_learn_port`` ``8002``:

This guide does not set ``admin_port`` in ``project.yml``. When ``admin_port``
is omitted, NVFlare uses the same value as ``fed_learn_port``. Therefore, the
Brev server environment only needs to expose ``fed_learn_port`` ``8002``.

#. Find ``TCP/UDP Ports``.
#. In ``Expose Port(s)``, enter ``8002``.
#. Select the access scope. ``Allow All IPs`` is convenient for a quick test;
   restrict this to known client/admin source IPs for a real deployment.
#. Click ``Expose Port``.
#. Confirm that the table lists port ``8002`` and shows a public endpoint such
   as ``<server-ip>:8002``.

.. figure:: ../../../resources/brev_port.png
   :alt: Brev Access page showing copy, secure links, and TCP ports.

   In ``Using Ports``, expose the server ``fed_learn_port`` ``8002``. The same
   page also shows the ``brev copy`` command format for uploading files to an
   environment.

Copy the public ``host:port`` value for port ``8002``. Point
``server1.example.com`` to the host/IP portion of that endpoint. Do not include
the port in ``default_host``; the port is already configured as
``fed_learn_port: 8002`` in ``project.yml``.

The environment also provides SSH instructions through the ``Access`` page:

.. figure:: ../../../resources/brev_ssh.png
   :alt: Brev Access page showing Brev CLI install, login, and shell commands.

   Use the Brev CLI commands shown in the UI to install the CLI, log in, and
   open a shell on the Kubernetes environment.

Install and authenticate the Brev CLI on your local workstation if it is not
already available:

.. code-block:: shell

   sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"
   brev login

Set environment variables on your local workstation for the rest of the guide:

.. code-block:: shell

   export SERVER_BREV=nvflare-server-k8s
   export CLIENT_BREV=nvflare-site-1-k8s
   export NAMESPACE=nvflare
   export SERVER_HOST=server1.example.com
   export IMAGE=registry.example.com/nvflare:dev

Verify that you can SSH to both Brev Kubernetes environments:

.. code-block:: shell

   brev shell "$SERVER_BREV"
   exit
   brev shell "$CLIENT_BREV"
   exit

Inside each Brev Kubernetes environment, ``kubectl`` and ``helm`` should already
be configured for the local single-node cluster. You can verify this after SSH:

.. code-block:: shell

   kubectl get nodes
   kubectl get storageclass

Build and Push the FLARE Image
==============================

Build the FLARE runtime image from an NVFlare source checkout and push it to a
registry that both Brev Kubernetes clusters can pull from:

The ``ServerK8sJobLauncher`` and ``ClientK8sJobLauncher`` use the Kubernetes
Python client from inside the running FLARE container. If you use a custom
Dockerfile, install the dependency in the image:

.. code-block:: dockerfile

   RUN pip install kubernetes

The repository ``docker/Dockerfile`` already installs the NVFlare ``K8S`` extra,
which includes this dependency. Keep that install line, or add the explicit
``pip install kubernetes`` line above before building your image.

.. code-block:: shell

   docker build -t "$IMAGE" -f docker/Dockerfile .
   docker push "$IMAGE"

If the registry is private, make sure both clusters can pull the image. Depending
on your registry and cluster configuration, this can mean configuring node-level
registry credentials or adding Kubernetes image pull secrets. The generated
chart does not add ``imagePullSecrets`` by default, so use a registry already
trusted by the nodes or customize the chart for your environment.

Edit project.yml
================

Generate a sample project file if you do not already have one:

.. code-block:: shell

   nvflare provision -g

Edit ``project.yml`` with these deployment-specific goals:

#. Define only one client, ``site-1``.
#. Set the server ``default_host`` to the stable external DNS name that the
   client cluster will use.
#. Include the same DNS name in ``host_names`` so the server certificate is
   valid for that endpoint.
#. Leave ``admin_port`` unset so it defaults to ``fed_learn_port``. The Brev
   server only needs to expose the ``fed_learn_port`` value.
#. Add ``HelmChartBuilder`` after ``StaticFileBuilder``.
#. Use the container image that both clusters can pull.

Example:

.. code-block:: yaml

   api_version: 3
   name: example_project
   description: NVFlare Brev Kubernetes Helm deployment

   participants:
     - name: server1
       type: server
       org: nvidia
       default_host: server1.example.com
       host_names:
         - server1
         - server1.example.com
       fed_learn_port: 8002
     - name: site-1
       type: client
       org: nvidia
     - name: admin@nvidia.com
       type: admin
       org: nvidia
       role: project_admin

   builders:
     - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
       args:
         template_file:
           - master_template.yml
     - path: nvflare.lighter.impl.static_file.StaticFileBuilder
       args:
         config_folder: config
         scheme: tcp
     - path: nvflare.lighter.impl.helm_chart.HelmChartBuilder
       args:
         docker_image: registry.example.com/nvflare:dev
         parent_port: 8102
         workspace_pvc: nvflws
         etc_pvc: nvfletc
     - path: nvflare.lighter.impl.cert.CertBuilder
     - path: nvflare.lighter.impl.signature.SignatureBuilder

The value of ``default_host`` must be chosen before provisioning because it is
written into startup configuration and server certificates. Use a stable DNS
name that you control, such as ``server1.example.com``, in ``project.yml`` and
point that DNS name to the Brev server environment's exposed host after you
enable port access.

``HelmChartBuilder`` must appear after ``StaticFileBuilder``. During
provisioning, ``StaticFileBuilder.initialize()`` prepares communication config
state, ``HelmChartBuilder.build()`` updates it with Kubernetes Service names and
ports, and ``StaticFileBuilder.finalize()`` writes the final
``comm_config.json`` files.

Run Provisioning
================

Run the provision command:

.. code-block:: shell

   nvflare provision -p project.yml -w /tmp/nvflare/provision

Set ``PROD_DIR`` to the generated production folder:

.. code-block:: shell

   PROD_DIR=$(find /tmp/nvflare/provision/example_project -maxdepth 1 -type d -name 'prod_*' | sort | tail -n 1)
   echo "$PROD_DIR"

The generated folder should contain ``server1``, ``site-1``, the admin startup
kit, and one ``helm_chart`` directory under the server and client:

.. code-block:: shell

   ls "$PROD_DIR"
   ls "$PROD_DIR/server1/helm_chart"
   ls "$PROD_DIR/site-1/helm_chart"

Each participant folder has this structure:

.. code-block:: text

   server1/
     helm_chart/
       Chart.yaml
       values.yaml
       templates/
     local/
     startup/
     transfer/

Copy Startup Kits to Brev Environments
======================================

Package the provisioned server and client folders on your local workstation:

.. code-block:: shell

   tar -czf /tmp/nvflare-server1.tgz -C "$PROD_DIR" server1
   tar -czf /tmp/nvflare-site-1.tgz -C "$PROD_DIR" site-1

Use the ``Copy Files`` section of the Brev environment ``Access`` page, or run
the equivalent ``brev copy`` commands:

.. code-block:: shell

   brev copy /tmp/nvflare-server1.tgz "$SERVER_BREV:/home/ubuntu/"
   brev copy /tmp/nvflare-site-1.tgz "$CLIENT_BREV:/home/ubuntu/"

The archive contains the generated ``startup/``, ``local/``, and
``helm_chart/`` folders. The Helm chart is run from the Brev environment after
the archive is extracted.

Deploy the Server Environment
=============================

Open a shell on the server Brev environment:

.. code-block:: shell

   brev shell "$SERVER_BREV"

Run the rest of this section from inside the server environment. First extract
the uploaded archive and set deployment variables:

.. code-block:: shell

   export NAMESPACE=nvflare
   export IMAGE=registry.example.com/nvflare:dev

   mkdir -p ~/nvflare
   tar -xzf ~/nvflare-server1.tgz -C ~/nvflare
   kubectl get nodes
   helm version

Create the namespace and PVCs:

.. code-block:: shell

   kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

   cat > ~/nvflare/nvflare-pvcs.yaml <<'EOF'
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: nvflws
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 10Gi
   ---
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: nvfletc
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 1Gi
   ---
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: nvfldata
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 50Gi
   EOF

   kubectl -n "$NAMESPACE" apply -f ~/nvflare/nvflare-pvcs.yaml
   kubectl -n "$NAMESPACE" get pvc

If your Brev Kubernetes environment does not have a default storage class, add
``storageClassName: <storage-class-name>`` under each PVC ``spec``.

Patch the server launcher before copying the server folder into the workspace
PVC. If ``local/resources.json`` exists, edit that file. Otherwise, edit
``local/resources.json.default``. The generated startup kit normally uses
``ServerProcessJobLauncher``. For Kubernetes job pods, replace it with
``ServerK8sJobLauncher``.

Replace this server launcher path:

.. code-block:: text

   nvflare.app_common.job_launcher.server_process_launcher.ServerProcessJobLauncher

with this Kubernetes launcher path:

.. code-block:: text

   nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher

Set the server launcher ``args`` for this Brev Helm deployment:

.. code-block:: json

   {
     "id": "k8s_launcher",
     "path": "nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher",
     "args": {
       "config_file_path": null,
       "study_data_pvc_file_path": "/var/tmp/nvflare/workspace/local/study_data.yaml",
       "namespace": "nvflare",
       "python_path": "/usr/local/bin/python3",
       "pending_timeout": 300,
       "ephemeral_storage": "1Gi"
     }
   }

Replace the launcher component ``id`` with ``k8s_launcher`` when you update the
``path`` and ``args`` values.

The server K8s launcher args mean:

* ``config_file_path``: use ``null`` when the FL server pod launches job pods
  in the same Brev Kubernetes cluster. The launcher uses in-cluster Kubernetes
  credentials from the Helm-created service account.
* ``study_data_pvc_file_path``: path inside the FL server pod to the study data
  mapping file. This guide copies ``local/`` into
  ``/var/tmp/nvflare/workspace/local`` through the ``nvflws`` PVC.
* ``namespace``: Kubernetes namespace where launched job pods are created. Use
  the same namespace used by the server Helm release, ``nvflare`` in this guide.
* ``python_path``: Python executable inside the job container image.
* ``pending_timeout``: seconds to wait for a launched job pod to leave
  ``Pending`` before terminating it.
* ``ephemeral_storage``: temporary workspace size requested for each launched
  job pod.

Copy the server startup kit contents into the ``nvflws`` PVC. The chart starts
the server with ``-m /var/tmp/nvflare/workspace``, so the PVC root must contain
``startup/`` and ``local/`` directly.

.. code-block:: shell

   cat > ~/nvflare/copy-to-pvcs.yaml <<'EOF'
   apiVersion: v1
   kind: Pod
   metadata:
     name: nvflare-pvc-copy
   spec:
     restartPolicy: Never
     containers:
       - name: copy
         image: busybox:1.36
         command:
           - sh
           - -c
           - sleep 3600
         volumeMounts:
           - name: nvflws
             mountPath: /mnt/nvflws
           - name: nvfletc
             mountPath: /mnt/nvfletc
     volumes:
       - name: nvflws
         persistentVolumeClaim:
           claimName: nvflws
       - name: nvfletc
         persistentVolumeClaim:
           claimName: nvfletc
   EOF

   kubectl -n "$NAMESPACE" delete pod nvflare-pvc-copy --ignore-not-found=true
   kubectl -n "$NAMESPACE" apply -f ~/nvflare/copy-to-pvcs.yaml
   kubectl -n "$NAMESPACE" wait \
     --for=condition=Ready pod/nvflare-pvc-copy --timeout=120s
   kubectl -n "$NAMESPACE" cp ~/nvflare/server1/. nvflare-pvc-copy:/mnt/nvflws/
   kubectl -n "$NAMESPACE" exec nvflare-pvc-copy -- \
     ls -la /mnt/nvflws/startup /mnt/nvflws/local
   kubectl -n "$NAMESPACE" delete pod nvflare-pvc-copy

The trailing ``/.`` in ``~/nvflare/server1/.`` is intentional. It copies the
contents of the server folder into the PVC root. If the PVC root only contains a
nested ``server1/`` directory, the server pod will not find
``/var/tmp/nvflare/workspace/startup`` and
``/var/tmp/nvflare/workspace/local``.

Install the server Helm chart. Set ``hostPortEnabled=true`` so the server pod
binds ``fed_learn_port`` ``8002`` on the Brev host. This is the port exposed in
the Brev ``Using Ports`` UI.

.. code-block:: shell

   helm upgrade --install server1 ~/nvflare/server1/helm_chart \
     --namespace "$NAMESPACE" \
     --set image.repository="${IMAGE%:*}" \
     --set image.tag="${IMAGE##*:}" \
     --set service.type=ClusterIP \
     --set hostPortEnabled=true

   kubectl -n "$NAMESPACE" rollout status deployment/server1 --timeout=300s
   kubectl -n "$NAMESPACE" get pods
   kubectl -n "$NAMESPACE" logs deploy/server1

Deploy the site-1 Environment
=============================

Open a shell on the client Brev environment:

.. code-block:: shell

   brev shell "$CLIENT_BREV"

Run the rest of this section from inside the client environment. First extract
the uploaded archive and set deployment variables:

.. code-block:: shell

   export NAMESPACE=nvflare
   export IMAGE=registry.example.com/nvflare:dev
   export SERVER_HOST=server1.example.com

   mkdir -p ~/nvflare
   tar -xzf ~/nvflare-site-1.tgz -C ~/nvflare
   kubectl get nodes
   helm version

Create the namespace and PVCs:

.. code-block:: shell

   kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

   cat > ~/nvflare/nvflare-pvcs.yaml <<'EOF'
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: nvflws
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 10Gi
   ---
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: nvfletc
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 1Gi
   ---
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: nvfldata
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 50Gi
   EOF

   kubectl -n "$NAMESPACE" apply -f ~/nvflare/nvflare-pvcs.yaml
   kubectl -n "$NAMESPACE" get pvc

Patch the client launcher before copying the ``site-1`` folder into the
workspace PVC. If ``local/resources.json`` exists, edit that file. Otherwise,
edit ``local/resources.json.default``. The generated startup kit normally uses
``ClientProcessJobLauncher``. For Kubernetes job pods, replace it with
``ClientK8sJobLauncher``.

Replace this client launcher path:

.. code-block:: text

   nvflare.app_common.job_launcher.client_process_launcher.ClientProcessJobLauncher

with this Kubernetes launcher path:

.. code-block:: text

   nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher

Set the client launcher ``args`` for this Brev Helm deployment:

.. code-block:: json

   {
     "id": "k8s_launcher",
     "path": "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
     "args": {
       "config_file_path": null,
       "study_data_pvc_file_path": "/var/tmp/nvflare/workspace/local/study_data.yaml",
       "namespace": "nvflare",
       "python_path": "/usr/local/bin/python3",
       "pending_timeout": 300,
       "ephemeral_storage": "1Gi"
     }
   }

Replace the launcher component ``id`` with ``k8s_launcher`` when you update the
``path`` and ``args`` values.

The client K8s launcher args mean:

* ``config_file_path``: use ``null`` when the FL client pod launches job pods in
  the same Brev Kubernetes cluster. The launcher uses in-cluster Kubernetes
  credentials from the Helm-created service account.
* ``study_data_pvc_file_path``: path inside the FL client pod to the study data
  mapping file. This guide copies ``local/`` into
  ``/var/tmp/nvflare/workspace/local`` through the ``nvflws`` PVC.
* ``namespace``: Kubernetes namespace where launched job pods are created. Use
  the same namespace used by the client Helm release, ``nvflare`` in this guide.
* ``python_path``: Python executable inside the job container image.
* ``pending_timeout``: seconds to wait for a launched job pod to leave
  ``Pending`` before terminating it.
* ``ephemeral_storage``: temporary workspace size requested for each launched
  job pod.

Copy the ``site-1`` startup kit contents into the client ``nvflws`` PVC:

.. code-block:: shell

   cat > ~/nvflare/copy-to-pvcs.yaml <<'EOF'
   apiVersion: v1
   kind: Pod
   metadata:
     name: nvflare-pvc-copy
   spec:
     restartPolicy: Never
     containers:
       - name: copy
         image: busybox:1.36
         command:
           - sh
           - -c
           - sleep 3600
         volumeMounts:
           - name: nvflws
             mountPath: /mnt/nvflws
           - name: nvfletc
             mountPath: /mnt/nvfletc
     volumes:
       - name: nvflws
         persistentVolumeClaim:
           claimName: nvflws
       - name: nvfletc
         persistentVolumeClaim:
           claimName: nvfletc
   EOF

   kubectl -n "$NAMESPACE" delete pod nvflare-pvc-copy --ignore-not-found=true
   kubectl -n "$NAMESPACE" apply -f ~/nvflare/copy-to-pvcs.yaml
   kubectl -n "$NAMESPACE" wait \
     --for=condition=Ready pod/nvflare-pvc-copy --timeout=120s
   kubectl -n "$NAMESPACE" cp ~/nvflare/site-1/. nvflare-pvc-copy:/mnt/nvflws/
   kubectl -n "$NAMESPACE" exec nvflare-pvc-copy -- \
     ls -la /mnt/nvflws/startup /mnt/nvflws/local
   kubectl -n "$NAMESPACE" delete pod nvflare-pvc-copy

Before installing the client chart, verify that the client environment can
resolve the server host:

.. code-block:: shell

   kubectl -n "$NAMESPACE" run dns-test --rm -it \
     --image=busybox:1.36 -- \
     nslookup "$SERVER_HOST"

Install the ``site-1`` Helm chart:

.. code-block:: shell

   helm upgrade --install site-1 ~/nvflare/site-1/helm_chart \
     --namespace "$NAMESPACE" \
     --set image.repository="${IMAGE%:*}" \
     --set image.tag="${IMAGE##*:}"

   kubectl -n "$NAMESPACE" rollout status deployment/site-1 --timeout=300s
   kubectl -n "$NAMESPACE" get pods
   kubectl -n "$NAMESPACE" logs deploy/site-1

If you reprovision later, back up or remove old PVC contents before copying the
new folders. Certificates, local config, and communication settings are tied to
the provisioned project state.

Connect an Admin Console
========================

Run the admin client from a network location that can reach
``server1.example.com:8002``:

.. code-block:: shell

   cd "$PROD_DIR/admin@nvidia.com/startup"
   ./fl_admin.sh

The generated admin kit connects to the server host configured in
``project.yml``. If you used ``server1.example.com`` as ``default_host``, that
name must resolve to the Brev server environment endpoint.

Kubernetes Job Pods and nvfldata
================================

The server and client deployment sections tell you to patch
``local/resources.json`` or ``local/resources.json.default`` before copying each
startup kit into ``nvflws``. Those patched files replace the process launcher
with the Kubernetes launcher and set ``study_data_pvc_file_path`` to:

.. code-block:: text

   /var/tmp/nvflare/workspace/local/study_data.yaml

When launched job pods need the ``nvfldata`` PVC, create
``local/study_data.yaml`` in both the server and client folders before copying
those folders into ``nvflws``. This example maps the ``default`` study's
``data`` dataset to ``nvfldata``:

.. code-block:: yaml

   default:
     data:
       source: nvfldata
       mode: rw

Job pod images must also be specified in the submitted job's ``meta.json``
``launcher_spec`` or ``resource_spec`` for the ``k8s`` launcher.

Troubleshooting
===============

PVC stays ``Pending``
---------------------

Check that the Brev cluster has a default storage class, or add an explicit
``storageClassName`` to ``nvflare-pvcs.yaml``:

.. code-block:: shell

   kubectl get storageclass
   kubectl -n "$NAMESPACE" describe pvc nvflws

Pod has ``ImagePullBackOff``
----------------------------

Confirm the image exists and that both clusters can pull it:

.. code-block:: shell

   docker push "$IMAGE"
   kubectl -n "$NAMESPACE" describe pod -l app.kubernetes.io/name=server1
   kubectl -n "$NAMESPACE" describe pod -l app.kubernetes.io/name=site-1

Server pod cannot find ``startup`` or ``local``
-----------------------------------------------

The participant folder was copied to the wrong level in the PVC. The server
workspace root must contain:

.. code-block:: text

   /var/tmp/nvflare/workspace/startup
   /var/tmp/nvflare/workspace/local

Use the helper pod to inspect ``/mnt/nvflws`` and restage the contents of
``$PROD_DIR/server1/.`` if needed.

site-1 cannot connect to the server
-----------------------------------

Verify these items:

* ``default_host`` in ``project.yml`` matches the DNS name used by the client.
* The DNS name resolves from the client cluster.
* The server cluster exposes TCP port ``8002``.
* The server certificate includes the DNS name in ``host_names``.

Run a DNS check from the client cluster:

.. code-block:: shell

   kubectl -n "$NAMESPACE" run dns-test --rm -it \
     --image=busybox:1.36 -- \
     nslookup "$SERVER_HOST"

If you change ``default_host`` or ``host_names``, reprovision, restage the
updated folders, and redeploy the charts.

Cleanup
========

Remove the Helm releases:

.. code-block:: shell

   # Run inside the server Brev environment.
   helm uninstall server1 -n "$NAMESPACE"

   # Run inside the site-1 Brev environment.
   helm uninstall site-1 -n "$NAMESPACE"

Delete the namespaces and PVCs:

.. code-block:: shell

   # Run inside each Brev environment.
   kubectl delete namespace "$NAMESPACE"

Delete the Brev clusters from the web UI when you no longer need them:

#. Open the Brev console.
#. Open the Kubernetes or clusters page.
#. Select ``nvflare-server-k8s`` and delete it.
#. Select ``nvflare-site-1-k8s`` and delete it.
#. Confirm in the billing or usage page that the resources are no longer
   running.
