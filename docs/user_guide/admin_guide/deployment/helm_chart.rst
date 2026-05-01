.. _helm_chart:

###########################
Running FLARE in Kubernetes
###########################

NVIDIA FLARE can be deployed to Kubernetes by first provisioning normal startup
kits and then preparing each kit for the Kubernetes runtime.

Prepare Startup Kits
====================

The provisioning step remains responsible for identity material and FLARE
configuration:

.. code-block:: bash

   nvflare provision -p project.yml -w workspace

After provisioning, prepare each server or client startup kit with
``nvflare deploy prepare``:

.. code-block:: bash

   nvflare deploy prepare workspace/<project>/prod_00/server \
       --output server-k8s \
       --config k8s.yaml

Example ``k8s.yaml``:

.. code-block:: yaml

   runtime: k8s
   namespace: nvflare
   parent:
     docker_image: registry.example.com/nvflare:dev
     parent_port: 8102
     workspace_pvc: nvflws
     workspace_mount_path: /var/tmp/nvflare/workspace
   job_launcher:
     config_file_path:
     default_python_path: /usr/local/bin/python3
     pending_timeout: 300

Prepare Cluster Storage
=======================

Create and bind any workspace or study-data PVCs required by your cluster before
starting the participant. The generated chart mounts ``parent.workspace_pvc`` at
``parent.workspace_mount_path``, but it does not upload files to the PVC. Copy the
prepared kit's ``startup`` and ``local`` directories into the root of that
workspace PVC before installing the chart.

For example, with a workspace PVC named ``nvflws``:

.. code-block:: bash

   export NAMESPACE=nvflare
   export PREPARED_KIT=server-k8s
   export WORKSPACE_PVC=nvflws

   kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
   kubectl -n "$NAMESPACE" apply -f workspace-pvc.yaml
   kubectl -n "$NAMESPACE" get pvc "$WORKSPACE_PVC"

   kubectl -n "$NAMESPACE" delete pod nvflare-pvc-copy --ignore-not-found=true
   cat >/tmp/nvflare-pvc-copy.json <<EOF
   {
     "spec": {
       "volumes": [
         {"name": "ws", "persistentVolumeClaim": {"claimName": "${WORKSPACE_PVC}"}}
       ],
       "containers": [
         {
           "name": "nvflare-pvc-copy",
           "image": "busybox:1.36",
           "command": ["sleep", "600"],
           "volumeMounts": [{"name": "ws", "mountPath": "/mnt/nvflws"}]
         }
       ]
     }
   }
   EOF
   kubectl -n "$NAMESPACE" run nvflare-pvc-copy \
       --image=busybox:1.36 \
       --restart=Never \
       --overrides="$(cat /tmp/nvflare-pvc-copy.json)"
   kubectl -n "$NAMESPACE" wait --for=condition=Ready pod/nvflare-pvc-copy --timeout=120s
   kubectl -n "$NAMESPACE" exec nvflare-pvc-copy -- rm -rf /mnt/nvflws/startup /mnt/nvflws/local
   kubectl -n "$NAMESPACE" cp "$PREPARED_KIT/startup" nvflare-pvc-copy:/mnt/nvflws/startup
   kubectl -n "$NAMESPACE" cp "$PREPARED_KIT/local" nvflare-pvc-copy:/mnt/nvflws/local
   kubectl -n "$NAMESPACE" delete pod nvflare-pvc-copy

Study data mappings are configured in ``local/study_data.yaml`` inside the
prepared kit. Create the matching study-data PVCs in the same namespace before
submitting jobs that need those mounts.

Install The Chart
=================

The prepared kit contains a ``helm_chart`` directory that can be installed with
Helm:

.. code-block:: bash

   helm upgrade --install server server-k8s/helm_chart \
       --namespace "$NAMESPACE" \
       --create-namespace

Prepare, stage, and install each server or client kit in the Kubernetes cluster
where that participant runs. For a scripted Brev example that performs these
steps end to end, see :ref:`brev_deployment`.

Expose FL Traffic
=================

The generated server chart creates a Kubernetes Service for the FL server. The
service defaults to ``ClusterIP``, which is reachable only inside the cluster.
If clients or admin consoles connect from outside the cluster, expose the FL
server ports with the mechanism that matches your Kubernetes environment:

* Use a cloud load balancer when available:

  .. code-block:: bash

     helm upgrade --install server server-k8s/helm_chart \
         --namespace "$NAMESPACE" \
         --set service.type=LoadBalancer
     kubectl -n "$NAMESPACE" get svc nvflare-server

* For local testing from the same machine, use port forwarding:

  .. code-block:: bash

     kubectl -n "$NAMESPACE" port-forward svc/nvflare-server 8002:8002 8003:8003

* For single-node or ingress-based clusters, configure your cluster's TCP
  routing, firewall rules, or host ports so the FL and admin ports from
  ``project.yml`` reach the ``nvflare-server`` Service.

Make sure the server host name used during provisioning resolves to the exposed
address. For example, update DNS or ``/etc/hosts`` for the admin console and for
any remote client sites.

Verify The Deployment
=====================

After installing the chart, verify that the deployment, pods, services, and PVCs
are healthy:

.. code-block:: bash

   kubectl -n "$NAMESPACE" rollout status deployment/server --timeout=300s
   kubectl -n "$NAMESPACE" get pods,svc,pvc
   kubectl -n "$NAMESPACE" logs deploy/server --tail=200

If a pod is not ready, inspect the pod and recent events:

.. code-block:: bash

   kubectl -n "$NAMESPACE" describe pod -l app.kubernetes.io/instance=server
   kubectl -n "$NAMESPACE" get events --sort-by=.lastTimestamp

Common issues are missing PVCs, the prepared kit not being copied into the
workspace PVC, an image that the cluster cannot pull, or FL ports that are not
reachable from clients and admin consoles.

Login With The Admin Console
============================

Use the admin startup kit produced by ``nvflare provision``. The admin console
connects to the server host and ports written into the provisioned project, so
confirm that those names resolve to the exposed Kubernetes endpoint before
logging in.

.. code-block:: bash

   cd workspace/<project>/prod_00/admin@nvidia.com/startup
   bash fl_admin.sh

When prompted for ``User Name``, enter the admin identity from ``project.yml``
such as ``admin@nvidia.com``.

Uninstall
=========

To stop a participant installed by Helm:

.. code-block:: bash

   helm uninstall server -n "$NAMESPACE"

Delete the namespace only if it is dedicated to this deployment:

.. code-block:: bash

   kubectl delete namespace "$NAMESPACE"

Depending on the storage class reclaim policy, PVC-backed volumes may remain
after deleting Helm releases or namespaces. Remove retained volumes only after
confirming that the startup kits, logs, and any study data no longer need to be
preserved.
