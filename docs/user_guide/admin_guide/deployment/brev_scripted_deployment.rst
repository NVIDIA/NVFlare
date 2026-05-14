.. _brev_scripted_deployment:

###################################
Brev Scripted Deployment Quickstart
###################################

This guide shows the shortest path for using the Brev helper scripts to deploy
one FLARE server and two FLARE clients on three existing Brev single-node
Kubernetes environments.

For the full technical workflow, including ``project.yml`` details, PVC staging,
Helm chart behavior, and troubleshooting, see :ref:`brev_deployment`.

Scripts:

* :download:`prepare_brev_startup_kits.sh <brev_scripts/prepare_brev_startup_kits.sh>`
* :download:`launch_brev_nvflare.sh <brev_scripts/launch_brev_nvflare.sh>`

What the Scripts Do
===================

``prepare_brev_startup_kits.sh`` runs on your local workstation. It:

* creates a simple three-participant ``project.yml`` unless you provide one;
* runs ``nvflare provision``;
* runs ``nvflare deploy prepare`` for K8s on the generated ``server``,
  ``site-1``, and ``site-2`` startup kits;
* packages the prepared participant kits;
* copies the matching archive and launch script to each Brev environment with
  ``brev copy``.

``launch_brev_nvflare.sh`` runs inside each Brev environment. It:

* extracts the copied participant archive;
* verifies that ``nvflare deploy prepare`` configured the Kubernetes launcher
  (the expected ``ServerK8sJobLauncher`` or ``ClientK8sJobLauncher`` component
  with the same ``namespace`` you launch into);
* reads ``persistence.workspace.claimName`` from the prepared chart and rejects
  any conflicting launch-time ``WORKSPACE_PVC``;
* creates the ``nvflare`` namespace and the workspace/data PVCs;
* copies the prepared ``startup/`` and ``local/`` directories into the
  workspace PVC root so the chart finds them at ``workspace_mount_path``;
* runs a DNS lookup for ``SERVER_HOST`` from a temporary in-namespace pod for
  client participants;
* installs or upgrades the generated Helm chart;
* waits for the participant deployment and prints recent pod logs.

Expected Results
================

After the prepare script finishes, each Brev environment should contain:

* ``/home/ubuntu/nvflare-server.tgz``, ``/home/ubuntu/nvflare-site-1.tgz``, or
  ``/home/ubuntu/nvflare-site-2.tgz``;
* ``/home/ubuntu/launch_brev_nvflare.sh``.

After the launch script finishes in each environment:

* ``kubectl -n nvflare get pvc`` should show the PVCs as ``Bound``;
* ``helm list -n nvflare`` should show one release for that participant;
* ``kubectl -n nvflare get pods`` should show the participant pod running;
* the ``site-1`` and ``site-2`` logs should show that they are connecting to
  the server endpoint.

What You Need
=============

Before running the scripts, have:

* three running Brev Kubernetes environments for ``server``, ``site-1``, and
  ``site-2``;
* a DNS name or IP for the server that both sites can reach;
* an NVFlare container image that all three environments can pull;
* the Brev CLI installed and logged in on your local workstation.

Prepare and Copy Kits
=====================

From your local NVFlare checkout, set the server host and image, then run the
prepare script:

.. code-block:: shell

   export SERVER_HOST=server1.example.com
   export IMAGE=registry.example.com/nvflare:dev

   bash docs/user_guide/admin_guide/deployment/brev_scripts/prepare_brev_startup_kits.sh

By default, the script copies kits to Brev environments named ``server``,
``site-1``, and ``site-2``.

If your Brev environment names are different, set them before running:

.. code-block:: shell

   export SERVER_BREV=nvflare-server-k8s
   export SITE_1_BREV=nvflare-site-1-k8s
   export SITE_2_BREV=nvflare-site-2-k8s

   bash docs/user_guide/admin_guide/deployment/brev_scripts/prepare_brev_startup_kits.sh

Or let the script prompt for the Brev environment names:

.. code-block:: shell

   bash docs/user_guide/admin_guide/deployment/brev_scripts/prepare_brev_startup_kits.sh \
     --prompt-brev-names

Expose the Server Port
======================

In the Brev console, open the server environment's Access page and expose TCP
port ``8002``. Make sure ``SERVER_HOST`` resolves to that exposed server host.
Do not include ``:8002`` in ``SERVER_HOST``.

Launch Each Environment
=======================

Run the launch script inside each Brev environment.

Server:

.. code-block:: shell

   brev shell "${SERVER_BREV:-server}"
   IMAGE="$IMAGE" bash /home/ubuntu/launch_brev_nvflare.sh server

Site 1:

.. code-block:: shell

   brev shell "${SITE_1_BREV:-site-1}"
   IMAGE="$IMAGE" SERVER_HOST="$SERVER_HOST" bash /home/ubuntu/launch_brev_nvflare.sh site-1

Site 2:

.. code-block:: shell

   brev shell "${SITE_2_BREV:-site-2}"
   IMAGE="$IMAGE" SERVER_HOST="$SERVER_HOST" bash /home/ubuntu/launch_brev_nvflare.sh site-2

The launch script verifies the prepared Kubernetes launcher config and the
prepared chart's workspace PVC name, creates the namespace and PVCs, stages
``startup/`` and ``local/`` into the workspace PVC root so the chart finds them
at ``workspace_mount_path``, installs the Helm chart, and prints recent pod
logs.

Useful Overrides
================

Common optional environment variables:

``NAMESPACE``, ``PARENT_PORT``, ``WORKSPACE_PVC``, and
``WORKSPACE_MOUNT_PATH`` are deploy-prepare inputs that are embedded into the
prepared resources and ``helm_chart/``. Set them when running
``prepare_brev_startup_kits.sh``. Do not change them only at launch time; rerun
the prepare script first so the prepared kit matches the launch environment.

* ``NAMESPACE``: Kubernetes namespace, default ``nvflare``. Use the same value
  when running both scripts. The launch script reads the prepared
  ``resources.json.default`` and refuses to launch if the launch-time
  ``NAMESPACE`` disagrees with the prepared launcher.
* ``DATA_PVC``: optional job data PVC name, default ``nvfldata``. Used only at
  launch time to create an additional PVC for study data.
* ``CLEAN_WORKSPACE_PVC=true``: clear old workspace PVC contents before staging
  a new kit.

Prepare-only overrides:

* ``WORKSPACE_PVC``: workspace PVC name, default ``nvflws``. The prepare
  script writes this into the generated Helm chart's
  ``persistence.workspace.claimName``. The launch script reads that value
  from the prepared chart and uses it for PVC creation and staging. If you
  also set ``WORKSPACE_PVC`` at launch and it disagrees with the prepared
  chart, the launch script fails so the chart, PVC, and staged content stay
  consistent.
* ``PARENT_PORT``: in-cluster parent service port written by
  ``nvflare deploy prepare``, default ``8102``. The launch script reads the
  prepared Helm chart and launcher config; it does not read this variable.
* ``WORKSPACE_MOUNT_PATH``: parent and job pod workspace path written by
  ``nvflare deploy prepare``, default ``/var/tmp/nvflare/workspace``. The
  launch script verifies the prepared value from ``resources.json.default``; it
  does not read this variable.

Run either script with ``--help`` for the current options.
