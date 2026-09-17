.. _openshift_k8s_deployment:

##############################
Deploying FLARE on OpenShift
##############################

The OpenShift deployment guide and helper scripts now live in the DevOps
examples directory:

``examples/devops/openshift``

Where to Start
==============

Open ``examples/devops/openshift/README.md`` first for a concise folder
overview. It lists the Dockerfiles, helper scripts, and typical quickstart
commands.

Open ``examples/devops/openshift/index.md`` for the full OpenShift deployment
guide. That source document covers prerequisites, image requirements, the
scripted workflow, manual deployment steps, OpenShift SCC notes,
troubleshooting, and cleanup.

Run the scripts from the NVFlare repository root. For example:

.. code-block:: bash

   bash examples/devops/openshift/scripts/k8s_e2e.sh

The OpenShift example builds on the generic Kubernetes deployment runtime. See
:ref:`helm_chart` for the Kubernetes Helm chart workflow and runtime details.
