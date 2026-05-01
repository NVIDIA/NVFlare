.. _helm_chart:

###########################
Running FLARE in Kubernetes
###########################

NVIDIA FLARE can be deployed to Kubernetes by first provisioning normal startup
kits and then preparing each kit for the Kubernetes runtime.

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

The prepared kit contains a ``helm_chart`` directory that can be installed with
Helm:

.. code-block:: bash

   helm upgrade --install server server-k8s/helm_chart --namespace nvflare --create-namespace

Create and bind any workspace or study-data PVCs required by your cluster before
starting the participant. Study data mappings are configured in
``local/study_data.yaml`` inside the prepared kit.
