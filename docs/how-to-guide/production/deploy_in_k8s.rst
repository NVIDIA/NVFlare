.. _deploy_nvflare_in_k8s:

################################
How to Deploy NVFLARE in K8s
################################

This guide covers deploying NVIDIA FLARE on Kubernetes clusters using Helm Charts.

.. note::
  NVIDIA FLARE cloud-native support will release soon.

Prerequisites
=============

Before deploying to Kubernetes, ensure you have:

- A running Kubernetes cluster (microk8s, k3s, or managed Kubernetes)
- ``kubectl`` installed and configured to access your cluster
- Helm 3 installed
- A container registry accessible from your cluster
- NVIDIA FLARE installed locally (``pip install nvflare``)

For microk8s, ensure the following addons are enabled:

.. code-block:: shell

    microk8s enable dns helm3 hostpath-storage ingress registry


Provisioning with Helm Chart
============================

NVIDIA FLARE includes a ``HelmChartBuilder`` that generates Helm Charts during provisioning.

Configure project.yml
---------------------

Add the ``HelmChartBuilder`` to your ``project.yml`` file:

.. code-block:: yaml

    builders:
      - path: nvflare.lighter.impl.helm_chart.HelmChartBuilder
        args:
          docker_image: localhost:32000/nvfl-min:0.0.1

The ``docker_image`` specifies the container image for all pods. You must build and push
this image to a registry accessible by your cluster.

Run Provisioning
----------------

Generate the startup kits and Helm Chart:

.. code-block:: shell

    nvflare provision

This creates:

.. code-block:: text

    workspace/<project_name>/prod_00/
    ├── admin@nvidia.com
    ├── nvflare_hc          # Helm Chart package
    ├── server1
    ├── server2
    ├── site-1
    └── site-2

The ``nvflare_hc`` folder contains the generated Helm Chart.

.. note::

    The generated Helm Chart is a reference starting point. Depending on your Kubernetes
    cluster configuration, you may need to modify it for your environment.

For detailed provisioning options, see :ref:`provisioning`.


Prepare Container Image
=======================

Build and push a Docker image containing NVIDIA FLARE and your dependencies:

.. code-block:: shell

    # Example Dockerfile
    FROM python:3.10-slim
    RUN pip install nvflare
    # Add any additional dependencies

    # Build and push to local registry (for microk8s)
    docker build -t localhost:32000/nvfl-min:0.0.1 .
    docker push localhost:32000/nvfl-min:0.0.1


Configure Network Access
========================

Configure your cluster to allow incoming network traffic from clients and admin consoles.

For microk8s
------------

Edit the ingress configmap to route traffic:

.. code-block:: shell

    microk8s kubectl edit cm nginx-ingress-tcp-microk8s-conf -n ingress

Add the following data section:

.. code-block:: yaml

    data:
      "8002": default/server1:8002
      "8003": default/server1:8003

Edit the ingress DaemonSet to open ports:

.. code-block:: shell

    microk8s kubectl edit ds nginx-ingress-microk8s-controller -n ingress

Add port mappings under ``spec.template.spec.containers[0].ports``:

.. code-block:: yaml

    - containerPort: 8002
      hostPort: 8002
      name: server1fl
      protocol: TCP
    - containerPort: 8003
      hostPort: 8003
      name: server1adm
      protocol: TCP


Deploy with Helm
================

Install the Helm Chart
----------------------

Navigate to the provisioning output directory and install:

.. code-block:: shell

    # Create persistent storage directory
    mkdir -p /tmp/nvflare

    # Install the chart (for microk8s)
    microk8s helm3 install \
        --set workspace=$(pwd) \
        --set svc-persist=/tmp/nvflare \
        nvflare-deployment nvflare_hc/

For standard Kubernetes:

.. code-block:: shell

    helm install \
        --set workspace=$(pwd) \
        --set svc-persist=/tmp/nvflare \
        nvflare-deployment nvflare_hc/

Upon successful deployment:

.. code-block:: text

    NAME: nvflare-deployment
    LAST DEPLOYED: Fri Sep 23 12:28:24 2022
    NAMESPACE: default
    STATUS: deployed
    REVISION: 1


Verify Deployment
-----------------

Check that pods are running:

.. code-block:: shell

    kubectl get pods

Expected output:

.. code-block:: text

    NAME                        READY   STATUS    RESTARTS   AGE
    server1-7675668544-xvfvp    1/1     Running   0          4m50s
    server2-86bc4fc87f-s9n2s    1/1     Running   0          4m50s

For detailed pod information:

.. code-block:: shell

    kubectl describe pods


Uninstall the Chart
-------------------

To remove the deployment:

.. code-block:: shell

    # For microk8s
    microk8s helm3 uninstall nvflare-deployment

    # For standard Kubernetes
    helm uninstall nvflare-deployment


Connect Clients and Admin
=========================

After servers are running in the cluster, connect external clients and admin consoles.

Configure DNS
-------------

Update ``/etc/hosts`` on client machines to point server names to the cluster IP:

.. code-block:: text

    192.168.1.123 server1 server2

Replace ``192.168.1.123`` with your cluster's external IP address.

Start Clients
-------------

On each client machine, run the client startup script:

.. code-block:: shell

    ./startup/start.sh

Start Admin Console
-------------------

Launch the admin console:

.. code-block:: shell

    ./startup/fl_admin.sh

Login with your admin credentials (e.g., ``admin@nvidia.com``).


Post-Deployment Verification
============================

After deployment, verify the system is running correctly.

Preflight Check
---------------

After the FL system starts but before running jobs, use the NVIDIA FLARE preflight check
to verify connectivity.

**Check Clients**: On each client machine, run:

.. code-block:: shell

    nvflare preflight_check -p /path/to/client_startup_kit

**Check Admin Console**: Verify the admin console can connect:

.. code-block:: shell

    nvflare preflight_check -p /path/to/admin_startup_kit

For detailed information, see :ref:`preflight_check`.

Check Status via Admin Console
------------------------------

After connecting with the admin console, check server status:

.. code-block:: text

    > check_status server


Security Considerations
=======================

- Use Kubernetes Network Policies to restrict pod-to-pod communication
- Configure TLS for all external traffic
- Use Kubernetes Secrets for sensitive configuration
- Enable RBAC for cluster access control
- Consider using a service mesh (e.g., Istio) for enhanced security
- Regularly update container images with security patches


Troubleshooting
===============

**Pods not starting**

Check pod events and logs:

.. code-block:: shell

    kubectl describe pod <pod-name>
    kubectl logs <pod-name>

**Clients cannot connect to server**

- Verify ingress configuration is correct
- Check that ports are open in the ingress controller
- Ensure DNS/hosts file is configured correctly on client machines

**Image pull errors**

Verify the container image is accessible from the cluster:

.. code-block:: shell

    kubectl describe pod <pod-name> | grep -A5 "Events"

**Persistent volume issues**

Ensure the storage directory exists and has correct permissions:

.. code-block:: shell

    mkdir -p /tmp/nvflare
    chmod 777 /tmp/nvflare


References
==========

**FLARE Documentation**

- :ref:`helm_chart` - Detailed Helm Chart documentation
- :ref:`preflight_check` - Preflight check tool documentation
- :ref:`provisioning` - Provisioning and startup kit generation
- :ref:`flare_security_overview` - Security architecture overview

**Kubernetes Installation Guides**

- `Kubernetes (k8s) <https://kubernetes.io/docs/setup/>`_ - Official Kubernetes installation guide
- `MicroK8s <https://microk8s.io/docs/getting-started>`_ - Lightweight Kubernetes for development and edge
- `K3s <https://docs.k3s.io/quick-start>`_ - Lightweight Kubernetes for resource-constrained environments
