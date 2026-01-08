.. _setup_nvflare_in_production_guide:

##################
How to Setup FLARE
##################

This guide covers the complete process of setting up NVIDIA FLARE for production deployment,
from installation to running your first federated learning job.

Overview
========

Setting up FLARE for production involves four main phases:

1. **Provision** - Generate startup kits for server, clients, and admin users
2. **Distribute** - Deliver startup kits to participating sites
3. **Start** - Launch the FL server and clients
4. **Operate** - Run federated learning jobs


Prerequisites
=============

Before setting up FLARE, ensure you have:

- Python 3.8 or higher
- Network connectivity between server and client sites
- Firewall rules allowing communication on required ports (default: 8002, 8003)
- DNS or hosts file configuration for hostname resolution

Install NVIDIA FLARE
--------------------

Install FLARE on all machines (server, clients, admin):

.. code-block:: shell

    pip install nvflare

For admin console with additional features:

.. code-block:: shell

    pip install nvflare[apt_opt]


Phase 1: Provisioning
=====================

Provisioning generates secure startup kits containing certificates and configurations
for all participants in your FL project.

Option A: Using FLARE CLI
-------------------------

Create a ``project.yml`` configuration file:

.. code-block:: yaml

    api_version: 3
    name: my_fl_project
    description: My Federated Learning Project

    participants:
      - name: server
        type: server
        org: nvidia
        cn: server.example.com

      - name: site-1
        type: client
        org: org1

      - name: site-2
        type: client
        org: org2

      - name: admin@nvidia.com
        type: admin
        org: nvidia
        role: project_admin

    builders:
      - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
        args:
          template_file: master_template.yml
      - path: nvflare.lighter.impl.template.TemplateBuilder
      - path: nvflare.lighter.impl.static_file.StaticFileBuilder
        args:
          config_folder: config
      - path: nvflare.lighter.impl.cert.CertBuilder
      - path: nvflare.lighter.impl.signature.SignatureBuilder

Run the provisioning command:

.. code-block:: shell

    nvflare provision -p project.yml

This generates startup kits in ``workspace/<project_name>/prod_00/``:

.. code-block:: text

    workspace/my_fl_project/prod_00/
    ├── server/
    ├── site-1/
    ├── site-2/
    └── admin@nvidia.com/

Option B: Using FLARE Dashboard
-------------------------------

For easier distribution, use the FLARE Dashboard web UI:

.. code-block:: shell

    nvflare dashboard --start

The Dashboard provides:

- Web-based project configuration
- Participant invitation and approval
- On-demand startup kit downloads

For Dashboard details, see :ref:`nvflare_dashboard_ui`.


Phase 2: Distribution
=====================

Distribute startup kits to each participating site securely.

CLI Provisioning
----------------

When using CLI provisioning, manually distribute packages via:

- Secure file transfer (SFTP, SCP)
- Encrypted email
- Secure shared storage

.. attention::

    Startup kits contain private keys and certificates. Handle with appropriate security measures.

Dashboard Provisioning
----------------------

When using the Dashboard, participants download their own startup kits directly
from the web interface after approval.


Phase 3: Starting the System
============================

Start the FL server first, then clients, and finally connect with admin console.

Start FL Server
---------------

On the server machine, navigate to the startup directory:

.. code-block:: shell

    cd /path/to/server/startup
    ./start.sh

The server will start and wait for client connections.

.. note::

    The FL server must bind to the hostname specified during provisioning.
    Ensure DNS or ``/etc/hosts`` resolves the hostname correctly.

Start FL Clients
----------------

On each client machine:

.. code-block:: shell

    cd /path/to/site-1/startup
    ./start.sh

Upon successful connection, you'll see confirmation in both server and client logs:

**Server log:**

.. code-block:: text

    Client: New client site-1@192.168.1.100 joined. Sent token: f279157b-df8c-aa1b-8560-2c43efa257bc

**Client log:**

.. code-block:: text

    Successfully registered client:site-1. Got token:f279157b-df8c-aa1b-8560-2c43efa257bc

Start Admin Console
-------------------

Launch the admin console to manage the FL system:

.. code-block:: shell

    cd /path/to/admin@nvidia.com/startup
    ./fl_admin.sh

Enter your admin email when prompted (e.g., ``admin@nvidia.com``).


Phase 4: Verification and Operation
===================================

Preflight Check
---------------

Before running jobs, verify the system configuration:

.. code-block:: shell

    # On server
    nvflare preflight_check -p /path/to/server

    # On clients (after server is running)
    nvflare preflight_check -p /path/to/site-1

    # On admin
    nvflare preflight_check -p /path/to/admin@nvidia.com

For details, see :ref:`preflight_check`.

Check System Status
-------------------

From the admin console, verify all components are connected:

.. code-block:: text

    > check_status server

This displays server status and connected clients.


Workload Deployment
===================

Deploy training code to clients using one of two methods:

Dynamic Code (BYOC)
-------------------

For development and POC, use Bring Your Own Code (BYOC):

- Include custom code in the job's ``custom/`` folder
- Code is automatically deployed when the job is submitted
- Suitable for experimentation

Pre-deployed Code
-----------------

For production with strict security requirements:

.. code-block:: shell

    nvflare pre-install -j /path/to/job

This pre-installs workload code on all sites before job execution.

.. note::

    Ensure all sites have required dependencies installed (PyTorch, TensorFlow, etc.)
    as FLARE does not manage deep learning framework installation.


Running Jobs
============

Submit jobs using the admin console or FLARE API.

Via Admin Console
-----------------

.. code-block:: text

    > submit_job /path/to/job

Via FLARE API
-------------

For programmatic control from Python or Jupyter notebooks:

.. code-block:: python

    from nvflare.fuel.flare_api.flare_api import new_secure_session

    sess = new_secure_session("admin@nvidia.com", "/path/to/admin/startup")
    sess.submit_job("/path/to/job")

See :ref:`flare_api` for details.


Docker Deployment
=================

For containerized deployments, add the Docker builder to ``project.yml``:

.. code-block:: yaml

    builders:
      - path: nvflare.lighter.impl.docker.DockerBuilder
        args:
          docker_image: nvflare/nvflare:latest

This generates ``docker.sh`` scripts for each component:

.. code-block:: shell

    cd /path/to/server/startup
    ./docker.sh

.. note::

    When running the server in Docker, use ``--net=host`` to properly map hostnames.


Security Best Practices
=======================

- **Protect private keys**: The ``client.key`` file in each startup kit is sensitive
- **Use HTTPS for Dashboard**: Place ``web.crt`` and ``web.key`` in the working directory
- **Rotate certificates**: Periodically re-provision with new certificates
- **Network isolation**: Use VPNs or private networks between sites when possible
- **Audit logging**: Enable and monitor FLARE audit logs


Troubleshooting
===============

**Client cannot connect to server**

- Verify server is running and listening on the correct port
- Check firewall rules allow traffic on ports 8002 and 8003
- Ensure hostname resolves correctly (DNS or ``/etc/hosts``)

**Authentication failures**

- Verify startup kits are from the same provisioning run
- Check that certificates haven't expired
- Ensure the correct startup kit is used on each machine

**Connection timeouts**

- Check network connectivity between client and server
- Verify no proxy or firewall is blocking gRPC traffic


References
==========

- :ref:`provisioning` - Detailed provisioning documentation
- :ref:`nvflare_dashboard_ui` - Dashboard UI guide
- :ref:`preflight_check` - Preflight check tool
- :ref:`operating_nvflare` - Admin console commands
- :ref:`flare_api` - FLARE API for programmatic control
- :ref:`flare_security_overview` - Security architecture
