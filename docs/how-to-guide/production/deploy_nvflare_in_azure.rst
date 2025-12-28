.. _deploy_nvflare_in_azure:

##############################
How to Deploy NVFLARE in Azure
##############################

This guide covers deploying NVIDIA FLARE on Microsoft Azure using Virtual Machines.

Prerequisites
=============

Before deploying to Azure, ensure you have:

- An Azure account with appropriate permissions (ability to create Resource Groups, Virtual Machines, and configure Network Security Groups)
- Azure CLI installed and configured (see `Azure CLI installation guide <https://learn.microsoft.com/en-us/cli/azure/install-azure-cli>`_)
- Required utilities installed:

  .. code-block:: shell

      sudo apt install sshpass bind9-dnsutils jq

- NVIDIA FLARE installed locally (``pip install nvflare``)


Provisioning
============

Before deploying to Azure, you need startup kits for the server, clients, and admin console.
There are two ways to generate these kits:

Using FLARE CLI
---------------

Create a ``project.yml`` configuration file defining your FL project, then run:

.. code-block:: shell

    nvflare provision

This generates startup kits in ``workspace/<project_name>/prod_00/`` containing folders for
the server, each client site, and admin users.

For Azure deployment, ensure the server CN matches the expected Azure hostname:

.. code-block:: yaml

    participants:
      - name: server
        type: server
        org: nvidia
        cn: nvflareserver.westus2.cloudapp.azure.com

      - name: site-1
        type: client
        org: site1

      - name: site-2
        type: client
        org: site2

      - name: admin@nvidia.com
        type: admin
        org: nvidia
        role: project_admin

For detailed provisioning options, see :ref:`provisioning`.

Using FLARE Dashboard
---------------------

Alternatively, deploy the FLARE Dashboard first (see :ref:`dashboard_api`), which provides
a web UI for:

- Configuring the FL project
- Inviting participants
- Generating and distributing startup kits on-demand

This approach is useful when you don't know all participant details upfront.


Deployment Options
==================

FLARE offers two deployment approaches for Azure:

1. **VM Deployment**: Direct deployment to Azure Virtual Machines using the cloud CLI
2. **Hybrid Deployment**: Mix of Azure cloud and on-premises deployments


Azure VM Deployment
===================

The simplest way to deploy FLARE on Azure is using the built-in cloud deployment CLI, which automatically
provisions Virtual Machines, resource groups, and networking.

Deploy FL Server on Azure
-------------------------

With your FL server startup kit, launch the server on Azure:

.. code-block:: shell

    ./startup/start.sh --cloud azure

You can accept all default values by pressing ENTER. The script will prompt for Azure login
via browser authentication.

Or provide a configuration file:

.. code-block:: shell

    ./startup/start.sh --cloud azure --config my_config.txt

The configuration file format:

.. code-block:: shell

    VM_IMAGE=Canonical:0001-com-ubuntu-server-focal:20_04-lts-gen2:latest
    VM_SIZE=Standard_B2ms
    LOCATION=westus2

Upon successful deployment, you will see:

.. code-block:: text

    Creating Resource Group nvflare_rg at Location westus2
    Creating Virtual Machine, will take a few minutes
    Setting up network related configuration
    Copying files to nvflare_server
    Installing packages in nvflare_server, may take a few minutes.
    System was provisioned


Deploy FL Client on Azure
-------------------------

With your FL client startup kit, launch the client on Azure:

.. code-block:: shell

    ./startup/start.sh --cloud azure

Or with a configuration file:

.. code-block:: shell

    ./startup/start.sh --cloud azure --config my_config.txt

The configuration file uses the same format as the server deployment.


Deploy FLARE Dashboard on Azure
-------------------------------

To deploy the FLARE Dashboard for managing projects and distributing startup kits:

.. code-block:: shell

    nvflare dashboard --cloud azure

You will be prompted to:

1. Enter an email address for the project admin login
2. Complete Azure login via browser authentication

Upon successful deployment:

.. code-block:: text

    Starting dashboard
    Dashboard: Project admin credential is hello@world.com and the password is E3pZkD50, running at IP address 20.20.123.123
    To stop it, run az group delete -n nvflare_dashboard_rg

.. note::

    For HTTPS mode, place ``web.crt`` and ``web.key`` files in the current working directory
    before running the dashboard command. HTTPS is highly recommended for production use.


Terminating Azure Resources
---------------------------

To terminate Azure resources and clean up:

.. code-block:: shell

    # Delete server resources
    az group delete -n nvflare_rg

    # Delete client resources
    az group delete -n nvflare_client_rg

    # Delete dashboard resources
    az group delete -n nvflare_dashboard_rg


Hybrid Deployment
=================

FLARE supports hybrid deployments where components run across different environments:

- FL Server on Azure with clients on-premises
- FL Server on AWS with clients on Azure
- Mix of cloud and on-premises clients

To configure hybrid deployment, use the Azure VM's public DNS name as the server CN in your
``project.yml`` file:

.. code-block:: yaml

    server:
      cn: nvflareserver.westus2.cloudapp.azure.com

Then provision and distribute startup kits accordingly.


Post-Deployment Verification
============================

After deployment, verify the system is running correctly using preflight checks and system status commands.

Preflight Check
---------------

After the FL system starts but before running jobs, use the NVIDIA FLARE preflight check to verify
the deployment configuration and connectivity.

**Check Server**: On the server VM, run:

.. code-block:: shell

    nvflare preflight_check -p /path/to/server_startup_kit

This verifies gRPC port binding, admin port binding, storage writability, and dry run.

**Check Clients**: After the server is running, on each client VM, run:

.. code-block:: shell

    nvflare preflight_check -p /path/to/client_startup_kit

This verifies the client can connect to the server and dry run succeeds.

**Check Admin Console**: Verify the admin console can connect:

.. code-block:: shell

    nvflare preflight_check -p /path/to/admin_startup_kit

For detailed information, see :ref:`preflight_check`.

Using FLARE Console
-------------------

Launch the admin console from your admin startup kit:

.. code-block:: shell

    ./startup/fl_admin.sh

Check server status:

.. code-block:: text

    > check_status server

Using FLARE API
---------------

Use the ``system_info.ipynb`` notebook included in the admin startup kit, or use the FLARE API
programmatically to check system status. The notebook can be run directly in Azure ML Notebook
by uploading the startup kit to Azure's web UI.


Security Considerations
=======================

- Configure Network Security Groups to allow inbound traffic only on required ports (8002, 8003 by default)
- Use private subnets and Azure Private Link for enhanced security
- Enable Azure Network Watcher for traffic monitoring
- Use Azure Key Vault for certificate management
- Regularly rotate credentials and certificates
- Consider using Azure Trusted Launch VMs for enhanced security


Troubleshooting
===============

**Azure login fails**

If browser-based login is not available, use device code flow:

.. code-block:: shell

    az login --use-device-code

**Cannot connect to server from external machine**

Update the Network Security Group inbound rules to include your IP address.

**Startup kit authentication fails**

Ensure the server hostname in the startup kit matches the Azure VM's public DNS name.

**Resource group already exists**

The deployment script will fail if resources from a previous deployment exist. Clean up
existing resources before redeploying:

.. code-block:: shell

    az group delete -n nvflare_rg


References
==========

- :ref:`cloud_deployment` - Complete cloud deployment guide
- :ref:`preflight_check` - Preflight check tool documentation
- :ref:`provisioning` - Provisioning and startup kit generation
- :ref:`flare_security_overview` - Security architecture overview
