.. _deploy_nvflare_in_aws:

############################
How to Deploy NVFLARE in AWS
############################

This guide covers deploying NVIDIA FLARE on Amazon Web Services (AWS) using EC2 instances.

Prerequisites
=============

Before deploying to AWS, ensure you have:

- An AWS account with appropriate permissions (AmazonEC2FullAccess for EC2 deployments)
- AWS CLI installed and configured with your credentials
- Required utilities installed:

  .. code-block:: shell

      sudo apt install sshpass bind9-dnsutils jq

- NVIDIA FLARE installed locally (``pip install nvflare``)


Provisioning
============

Before deploying to AWS, you need startup kits for the server, clients, and admin console.
There are two ways to generate these kits:

Using FLARE CLI
---------------

Create a ``project.yml`` configuration file defining your FL project, then run:

.. code-block:: shell

    nvflare provision

This generates startup kits in ``workspace/<project_name>/prod_00/`` containing folders for
the server, each client site, and admin users.

For AWS deployment, ensure the server CN matches the expected AWS hostname:

.. code-block:: yaml

    participants:
      - name: server
        type: server
        org: nvidia
        cn: ec2-xx-xx-xx-xx.compute-1.amazonaws.com

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

FLARE offers two deployment approaches for AWS:

1. **EC2 Deployment**: Direct deployment to EC2 instances using the cloud CLI
2. **Hybrid Deployment**: Mix of AWS cloud and on-premises deployments


EC2 Deployment
==============

The simplest way to deploy FLARE on AWS is using the built-in cloud deployment CLI, which automatically
provisions EC2 instances, security groups, and networking.

Deploy FL Server on AWS
-----------------------

With your FL server startup kit, launch the server on AWS:

.. code-block:: shell

    ./startup/start.sh --cloud aws

You can accept all default values by pressing ENTER, or provide a configuration file:

.. code-block:: shell

    ./startup/start.sh --cloud aws --config my_config.txt

The configuration file format:

.. code-block:: shell

    AMI_IMAGE=ami-03c983f9003cb9cd1
    EC2_TYPE=t2.small
    REGION=us-west-2

.. note::

    Recommended Ubuntu AMIs by version:

    - Ubuntu 20.04: ``ami-04bad3c587fe60d89``
    - Ubuntu 22.04: ``ami-03c983f9003cb9cd1``
    - Ubuntu 24.04: ``ami-0406d1fdd021121cd``

Upon successful deployment, you will see:

.. code-block:: text

    System was provisioned
    To terminate the EC2 instance, run the following command.
    aws ec2 terminate-instances --instance-ids i-0bf2666d27d3dd31d
    Other resources provisioned
    security group: nvflare_server_sg
    key pair: NVFlareServerKeyPair


Deploy FL Client on AWS
-----------------------

With your FL client startup kit, launch the client on AWS:

.. code-block:: shell

    ./startup/start.sh --cloud aws

Or with a configuration file:

.. code-block:: shell

    ./startup/start.sh --cloud aws --config my_config.txt

The configuration file uses the same format as the server deployment.


Deploy FLARE Dashboard on AWS
-----------------------------

To deploy the FLARE Dashboard for managing projects and distributing startup kits:

.. code-block:: shell

    nvflare dashboard --cloud aws

You will be prompted to enter an email address for the project admin login. The dashboard URL and
credentials will be displayed upon successful deployment.


Terminating EC2 Resources
-------------------------

To terminate EC2 instances and clean up resources:

.. code-block:: shell

    # Terminate server instance
    aws ec2 terminate-instances --instance-ids <instance-id>

    # Delete security groups (after terminating instances)
    aws ec2 delete-security-group --group-name nvflare_server_sg

    # Delete key pairs
    aws ec2 delete-key-pair --key-name NVFlareServerKeyPair


Hybrid Deployment
=================

FLARE supports hybrid deployments where components run across different environments:

- FL Server on AWS with clients on-premises
- FL Server on Azure with clients on AWS
- Mix of cloud and on-premises clients

To configure hybrid deployment, use the AWS instance's public DNS name as the server CN in your
``project.yml`` file:

.. code-block:: yaml

    server:
      cn: ec2-3-99-123-456.compute-1.amazonaws.com

Then provision and distribute startup kits accordingly.


Post-Deployment Verification
============================

After deployment, verify the system is running correctly using preflight checks and system status commands.

Preflight Check
---------------

After the FL system starts but before running jobs, use the NVIDIA FLARE preflight check to verify
the deployment configuration and connectivity.

**Check Server**: On the server EC2 instance, run:

.. code-block:: shell

    nvflare preflight_check -p /path/to/server_startup_kit

This verifies gRPC port binding, admin port binding, storage writability, and dry run.

**Check Clients**: After the server is running, on each client EC2 instance, run:

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
programmatically to check system status.


Security Considerations
=======================

- Ensure security groups allow inbound traffic only on required ports (8002, 8003 by default)
- Use private subnets for clients when possible
- Enable VPC flow logs for network monitoring
- Consider using AWS PrivateLink for secure communication
- Regularly rotate credentials and certificates


Troubleshooting
===============

**Cannot connect to server from external machine**

Update the security group inbound rules to include your IP address.

**Startup kit authentication fails**

Ensure the server hostname in the startup kit matches the EC2 instance's public DNS name.

**Insufficient permissions**

Verify your AWS IAM role has AmazonEC2FullAccess or equivalent permissions.


References
==========

- :ref:`cloud_deployment` - Complete cloud deployment guide
- :ref:`preflight_check` - Preflight check tool documentation
- :ref:`provisioning` - Provisioning and startup kit generation
- :ref:`flare_security_overview` - Security architecture overview
