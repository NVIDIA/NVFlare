.. _cloud_deployment:

################
Cloud Deployment
################
To deploy NVIDIA FLARE dashboards, servers or clients to Azure or AWS, you must have the account with proper permissions.  In Azure, your role in Azure
subscription must be able to create Resource Group, Virtual Machine and configure Network Security Group and its rules.
In AWS, your role should have AmazonEC2FullAccess.

Dashboard
=========

To launch a dashboard in AWS, run 

.. code-block:: shell

    nvflare dashboard --cloud aws


To launch a dashboard in Azure, run 

.. code-block:: shell

    nvflare dashboard --cloud azure

Server/Client
=============

After downloading the startup kit from NVIDIA FLARE dashboard, unzip it and run ``start.sh`` with ``--cloud aws`` (or ``--cloud azure``) to launch the server
or client in AWS (or Azure).

You can also provide a configuration file with ``--config $FILE_NAME`` to the ``start.sh`` script. The configuration
file will take the place of the user responding to prompts when starting the server or client.
The configuration file is in the format of Bash setting variables.

.. attention:: The variable names are different in AWS and Azure.

In AWS:

.. code-block::

    AMI_IMAGE=ami-04bad3c587fe60d89
    EC2_TYPE=t2.small
    REGION=us-west-2


In Azure:

.. code-block::

    VM_IMAGE=Canonical:0001-com-ubuntu-server-focal:20_04-lts-gen2:latest
    VM_SIZE=Standard_B2ms
    LOCATION=westus2

For example, launch a NVIDIA FLARE server in AWS with a configuration file ``my_config.txt``, run 

.. code-block::

    $ ./startup/start.sh --cloud aws --config my_config.txt
    This script requires aws (AWS CLI), sshpass and jq.  Now checking if they are installed.
    Checking if aws exists. => found
    Checking if sshpass exists. => found
    Checking if jq exists. => found
    If the server requires additional dependencies, please copy the requirements.txt to /home/nvflare/workspace/aws/nvflareserver/startup.
    Press ENTER when it's done or no additional dependencies. 
    Generating key pair for VM
    Creating VM at region us-west-2, may take a few minutes.
    VM created with IP address: 34.223.3.172
    Copying files to nvflare_server
    Destination folder is ubuntu@34.223.3.172:/var/tmp/cloud
    Installing packages in nvflare_server, may take a few minutes.
    System was provisioned
    To terminate the EC2 instance, run the following command.
    aws ec2 terminate-instances --instance-ids i-0bf2666d27d3dd31d
    Other resources provisioned
    security group: nvflare_server_sg
    key pair: NVFlareServerKeyPair

Post Deployment
===============

After deploying dashboard/server/client to the cloud, you can ssh into the VM.  If you try to run ssh from a computer other than the one you ran the scripts,
its public IP address might not be within the source IP range of inbound rules.  Please use AWS or Azure web to update the inbound rules.