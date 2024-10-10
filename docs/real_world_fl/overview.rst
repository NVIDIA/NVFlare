########
Overview
########

************
Introduction
************

NVIDIA FLARE utilizes provisioning and admin clients to reduce the amount of human coordination involved to set up a
federated learning project. A provisioning tool can be configured to create a startup kit for each site.
These packages can then be delivered to each site ready to go, streamlining the process to provision, start,
and operate federated learning with a trusted setup.

Provision - Start - Operate
===========================

Provision
---------
There are different ways to generate startup packages (or startup kits) for the FL server, FL clients, and admin users. Depending on how the
packages are created and distributed then deployed, there are multiple ways to provision the packages. See :ref:`provisioning` for details.

Start
-----
Each site should have an org admin to receive or download the startup packages. The org admin can then install their own packages, start
the services, map the data location, and instrument the authorization policies and organization level site privacy policies.

Again, see :ref:`provisioned_setup` for client deployment, and :ref:`site_policy_management` for how to set up federated authorization and privacy policies.

Operate
-------
Lead scientists / administrators control the federated learning process: submit jobs to deploy applications, check statuses,
abort / shutdown training.

See the sections on :ref:`operating_nvflare` and see how to use different ways to perform the above operations. 

.. _provisioned_setup:

******************************************************************************
Provisioning and startup package distribution
******************************************************************************
A startup kit package is a set of scripts and credentials that allows for FL participants to communicate securely. The process of
generating such packages is called **Provisioning** in NVIDIA FLARE.

NVIDIA FLARE provides two ways to provision, using the NVIDIA FLARE ``provision`` CLI tool or using the NVFLARE Dashboard.

Provisioning via the CLI tool
=============================
The :ref:`provisioning` page has details on the contents of the provisioning tool and the underlying NVIDIA FLARE Open Provision API,
which you can use to customize configurations to fit your own requirements.

Edit the :ref:`Project yaml file <project_yml>` in the directory with the provisioning tool to meet your project requirements (make sure the
server, client sites, admin, orgs, and everything else are right for your project).

Then run the provision command with (here we assume your
project.yml is in current working directory):

.. code-block:: bash

    nvflare provision -p project.yml

The generated startup kits are created by default in a directory prefixed with "prod\_" within a folder of the project
name in the workspace folder created where ``provision.py`` is run.

Customize the provision configuration
-------------------------------------
For advanced users, you can customize your provision with additional behavior through additional builders:

    - **Zip**: To create password protected zip archives for the startup kits, see :ref:`distribution_builder`
    - **Docker-compose**: Provision to launch NVIDIA FLARE system via docker containers. You can customize the provisioning process and ask the provisioner to generate a docker-compose file. This can be found in :ref:`docker_compose`.
    - **Docker**: Provision to launch NVIDIA FLARE system via docker containers. If you just want to use docker files, see :ref:`containerized_deployment`.
    - **Helm**: To change the provisioning tool to generate an NVIDIA FLARE Helm chart for Kubernetes deployment, see :ref:`helm_chart`.
    - **CUSTOM**: you can build custom builders specific to your needs like in :ref:`distribution_builder`.

Package distribution
--------------------
Once provisioned, you will have startup packages in different server and client folders. With the CLI approach, you not only
need to collect all participants and organization/client host information, but you also need to distribute the packages to the participating
organizations. You can use email, sftp etc. to do so as long as you can ensure that it is secure.

.. attention::

   In order to change configurations, it may be necessary to alter nvflare/lighter/impl/master_template.yml before
   running provision with your checked out version of the code (make sure PYTHONPATH points to the location of where you
   checked out the NVFlare repository).

   You cannot directly edit the contents of the startup kits because the contents of the generated startup kits are
   signed by :class:`SignatureBuilder<nvflare.lighter.impl.signature.SignatureBuilder>` so the system will detect if any
   of the files have been altered and may not run.

Provision via Dashboard UI
==========================
The :ref:`nvflare_dashboard_ui` is a new optional addition to NVIDIA FLARE introduced in version 2.2.1 that allows for the project
administrator to deploy a website to gather information about the sites and distribute startup kits.

Introduction to NVFLARE Dashboard
---------------------------------
You can install and run :ref:`nvflare_dashboard_ui` using the dashboard CLI command, ``nvflare dashboard –start`` (stop with ``nvflare dashboard –stop``).

For details on how to start Dashboard can be found :ref:`here <dashboard_api>`. The usage information for the Dashboard UI can be found :ref:`here <nvflare_dashboard_ui>`.

Once the dashboard is set up and the project is published, the start up kits for all the participants can be downloaded from the Dashboard.

Unlike the CLI provision, there is not as much customization option, as the information is captured by the Dashboard UI and default builders are used.

Compared to the CLI provision option, there is a much simplified effort in distribution of the startup kit, as each user downloads his own startup kit. 

.. note::

   It is important that the "startup" folder in each startup kit is not renamed because the code relies upon this for operation. Please
   note that a "transfer" directory and deployed applications will be created at the level of this "startup" folder. See the
   section on `Internal folder and file structures for NVIDIA FLARE`_ below for more details.

************************************************************************************
Start: NVIDIA FLARE Package Deployment
************************************************************************************
There are multiple ways to deploy NVFLARE depending on use case:

    - On-premise
    - In the cloud
    - On bare-metal
    - On docker
    - On K8s

In this section, we will discuss how to deploy for different cases.

On-Premise Deployment 
=============================

Local host deployment
---------------------

Production mode, non-HA, secure, local
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can use the CLI provision command to generate a local production mode.

Docker mode
^^^^^^^^^^^
Use builder to generate docker-compose files to allow different dockers run locally.  Similar to production mode, the server needs either to be localhost or one needs to modify /etc/hosts
Run docker-compose up/down

Bare-metal deployment
---------------------
To deploy on-premise, copy the startup package to each host machine then start with the start script.

Docker-based deployment
-----------------------
Docker: Build the docker image for each startup kit,

Then run docker run : docker run

See the details at :ref:`docker_compose`.

Cloud Deployment
================
If you decide to leverage the public cloud (Azure or AWS) to deploy NVIDIA FLARE, the newly added cloud deployment features allow hybrid
cloud deployment such as deployment of the FL Server at Azure and FL Clients in AWS, with another FL Client on premises for example.

See how to deploy to Azure and AWS clouds can be found in :ref:`cloud_deployment`.

Deploy to Google Cloud will be made available in a future release.

Kubernetes Deployment
=====================
As mentioned above, you can run NVIDIA FLARE in the public cloud.  If you prefer to deploy NVIDIA FLARE in Amazon Elastic Kubernetes Service (EKS),
you can find the deployment guide in :ref:`eks_deployment`.


Starting Federated Learning Servers
=============================================
The FL Server will coordinate the federated learning training and be the main hub all clients and admin
clients connect to.

In the package for each server, run the start.sh file from the "startup" folder to start the server.

The rootCA.pem file is pointed to by "ssl_root_cert" in fed_server.json.  If you plan to move/copy it to a different place,
you will need to modify fed_server.json.  The same applies to the other two files, server.crt and server.key.

.. note::

   When launching the FL server inside a docker with ``docker run``, use ``--net=host`` to map hostname into that
   docker instance.  For secure gRPC communication, the FL server has to bind to the hostname specified in the
   provisioning stage. Always make sure that hostname is what FL server can bind to. Additionally,
   the port that the server communicates on must also not be blocked by any firewalls.

If clients from other machines cannot connect to the server, make sure that the hostname (name of the server under
participants in project.yml) specified when generating the startup kits in the provisioning process resolves to the
correct IP. If the FL server is on an internal network without a DNS hostname, in Ubuntu, an entry may need to be added
to ``/etc/hosts`` with the internal IP and the hostname.

Starting Federated Learning Clients
============================================
Each site participating in federated learning training is a client. Each package for a client is named after the client
name specified when provisioning the project.

In the package for each client, run ``start.sh``
from the "startup" folder to start the client.

.. tip::

   You need to first install NVIDIA FLARE package before running the ``start.sh`` shell script.  NVIDIA FLARE is available
   on PyPi and can be installed with ``python3 -m pip install nvflare``.

   Depending on the deployed application which shall start later, your environment may need some additional
   Python packages.  If you haven't installed them, do it after you install NVIDIA FLARE.  NVIDIA FLARE does not dictate
   your deep learning environments.  It's completely up to you to set it up.

.. note::

    Coordination for where to mount the data may be needed depending on where the dataset is located in the application to be deployed.

The rootCA.pem file is pointed to by "ssl_root_cert" in fed_client.json.  If you plan to move/copy it to a different place,
you will need to modify fed_client.json.  The same applies to the other two files, client.crt and client.key.

The client name in your submission to participate this federated learning project is embedded in the CN field of client
certificate, which uniquely identifies the participant. As such, please safeguard its private key, client.key.

When a client successfully connects to the FL server, the server and that client will both log a token confirming that
the client successfully connected:

Server::

    2020-07-07 03:48:49,712 - ClientManager - INFO - Client: New client abcd@127.0.0.1 joined. Sent token: f279157b-df8c-aa1b-8560-2c43efa257bc.  Total clients: 1

Client::

    2020-07-07 03:48:49,713 - FederatedClient - INFO - Successfully registered client:abcd for exampletraining. Got token:f279157b-df8c-aa1b-8560-2c43efa257bc

If a connection cannot be made, the client will repeatedly try to connect and for each failure log::

    Could not connect to server. Setting flag for stopping training. failed to connect to all addresses

If the server is up, you may need to troubleshoot with settings for firewall ports to make sure that the proper
permissions are in place. This could require coordination between the lead IT and site IT personnel.

Federated Learning Administration Console
=========================================
Each admin console will be able to connect and submit commands to the server. Each admin console package is named after
the email specified when provisioning the project, and the same email will need to be entered for authentication when
the admin console is launched.

Install the wheel package first with::

    python3 -m pip install nvflare[apt_opt]


After installation, you can run the ``fl_admin.sh`` file to start communicating with the FL server.
The FL server must be running and there must be a successful connection between the admin
console and the FL server in order for the admin console to start. For the prompt **User Name:**, enter the email that was
used for that admin console in the provisioning of the project.

The ``rootCA.pem`` file is pointed to by "ca_cert" in fl_admin.sh.  If you plan to move/copy it to a different place,
you will need to modify the corresponding script.  The same applies to the other two files, client.crt and client.key.

The email to participate this FL project is embedded in the CN field of client certificate, which uniquely identifies
the participant. As such, please safeguard its private key, client.key.

.. attention::

   You will need write access in the directory containing the "startup" folder because the "transfer" directory for
   uploading files as well as directories created for federated learning runs will live here. For details, see
   `Internal folder and file structures for NVIDIA FLARE`_.

*******************************************************
Operate: Running federated learning as an administrator
*******************************************************

Running federated learning from the administration console
==========================================================
With all connections between the FL server, FL clients, and administration consoles open and all of the parties
started successfully as described in the preceding section, `Federated Learning Administration Console`_,
admin commands can be used to operate a federated learning project. The FLAdminAPI provides a way to programmatically
issue commands to operate the system so it can be run with a script.

For a complete list of admin commands, see :ref:`operating_nvflare`.

For examples of using the commands to operate a FL system, see the examples in the :ref:`getting_started` section.

Operate from Notebook or FLARE API
==================================
Many of the tasks previously only available through admin console can now be done through the FLARE API from a notebook.
See :ref:`flare_api`.

****************************************************
Internal folder and file structures for NVIDIA FLARE
****************************************************

Please refer to :ref:`server workspace <server_workspace>` and :ref:`client workspace <client_workspace>`
for the folder and file structures on the server/client side.

Administrator side folder and file structure
============================================
::

    /some_path_on_fl_admin/fl_administrator_workspace_root/
        startup/
            client.crt
            client.key
            fl_admin.sh
            readme.txt
            rootCA.pem
            signature.pkl
        transfer/
            application_for_uploading/
                config/
                models/
                resources/
            application2_for_uploading/
                config/
                models/
                resources/
