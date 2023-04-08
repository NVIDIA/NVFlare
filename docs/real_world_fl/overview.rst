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
Project administrator generates the packages for the server / clients / admins

Start
-----
Org admins install their own packages, starts the services, and maps the data location

Operate
-------
Lead scientists / administrators control the federated learning process: submit jobs to deploy applications, check statuses, abort / shutdown training

.. _provisioned_setup:

******************************************************************************
Provision: Configure and generate packages for the server, clients, and admins
******************************************************************************
One party leads the process of configuring the provisioning tool and using it to generate startup kits for each party in
the federated learning training project:

Preparation for using the provisioning tool
===========================================
After :ref:`installation`, the provisioning tool is available via ``provision`` command.

Provisioning a federated learning project
=========================================
The :ref:`provisioning` page has details on the contents of the provisioning tool and the underlying NVIDIA FLARE Open
Provision API, which you can use to customize configurations to fit your own requirements.

.. note::

    Starting in NVIDIA FLARE version 2.2.1, the :ref:`nvflare_dashboard_ui` has been introduced for an easier experience for
    provisioning a project and distributing the startup kits. If you are using the Dashboard UI, see :ref:`dashboard_api` for
    details on how to set it up, and you can skip the rest of this :ref:`provisioned_setup` section.

Edit the :ref:`programming_guide/provisioning_system:Project yaml file` in the directory with the provisioning tool to meet your
project requirements (make sure the server, client sites, admin, orgs, and everything else are right
for your project).

Then run the provision command with (here we assume your project.yml is in current working directory)::

    nvflare provision -p project.yml

The generated startup kits are created by default in a directory prefixed with "prod\_" within a folder of the project
name in the workspace folder created where provision.py is run. To create password protected zip archives for the startup
kits, see :ref:`distribution_builder`.

.. attention::

   In order to change configurations, it may be necessary to alter nvflare/lighter/impl/master_template.yml before
   running provision with your checked out version of the code (make sure PYTHONPATH points to the location of where you
   checked out the NVFlare repository).

   You cannot directly edit the contents of the startup kits because the contents of the generated startup kits are
   signed by :class:`SignatureBuilder<nvflare.lighter.impl.signature.SignatureBuilder>` so the system will detect if any
   of the files have been altered and may not run.

.. note::

   It is important that the "startup" folder in each startup kit is not renamed because the code relies upon this for operation. Please
   note that a "transfer" directory and deployed applications will be created at the level of this "startup" folder. See the
   section on `Internal folder and file structures for NVIDIA FLARE`_ below for more details.

************************************************************************************
Start: Instructions for each participant to start running FL with their startup kits
************************************************************************************

.. attention:: Please always safeguard .key files! These are the critical keys for secure communication!

Overseer (HA mode only)
=============================
In HA mode, one single Overseer will keep track of all the FL servers and communicate to all the participants through their Overseer
Agents the active FL server or SP.

In the package for the Overseer, run the start.sh file from the "startup" folder to start the Overseer.

If clients from other machines cannot connect to the Overseer, make sure that the hostname (name of the server under
participants in project.yml) specified when generating the startup kits in the provisioning process resolves to the
correct IP. If the FL server is on an internal network without a DNS hostname, in Ubuntu, an entry may need to be added
to ``/etc/hosts`` with the internal IP and the hostname.

Federated learning servers
=============================================
Server will coordinate the federated learning training and be the main hub all clients and admin
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

Federated learning clients
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

Federated learning administration client
========================================
Each admin client will be able to connect and submit commands to the server. Each admin client package is named after
the email specified when provisioning the project, and the same email will need to be entered for authentication when
the admin client is launched.

Install the wheel package first with::

    python3 -m pip install nvflare


After installation, you can run the **fl_admin.sh** file to start communicating to the FL server.
The FL server must be running and there must be a successful connection between the admin
client and the FL server in order for the admin client to start. For the prompt **User Name:**, enter the email that was
used for that admin client in the provisioning of the project.

The rootCA.pem file is pointed to by "ca_cert" in fl_admin.sh.  If you plan to move/copy it to a different place,
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

Running federated learning from the administration client
=========================================================
With all connections between the FL server, FL clients, and administration clients open and all of the parties
started successfully as described in the preceding section, `Federated learning administration client`_,
admin commands can be used to operate a federated learning project. The FLAdminAPI provides a way to programmatically
issue commands to operate the system so it can be run with a script.

For a complete list of admin commands, see :ref:`operating_nvflare`.

For examples of using the commands to operate a FL system, see the examples in the :ref:`getting_started` section.

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
