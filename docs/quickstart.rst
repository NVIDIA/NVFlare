###########
Quick Start
###########

This quick start guide means to help the user get FLARE up & running
quickly without introducing any advanced concepts. For more details, refer
to :ref:`getting_started`.

Since FLARE offers different modes of running the system, we only cover the simplest approaches here.
This quick start guide uses the **examples/hello-world/hello-numpy-sag** as an example.
You will find the details in the example's README.md file.

We also assume you have worked with Python, already set up the virtual env.
If you are new to this, please refer :ref:`getting_started`.

**Install NVFLARE**
======================

.. code-block:: shell

  $ python3 -m pip install nvflare

Clone NVFLARE repo to get examples, switch main branch (latest stable branch)

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git
  $ cd NVFlare
  $ git switch main


**Quick Start with Simulator**
==============================


.. code-block:: shell

   nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 examples/hello-world/hello-numpy-sag

Now you can watch the simulator run two clients (n=2) with two threads (t=2)
and logs are saved in the `/tmp/nvflare/hello-numpy-sag` workspace.


**Quick start with POC mode**
=============================


Instead of using the simulator, you can simulate the real deployment with
multiple processes via POC mode:


.. code-block:: shell

   $ nvflare poc --prepare -n 2
   $ nvflare poc --start -ex admin


From another terminal, start FLARE console:

.. code-block::

   $ nvflare poc --start -p admin


Once FLARE Console started, you can check the status of the server.


.. code-block:: console

   $ check_status server
   $ submit_job hello-world/hello-numpy-sag
   $ list_jobs

You can find out the other commands by using "?",  you can download job results. use "bye" to exit.

.. code-block:: console

   $ bye

You can use poc command to shutdown clients and server

.. code-block:: shell

   $ nvflare poc --stop


**Quick start with Production Mode**
====================================

   Before you work in production mode, you need to first **provision**, a process to generate **startup kit**.
   Startup kits are set of start scripts, configuration and certificates associated with different sites and server.
   In this quick guide, we only show None-HA (non high availability mode), we will only have one FL server.

   There are two way of provisions: CLI and Flare Dashboard (UI Application), we are going to use CLI here.


**provision with CLI**
---------------


.. code-block:: shell

   $ cd /tmp
   $ nvflare provision

select 2 for non-HA mode.  If you will generate a project.yml in the current directory. This will be the base configuration
files for provision. By default, the project.yml will have one server and two clients pre-defined

  * server1
  * site-1
  * site-2

Now we are ready to provision,

.. code-block:: shell

  $ cd /tmp
  $ nvflare provision -p project.yml


it will generate startup kits in the following directory

.. code-block:: shell

  /tmp/workspace/example_project/prod_00


**Start Flare Server, Clients, Flare Console**
------------------------------------------------


First start FL Server, open a new **terminal** for server

.. code-block:: shell

  $ cd /tmp/workspace/example_project/prod_00
  $ ./server1/startup/start.sh


Next start Site-1 and Site-2, open a new **terminal** for each site
in site-1 terminal:

.. code-block:: shell

  $ cd /tmp/workspace/example_project/prod_00
  $ ./site-1/startup/start.sh

in site-2 terminal:

.. code-block:: shell

  $ cd /tmp/workspace/example_project/prod_00
  $ ./site-2/startup/start.sh


Next finally for Flare console, open a new **terminal**

.. code-block:: shell

  $ cd /tmp/workspace/example_project/prod_00
  $ ./admin@nvidia.com/startup/fl_admin.sh

Once console started, you can use check-status command just like POC mode


**Provision and distributing startup kits via Flare Dashboard UI**
--------------------------------------------------------------------

Start the dashboard, then following the instructions. Once Dashboard started, you can setup project, invite users
to participate, once user add the sites, you can approve the user and sites, then freeze the project. The user can download
the startup kits from the UI.

.. code-block:: shell

 nvflare dashboard --start

