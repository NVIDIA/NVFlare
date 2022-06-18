.. _quickstart:

##########
Quickstart
##########

This section provides a starting point for new users to start NVIDIA FLARE.
Users can go through the :ref:`example_applications` and get familiar with how NVIDIA FLARE is designed,
operates and works.

Each example introduces concepts about NVIDIA FLARE, while showcasing how some popular libraries and frameworks can
easily be integrated into the FL process.

.. _installation:

Installation
=============

Python Version
--------------

NVIDIA FLARE requires Python 3.8.  It may work with Python 3.7 but currently is not compatible with Python 3.9 and above.

Install NVIDIA FLARE in virtual environments
--------------------------------------------

It is highly recommended to install NVIDIA FLARE in a virtual environment.
This guide briefly describes how to create a virtual environment with venv.

Virtual Environments and Packages
.................................

Python's official document explains the main idea about virtual environments.
The module used to create and manage virtual environments is called `venv <https://docs.python.org/3.8/library/venv.html#module-venv>`_.
You can find more information there.  We only describe a few necessary steps for a virtual environment for NVIDIA FLARE.


Depending on your OS and the Python distribution, you may need to install the Python's venv package separately.  For example, in ubuntu
20.04, you need to run the following commands to continue creating a virtual environment with venv.

.. code-block:: shell

   $ sudo apt update
   $ sudo apt-get install python3-venv


Once venv is installed, you can use it to create a virtual environment with:

.. code-block:: shell

    $ python3 -m venv nvflare-env

This will create the ``nvflare-env`` directory in current working directory if it doesn’t exist,
and also create directories inside it containing a copy of the Python interpreter,
the standard library, and various supporting files.


Activate the virtualenv by running the following command:

.. code-block:: shell

    $ source nvflare-env/bin/activate


You may find that the pip and setuptools versions in the venv need updating:

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install -U pip
  (nvflare-env) $ python3 -m pip install -U setuptools


Install Stable Release
----------------------

Stable releases are available on `NVIDIA FLARE PyPI <https://pypi.org/project/nvflare>`_:

.. code-block:: shell

  $ python3 -m pip install nvflare


.. _containerized_deployment:

Containerized Deployment with Docker
====================================

Running NVIDIA FLARE in a Docker container is sometimes a convenient way to ensure a
uniform OS and software environment across client and server systems.  This can be used
as an alternative to the bare-metal Python virtual environment described above and will
use a similar installation to simplify transitioning between the environments.

A simple Dockerfile is used to capture the base requirements and dependencies.  In
this case, we're building an environment that will support PyTorch-based workflows,
in particular the `Hello PyTorch with Tensorboard Streaming <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-pt-tb>`_
example. The base for this build is the NGC PyTorch container.  On this base image,
we will install the necessary dependencies and clone the NVIDIA FLARE GitHub
source code into the root workspace directory.

.. code-block:: dockerfile

   ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:22.04-py3
   FROM ${PYTORCH_IMAGE}

   RUN python3 -m pip install -U pip
   RUN python3 -m pip install -U setuptools
   RUN python3 -m pip install torch torchvision tensorboard nvflare

   WORKDIR /workspace/
   RUN git clone https://github.com/NVIDIA/NVFlare.git

We can then build the new container by running docker build in the directory containing
this Dockerfile, for example tagging it nvflare-pt:

.. code-block:: shell

  docker build -t nvflare-pt .

You will then have a docker image nvflare-pt:latest.

This container can be used to run any of the client or server deployments.

In POC mode (described in the next section), you can do this by mounting the directory
containing the server or client subdirectories and startup scripts when you run the
docker container.

.. _setting_up_poc:

Setting Up the Application Environment in POC Mode
==================================================

.. warning::

    POC mode is not intended to be secure and should not be run in any type of production environment or any environment
    where the server's ports are exposed. For actual deployment and even development, it is recommended to use a
    :ref:`secure provisioned setup <provisioned_setup>`.

To get started with a proof of concept (POC) setup after :ref:`installation`, run this command to generate a poc folder
with an overseer, server, two clients, and one admin client:

.. code-block:: shell

    $ poc -n 2

The resulting poc folder will contain the following structure, with start.sh scripts for each of the participants.::

  poc/
    admin/
        startup/
            fed_admin_HA.json
            fed_admin.json
            fl_admin.sh
    overseer/
        startup/
            start.sh
    Readme.rst
    server/
        startup/
            fed_server_HA.json
            fed_server.json
            log.config
            start.sh
            stop_fl.sh
            sub_start.sh
    site-1/
        startup/
            fed_client_HA.json
            fed_client.json
            log.config
            start.sh
            stop_fl.sh
            sub_start.sh
    site-2/
        startup/
            fed_client_HA.json
            fed_client.json
            log.config
            start.sh
            stop_fl.sh
            sub_start.sh


Before we use these scripts to connect the overseer, server, and clients, we will clone the NVFlare Repository
that contains the set of example applications.

.. _cloning_and_examples:

Cloning the NVFlare Repository and Examples
===========================================

In this section, we will focus on the hello-pt-tb example as a simple POC.
For more details on all examples please refer to :ref:`example_applications`.

First, we need to clone the repo to get the source code including examples:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

We can then copy the necessary files (the exercise code in the examples directory of the NVFlare repository)
to a working folder (the transfer folder for the admin client):

.. code-block:: shell

  mkdir -p poc/admin/transfer
  cp -rf NVFlare/examples/* poc/admin/transfer

This step has copied all the NVFlare examples into the admin client's transfer folder.

Once the server and clients are connected, the admin client can be used to deploy and run any of these applications.

The hello-pt-tb application requires a few dependencies to be installed.

As in the installation section, we can install these in the Python virtual environment by running:

.. code-block:: shell

  source nvflare-env/bin/activate
  python3 -m pip install torch torchvision tensorboard

If using the Dockerfile above to run in a container, these dependencies have already been installed.

.. _starting_poc:

Starting the Application Environment in POC Mode
================================================

Once you are ready to start the FL system, you can run the following commands to start the server and client systems.  Following that, we will use the admin client to deploy and run an example app.

.. note::
  Each of the participants will run in a separate terminal or in a terminal multiplexer like screen or tmux.  Each of these sessions reqiures the NVFlare Python environment, either built into a container as described above, or by running

  .. code-block:: shell

    source nvflare-env/bin/activate
  
  as described in the :ref:`installation <installation>` section.

  If running containerized, you can use a terminal multiplexer like screen or tmux if available.  Another option is creating multiple interactive shells by running ``docker exec`` into the running container.

The first step is starting the FL server:

.. code-block:: shell

    $ ./poc/server/startup/start.sh

Once the server is running, open a new terminal and start the first client:

.. code-block:: shell

    $ ./poc/site-1/startup/start.sh

Open another terminal and start the second client:

.. code-block:: shell

    $ ./poc/site-2/startup/start.sh

In one last terminal, start the admin client:

.. code-block:: shell

  $ ./poc/admin/startup/fl_admin.sh localhost

This will launch a command prompt where you can input admin commands to control and monitor many aspects of
the FL process.

.. tip::

   For anything more than the most basic proof of concept examples, it is recommended that you use a
   :ref:`secure provisioned setup <provisioned_setup>`.

Deploying an example application
================================
After connecting the admin client in the previous section, you will see the admin CLI's prompt:

.. code-block:: shell

  login_result: OK
  Type ? to list commands; type "? cmdName" to show usage of a command.
  >

Typing ``?`` at the admin prompt will show the list of available commands.

.. note::
  
  Some commands require a password.  In POC mode, the admin password is ``admin``.
  
As an example, we can check the status of the server:

.. code-block:: shell

  > check_status server
  Engine status: stopped
  ---------------------
  | JOB_ID | APP NAME |
  ---------------------
  -------------------------
  Registered clients: 2 
  ----------------------------------------------------------------------------
  | CLIENT | TOKEN                                | LAST CONNECT TIME        |
  ----------------------------------------------------------------------------
  | site-1 | dedb907c-11d1-4235-a232-0b40d84dcebe | Tue May 24 12:49:15 2022 |
  | site-2 | 56b6ebc0-a414-40a8-aaf7-dc48a8d51440 | Tue May 24 12:48:57 2022 |
  ----------------------------------------------------------------------------
  Done [1752 usecs] 2022-05-24 12:49:20.921073

Now we can submit the hello-pt-tb job for execution:

.. code-block:: shell

  > submit_job hello-pt-tb

Now you can verify that the job has been submitted and clients started with

.. code-block:: shell

  > check_status client
  -------------------------------------------------------------------------
  | CLIENT | APP_NAME    | JOB_ID                               | STATUS  |
  -------------------------------------------------------------------------
  | site-1 | hello-pt-tb | aefdb0a3-6fbb-4c53-a677-b6951d6845a6 | started |
  | site-2 | hello-pt-tb | aefdb0a3-6fbb-4c53-a677-b6951d6845a6 | started |
  -------------------------------------------------------------------------
  Done [302546 usecs] 2022-05-24 13:09:27.815476

Please check :doc:`examples/hello_pt_tb` example for additional details on the structure of
the application and the configuration for streaming analytics.