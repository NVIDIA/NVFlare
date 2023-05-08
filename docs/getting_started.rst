.. _getting_started:

###############
Getting Started
###############

.. _quickstart:

Getting Started: Quick Start
============================

Install NVFLARE
---------------

.. code-block:: shell

  $ python3 -m pip install nvflare

Clone NVFLARE repo to get examples, switch main branch (latest stable branch)

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git
  $ cd NVFlare
  $ git switch main


Note on branches:

* The `dev <https://github.com/NVIDIA/NVFlare/tree/dev>`_ branch is the default (unstable) development branch

* The `main <https://github.com/NVIDIA/NVFlare/tree/main>`_ branch is the stable branch, reflecting the latest release

* The 2.0, 2.1, 2.2, and 2.3 etc. branches are the branches for each major release and minor patches


Quick Start with Simulator
--------------------------
Making sure the NVFLARE environment is set up correctly following :ref:`installation`, you can run an example application with :ref:`starting_fl_simulator`
using the following script:

.. code-block:: shell

   nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag

Now you can watch the simulator run two clients (n=2) with two threads (t=2)
and logs are saved in the `/tmp/nvflare/hello-numpy-sag` workspace.

Getting Started Guide
=====================

This Getting Started guide is geared towards new users of NVIDIA FLARE and walks through installation, the FL Simulator,
and a simple "hello world" application.

Once you're familiar with the platform, the :ref:`Example Applications <example_applications>` are a great next step.
These examples introduce some of the key concepts of the platform and showcase the integration of popular libraries
and frameworks like Numpy, Pytorch, Tensorflow, and MONAI.

Any FLARE application used with the FL Simulator can also be run in a real-world, distributed FL deployment.
The :ref:`Real-World FL <real_world_fl>` section describes some of the considerations and tools used for
establishing a secure, distributed FL workflow.

.. _installation:

Installation
=============

Python Version
--------------

NVIDIA FLARE requires Python 3.8+.

Install NVIDIA FLARE in a virtual environment
---------------------------------------------

It is highly recommended to install NVIDIA FLARE in a virtual environment if you are not using :ref:`containerized_deployment`.
This guide briefly describes how to create a virtual environment with venv.

Virtual Environments and Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python's official document explains the main idea about virtual environments.
The module used to create and manage virtual environments is called `venv <https://docs.python.org/3.8/library/venv.html#module-venv>`_.
You can find more information there.  We only describe a few necessary steps for a virtual environment for NVIDIA FLARE.


Depending on your OS and the Python distribution, you may need to install the Python's venv package separately.  For example, in Ubuntu
20.04, you need to run the following commands to continue creating a virtual environment with venv. Note that in newer versions of Ubuntu,
you may need to make sure you are using Python 3.8 and not a newer version.

.. code-block:: shell

   $ sudo apt update
   $ sudo apt-get install python3-venv


Once venv is installed, you can use it to create a virtual environment with:

.. code-block:: shell

    $ python3 -m venv nvflare-env

This will create the ``nvflare-env`` directory in current working directory if it doesn't exist,
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
use a similar installation to simplify transitioning between a bare metal and containerized
environment.

To get started with a containerized deployment, you will first need to install a supported
container runtime and the NVIDIA Container Toolkit to enable support for GPUs.  System requirements
and instructions for this can be found in the `NVIDIA Container Toolkit Install Guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_.

A simple Dockerfile is used to capture the base requirements and dependencies.  In
this case, we're building an environment that will support PyTorch-based workflows,
in particular the `Hello PyTorch <https://github.com/NVIDIA/NVFlare/tree/2.3/examples/hello-pt>`_
example. The base for this build is the NGC PyTorch container.  On this base image,
we will install the necessary dependencies and clone the NVIDIA FLARE GitHub
source code into the root workspace directory.

Let's first create a folder called ``build`` and then create a file inside named ``Dockerfile``:

.. code-block:: shell

  mkdir build
  cd build
  touch Dockerfile

Using any text editor to edit the Dockerfile and paste the following:

.. literalinclude:: resources/Dockerfile.doc
    :language: dockerfile

We can then build the new container by running docker build in the directory containing
this Dockerfile, for example tagging it nvflare-pt:

.. code-block:: shell

  docker build -t nvflare-pt . -f Dockerfile

This will result in a docker image, ``nvflare-pt:latest``.  You can run this container with Docker,
in this example mounting a local ``my-workspace`` directory into the container for use as a persistent
workspace:

.. code-block:: shell
  
  mkdir my-workspace
  docker run --rm -it --gpus all \
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      -w $(pwd -P)/my-workspace:/workspace/my-workspace \
      nvflare-pt:latest

Once the container is running, you can also exec into the container, for example if you need another
terminal to start additional FLARE clients.  First find the ``CONTAINER ID`` using ``docker ps``, and then
use that ID to exec into the container:

.. code-block:: shell

  docker ps  # use the CONTAINER ID in the output
  docker exec -it <CONTAINER ID> /bin/bash
  
This container can be used to run the FL Simulator or any FL server or client.  When using the
FL Simulator (described in the next section), you can simply mount in any directories needed for
your FLARE application code, and run the Simulator within the Docker container with
all dependencies installed.

Ways to Run NVFLARE
===================
NVFLARE can currently support running with the FL Simulator, POC mode, or Production mode.

FL Simulator is lightweight and uses threads to simulate different clients.
The code used for the simulator can be directly used in production mode.

POC mode is an insecure deployment run locally on one machine without worry about TLS certificates. Each client 
and Server are running on different processes

Production mode is secure with TLS certificates - depending the choice the deployment, you can further choose:

  - HA or non-HA
  - Local or remote
  - On-premise or on cloud

Using non-HA, secure, local mode (all clients and server running on the same host), production mode is very similar to POC mode except it is secure.

Which mode should I choose for running NVFLARE?

  - For a quick research run, use the FL Simulator
  - For simulating real cases within the same machine, use POC or production (local, non-HA, secure) mode. POC has convenient ``nvflare poc`` commands for ease of use.
  - For all other cases, use production mode.

.. _starting_fl_simulator:

The FL Simulator
=========================

After installing the nvflare pip package, you have access to the NVFlare CLI including the FL Simulator.
The Simulator allows you to start a FLARE server and any number of connected clients on your local
workstation or laptop, and to quickly deploy an application for testing and debugging.

Basic usage for the FL Simulator is available with ``nvflare simulator -h``:

.. code-block:: shell

  $ nvflare simulator -h
  usage: nvflare simulator [-h] [-w WORKSPACE] [-n N_CLIENTS] [-c CLIENTS] [-t THREADS] [-gpu GPU] [-m MAX_CLIENTS] job_folder
  
  positional arguments:
    job_folder
  
  optional arguments:
    -h, --help            show this help message and exit
    -w WORKSPACE, --workspace WORKSPACE
                          WORKSPACE folder
    -n N_CLIENTS, --n_clients N_CLIENTS
                          number of clients
    -c CLIENTS, --clients CLIENTS
                          client names list
    -t THREADS, --threads THREADS
                          number of parallel running clients
    -gpu GPU, --gpu GPU   list of GPU Device Ids, comma separated
    -m MAX_CLIENTS, --max_clients MAX_CLIENTS
                          max number of clients


Before we get into the Simulator, we'll walk through a few additional setup steps in the next section required
to run an example application.


Running an example application
================================

Any of the :ref:`example_applications` can be used with the FL Simulator.  We'll demonstrate the steps here
using the hello-pt example.

First, we need to clone the NVFlare repo to get the source code for the examples:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

We can then copy the necessary files (the exercise code in the examples directory of the NVFlare repository)
to a working directory:

.. code-block:: shell

  mkdir simulator-example
  cp -rf NVFlare/examples/hello-pt simulator-example/

The hello-pt application requires a few dependencies to be installed.  As in the installation section,
we can install these in the Python virtual environment by running:

.. code-block:: shell

  source nvflare-env/bin/activate
  python3 -m pip install -r simulator-example/requirements.txt

If using the Dockerfile above to run in a container, these dependencies have already been installed.

Next, we can create a workspace for the Simulator to use for outputs of the application run, and launch
the simulator using ``simulator-example/hello-pt`` as the input job directory.  In this example, we'll
run on two clients using two threads:

.. code-block:: shell

  mkdir simulator-example/workspace
  nvflare simulator -w simulator-example/workspace -n 2 -t 2 simulator-example/hello-pt

Now you will see output streaming from the server and client processes as they execute the federated
application.  Once the run completes, your workspace directory will contain the input application configuration
and codes, logs of the output, site and global models, cross-site validation results.

.. code-block:: shell
  
  $ tree -L 3 simulator-example/workspace/
  simulator-example/workspace/
  ├── audit.log
  ├── local
      │  └── log.config
      ├── simulate_job
      │  ├── app_server
      │  │   ├── config
      │  │   ├── custom
      │  │   └── FL_global_model.pt
      │  ├── app_site-1
      │  │   ├── audit.log
      │  │   ├── config
      │  │   ├── custom
      │  │   └── log.txt
      │  ├── app_site-2
      │  │   ├── audit.log
      │  │   ├── config
      │  │   ├── custom
      │  │   └── log.txt
      │  ├── cross_site_val
      │  │   ├── cross_val_results.json
      │  │   ├── model_shareables
      │  │   └── result_shareables
      │  ├── log.txt
      │  ├── models
      │  │   └── local_model.pt
      │  └── tb_events
      │      ├── site-1
      │      └── site-2
      └── startup


Now that we've explored an example application with the FL Simulator, we can look at what it takes to bring
this type of application to a secure, distributed deployment in the :ref:`Real World Federated Learning <real_world_fl>`
section.


.. _setting_up_poc:

Setting Up the Application Environment in POC Mode
==================================================

.. warning::

    POC mode is not intended to be secure and should not be run in any type of production environment or any environment
    where the server's ports are exposed. For actual deployment and even development, it is recommended to use a
    :ref:`secure provisioned setup <provisioned_setup>` or :ref:`starting_fl_simulator`.

To get started with a proof of concept (POC) setup after :ref:`installation`, run this command to generate a poc folder
with an overseer, server, two clients, and one admin client:

.. code-block:: shell

    $ nvflare poc --prepare -n 2

For more details, see :ref:`poc_command`.

.. _starting_poc:

Starting the Application Environment in POC Mode
================================================

Once you are ready to start the FL system, you can run the following command
to start the server and client systems and an admin console:

.. code-block::

  nvflare poc --start

To start the server and client systems without an admin console:

.. code-block::

  nvflare poc --start -ex admin

For more details, see :ref:`poc_command`.

.. tip::

   For anything more than the most basic proof of concept examples, it is recommended that you use a
   :ref:`secure provisioned setup <provisioned_setup>`.
