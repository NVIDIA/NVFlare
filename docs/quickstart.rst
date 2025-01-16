##########
Quickstart
##########

.. _installation:

Installation
=============

.. note::
   The server and client versions of nvflare must match, we do not support cross-version compatibility.

Supported Operating Systems
---------------------------
- Linux
- OSX (Note: some optional dependencies are not compatible, such as tenseal and openmined.psi)

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
20.04, you need to run the following commands to continue creating a virtual environment with venv.

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

Install Stable Release of NVFlare
---------------------------------

Stable releases are available on `NVIDIA FLARE PyPI <https://pypi.org/project/nvflare>`_:

.. code-block:: shell

  $ python3 -m pip install nvflare

.. note::

    In addition to the dependencies included when installing nvflare, many of our example applications have additional packages that must be installed.
    Make sure to install from any requirement.txt files before running the examples. If you already have a specific version of nvflare installed in your
    environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare.
    See :github_nvflare_link:`nvflare/app_opt <nvflare/app_opt>` for modules and components with optional dependencies.

Cloning the NVFlare Repository and Checking Out a Branch
---------------------------------------------------------

Clone NVFlare repo to get examples, and switch to either the main branch or the latest stable branch:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git
  $ cd NVFlare
  $ git switch 2.5

Note on branches:

* The `main <https://github.com/NVIDIA/NVFlare/tree/main>`_ branch is the default (unstable) development branch

* The 2.1, 2.2, 2.3, 2.4, 2.5, etc. branches are the branches for each major release and there are tags based on these with a third digit for minor patches

Install NVFlare from source
----------------------------

Navigate to the NVFlare repository and use pip install with development mode (can be useful to access latest nightly features or test custom builds for example):

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git
  $ cd NVFlare
  $ pip install -e .


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
in particular the :github_nvflare_link:`Hello PyTorch <examples/hello-world/hello-pt>`
example. The base for this build is the NGC PyTorch container.  On this base image,
we will install the necessary dependencies and clone the NVIDIA FLARE GitHub
source code into the root workspace directory.

Let's first create a folder called ``build`` and then create a file inside named ``Dockerfile``:

.. code-block:: shell

  mkdir build
  cd build
  touch Dockerfile

Using any text editor to edit the Dockerfile and paste the following:

.. literalinclude:: resources/Dockerfile
    :language: dockerfile

.. note::

    For nvflare version 2.5 set PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3

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

For a notebook showcasing this example, see the :github_nvflare_link:`NVIDIA FLARE with Docker example <examples/advanced/docker>`.

Ways to Run NVFlare
===================
NVFlare can currently support running with the FL Simulator, POC mode, or Production mode.

FL Simulator is lightweight and uses threads to simulate different clients.
The code used for the simulator can be directly used in production mode.

Starting in 2.5, NVFlare supports running the FL Simulator with the Job API. The :ref:`Job API <fed_job_api>` allows
you to build jobs programatically and then export them or directly run them with the simulator.

POC mode is a quick way to get set up to run locally on one machine. The FL server and each client
run on different processes or dockers.

Production mode is secure with TLS certificates - depending the choice the deployment, you can further choose:

  - HA or non-HA
  - Local or remote
  - On-premise or on cloud (See :ref:`cloud_deployment`)

Using non-HA, secure, local mode (all clients and server running on the same host), production mode is very similar to POC mode except it is secure.

Which mode should I choose for running NVFLARE? (Note: the same jobs can be run in any of the modes, and the same project.yml deployment options can be run in both POC mode and production.)

.. list-table:: NVIDIA FLARE Modes
   :header-rows: 1

   * - **Mode**
     - **Documentation**
     - **Description**
   * - Simulator
     - :ref:`fl_simulator`
     - | The FL Simulator is a light weight simulation where the job run is automated on a 
       | single system. Useful for quickly running a job or experimenting with research 
       | or FL algorithms.
   * - POC
     - :ref:`poc_command`
     - | POC mode establishes and connects distinct server and client "systems" which can 
       | then be orchestrated using the FLARE Console all from a single machine. Users can 
       | also experiment with various deployment options (project.yml), which can be used 
       | in production modes.
   * - Production
     - :ref:`provisioned_setup`
     - | Real world production mode involves a distributed deployment with generated startup 
       | kits from the provisioning process. Provides provisioning tool, dashboard, and 
       | various deployment options.

.. _starting_fl_simulator:

The FL Simulator
=========================

After installing the nvflare pip package, you have access to the NVFlare CLI including the FL Simulator.
The Simulator allows you to start a FLARE server and any number of connected clients on your local
workstation or laptop, and to quickly deploy an application for testing and debugging.

Basic usage for the :ref:`FL Simulator <fl_simulator>` is available with ``nvflare simulator -h``:

.. code-block:: shell

  $ nvflare simulator -h
  usage: nvflare simulator [-h] [-w WORKSPACE] [-n N_CLIENTS] [-c CLIENTS] [-t THREADS] [-gpu GPU] [-m MAX_CLIENTS] [--end_run_for_all] job_folder

  positional arguments:
    job_folder

  options:
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
    --end_run_for_all     flag to indicate if running END_RUN event for all clients


Before we get into the Simulator, we'll walk through a few additional setup steps in the next section required
to run an example application.


Running an example application
================================

Any of the :ref:`example_applications` can be used with the FL Simulator.  We'll demonstrate the steps here
using the hello-pt example.

First, we need to clone the NVFlare repo to get the source code for the examples:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git


Please make sure to switch to the correct branch that matches the NVFlare library version you installed.

.. code-block:: shell

  $ git switch [nvflare version]


We can then copy the necessary files (the exercise code in the examples directory of the NVFlare repository)
to a working directory:

.. code-block:: shell

  mkdir simulator-example
  cp -rf NVFlare/examples/hello-world/hello-pt simulator-example/

The hello-pt application requires a few dependencies to be installed.  As in the installation section,
we can install these in the Python virtual environment by running:

.. code-block:: shell

  source nvflare-env/bin/activate
  python3 -m pip install -r simulator-example/hello-pt/requirements.txt

If using the Dockerfile above to run in a container, these dependencies have already been installed.

Next, we can directly run the ``fedavg_script_runner_pt.py`` script which is configured to build a job
with the Job API and then run it with the FL Simulator.

.. code-block:: shell

  cd simulator-example/hello-pt
  python3 fedavg_script_runner_pt.py

Now you will see output streaming from the server and client processes as they execute the federated
application.  Once the run completes, your workspace directory (by default ``/tmp/nvflare/jobs/workdir``),
will contain the input application configuration
and codes, logs of the output, site and global models, cross-site validation results.

.. code-block:: shell

  $ tree -L 4 /tmp/nvflare/jobs/workdir
  /tmp/nvflare/jobs/workdir
  ├── server
  │   ├── local
  │   │   └── log_config.json
  │   ├── log.txt
  │   ├── pool_stats
  │   │   └── simulator_cell_stats.json
  │   ├── simulate_job
  │   │   ├── app_server
  │   │   │   ├── FL_global_model.pt
  │   │   │   ├── config
  │   │   │   └── custom
  │   │   ├── cross_site_val
  │   │   │   └── cross_val_results.json
  │   │   ├── meta.json
  │   │   └── tb_events
  │   │       ├── site-1
  │   │       └── site-2
  │   └── startup
  ├── site-1
  │   ├── cifar_net.pth
  │   ├── local
  │   │   └── log_config.json
  │   ├── log.txt
  │   ├── simulate_job
  │   │   ├── app_site-1
  │   │   │   ├── config
  │   │   │   └── custom
  │   │   └── meta.json
  │   └── startup
  ├── site-2
  │   ├── cifar_net.pth
  │   ├── local
  │   │   └── log_config.json
  │   ├── log.txt
  │   ├── simulate_job
  │   │   ├── app_site-2
  │   │   │   ├── config
  │   │   │   └── custom
  │   │   └── meta.json
  │   └── startup
  └── startup


Now that we've explored an example application with the FL Simulator, we can look at what it takes to bring
this type of application to a secure, distributed deployment in the :ref:`Real World Federated Learning <real_world_fl>`
section.


.. _setting_up_poc:

Setting Up the Application Environment in POC Mode
==================================================

To get started with a proof of concept (POC) setup after :ref:`installation`, run this command to generate a poc folder
with an overseer, server, two clients, and one admin client:

.. code-block:: shell

    $ nvflare poc prepare -n 2

For more details, see :ref:`poc_command`.

.. _starting_poc:

Starting the Application Environment in POC Mode
================================================

Once you are ready to start the FL system, you can run the following command
to start the server and client systems and an admin console:

.. code-block::

  nvflare poc start

To start the server and client systems without an admin console:

.. code-block::

  nvflare poc start -ex admin@nvidia.com

We can use the :ref:`job_cli` to easily submit a job to the POC system. (Note: We can run the same jobs we ran with the simulator in POC mode. If using the :ref:`fed_job_api`, simply export the job configuration with ``job.export_job()``.)

.. code-block::

  nvflare job submit -j NVFlare/examples/hello-world/hello-numpy-sag/jobs/hello-numpy-sag

.. code-block::

  nvflare poc stop

.. code-block::

  nvflare poc clean

For more details, see :ref:`poc_command`.
