############
Quick Start
############

Quick start means to help user get the FLARE running quickly without introducing a lot of concepts or discussions.
The intent is to let user get the first hand running experience quickly, then user can use :ref:`getting_started`. to
go through each step in detail

Since FLARE offers different mode of running system, we only cover the simplest approaches here.
In this quick start guide, we are using **examples/hello-world/hello-numpy-sag**. This is pre-defined examples.
you find the details in example's README.md file. For now, you will need to clone the github repo NVFLARE https://github.com/NVIDIA/NVFlare/tree/main (from main branch)
in order to get the examples

We also assume you have worked with python, already setup the virtual env. If you are new this, please refer :ref:`getting_started`.

#. **Install NVFLARE**

Install NVFLARE
.. code-block:: shell

  $ python3 -m pip install nvflare

clone NVFLARE repo to get examples, switch main branch (latest stable branch)

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git
  $ git switch main

#. **Quick start with CLI**
  create a temp directory as workspace
  Install requirements/dependencies

.. code-block:: shell
  $ mkdir -p /tmp/nvflare
  $ python3 -m pip install -r NVFlare/examples/hello-world/hello-numpy-sag/requirements.txt

   * **Quick Start with Simulator**

.. code-block:: shell

   nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 hello-numpy-sag

   watch the simulator run two clients (n = 2) with two threads (t = 2) and logs are saved in the /tmp/nvflare workspace

   * **Quick start with POC mode**
   instead of using Simulator, you can simulate the real deployment with multiple processes via POC mode

.. code-block:: shell

   $ nvflare poc --prepare -n 2
   $ nvflare poc --start -ex admin

   from another terminal, start FLARE console

.. code-block:: shell

   $ nvflare poc --start -p admin

   once FLARE Console started, you can check the status of the server


.. code-block:: console
   $ check_status server
   $ submit_job hello-world/hello-numpy-sag
