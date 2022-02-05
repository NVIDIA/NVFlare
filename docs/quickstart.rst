.. _quickstart:

##########
Quickstart
##########

This section provides a starting point for new users to learn some basic concepts about NVIDIA FLARE.
Users can work through the examples at the bottom of this page and get familiar with how NVIDIA FLARE is designed,
operates and works.

Each example introduces concepts about NVIDIA FLARE, while showcasing how some popular libraries and frameworks can
easily be integrated into the FL process.

.. _setting_up_poc:

Setting Up the Application Environment in POC Mode
==================================================

To get started with a proof of concept (POC) setup after :ref:`installation`, run this command to generates a poc folder
with a server, two clients, and one admin:

.. code-block:: shell

    $ poc -n 2

Copy necessary files (the exercise code in the examples directory of the NVFlare repository) to a working folder (upload
folder for the admin):

.. code-block:: shell

  $ mkdir -p poc/admin/transfer
  $ cp -rf NVFlare/examples/* poc/admin/transfer

.. _starting_poc:

Starting the Application Environment in POC Mode
================================================

Once you are ready to start the FL system, you can run the following commands to start all the different parties (it is
recommended that you read into the specific example apps linked at the bottom of this page first, then start the FL
system to follow along at the parts with admin commands for you to run the example app).

FL systems usually have a server and multiple clients. We
therefore have to start the server first:

.. code-block:: shell

    $ ./poc/server/startup/start.sh

Once the server is running you can start the clients in different terminals (make sure your terminals are using the
environment with NVIDIA FLARE :ref:`installed <installation>`).
Open a new terminal and start the first client:

.. code-block:: shell

    $ ./poc/site-1/startup/start.sh

Open another terminal and start the second client:

.. code-block:: shell

    $ ./poc/site-2/startup/start.sh

In one last terminal, start the admin:

.. code-block:: shell

  $ ./poc/admin/startup/fl_admin.sh localhost

This will launch a command prompt where you can input admin commands to control and monitor many aspects of
the FL process. Log in by entering ``admin`` for both the username and password.

Custom Code in Example Apps
===========================

There are several ways to make :ref:`custom code <custom_code>` available to clients when using NVIDIA FLARE. Most
hello-* examples use a custom folder within the FL application. Note that using a custom folder in the app needs to be
:ref:`allowed <troubleshooting_byoc>` when using secure provisioning. By default, this option is disabled in the secure
mode. POC mode, however, will work with custom code by default.

In contrast, the `CIFAR-10 <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_,
`prostate segmentation <https://github.com/NVIDIA/NVFlare/tree/main/examples/prostate>`_,
and `BraTS18 segmentation <https://github.com/NVIDIA/NVFlare/tree/main/examples/brats18>`_ examples assume that the
learner code is already installed on the client's system and
available in the PYTHONPATH. Hence, the app folders do not include the custom code there. The PYTHONPATH is
set in the ``run_poc.sh`` or ``run_secure.sh`` scripts of the example. Running these scripts as described in the README
will make the learner code available to the clients.

.. _example_apps:

Example Apps for NVIDIA FLARE
=============================
NVIDIA FLARE has several examples to help you get started with federated learning and to explore certain features in
`the examples directory <https://github.com/NVIDIA/NVFlare/tree/main/examples>`_.

The following quickstart guides walk you through some of these examples:

.. toctree::
   :maxdepth: 1

   examples/hello_pt
   examples/hello_pt_tb
   examples/hello_numpy
   examples/hello_tf2
   examples/hello_cross_val
   Federated Learning with CIFAR-10 <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>

For the complete collection of example applications, see https://github.com/NVIDIA/NVFlare/tree/main/examples.
