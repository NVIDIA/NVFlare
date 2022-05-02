.. _quickstart:

##########
Quickstart
##########

This section provides a starting point for new users to start NVIDIA FLARE.
Users can go through the :ref:`example_apps` and get familiar with how NVIDIA FLARE is designed,
operates and works.

Each example introduces concepts about NVIDIA FLARE, while showcasing how some popular libraries and frameworks can
easily be integrated into the FL process.

.. _setting_up_poc:

Setting Up the Application Environment in POC Mode
==================================================

.. warning::

    POC mode is not intended to be secure and should not be run in any type of production environment or any environment
    where the server's ports are exposed. For actual deployment and even development, it is recommended to use a
    :ref:`secure provisioned setup <provisioned_setup>`.

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
the FL process.

.. tip::

   For anything more than the most basic proof of concept examples, it is recommended that you use a
   :ref:`secure provisioned setup <provisioned_setup>`.
