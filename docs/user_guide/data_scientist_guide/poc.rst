.. _poc:

POC: Prove of Concept: Simulate Production deployment locally
=============================================================


.. _setting_up_poc:

Setting Up the Application Environment in POC Mode
--------------------------------------------------

To get started with a proof of concept (POC) setup after :ref:`installation`, run this command to generate a poc folder
with an server, two clients, and one admin client:

.. code-block:: shell

    $ nvflare poc prepare -n 2

For more details, see :ref:`poc_command`.

.. _starting_poc:

Starting the Application Environment in POC Mode
--------------------------------------------------

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

For `POC Tutorials: <https://github.com/NVIDIA/NVFlare/tree/main/examples/tutorials/setup_poc.ipynb>`_
