.. _simulation_guide:

###################################
How to Run Simulation Code in FLARE
###################################

NVIDIA FLARE provides two simulation modes for local development and testing:

- **Simulator**: Simulates the FLARE system on a local host by running clients and server directly
  with multi-threading and processes, without worrying about actual deployment.

- **Proof of Concept (PoC) Mode**: Simulates FLARE deployment on a local host with provisioning,
  where clients and server are separated into different folders (startup kits). Users can interact
  with the FLARE system just as they would with a production FLARE system.


How to Run Simulation with Simulator
====================================

Running Job Recipe in SimEnv (Recommended)
------------------------------------------

To run a simulation, we recommend using Job Recipe with the Simulation Environment.

.. code-block:: python

   recipe = FedAvgRecipe(
       name="hello-pt",
       min_clients=n_clients,
       num_rounds=num_rounds,
       initial_model=SimpleNetwork(),
       train_script="client.py",
       train_args=f"--batch_size {batch_size}",
   )

   env = SimEnv(num_clients=n_clients, num_threads=n_clients)
   recipe.execute(env=env)

The ``SimEnv`` class represents the simulation environment. The recipe executes in this environment,
running all clients and the server in a single process with multi-threading.


Using FLARE Simulator Directly
------------------------------

If you prefer to use the command line, you can use the FLARE CLI to run the simulator.
First, use the ``recipe.export(job_dir)`` method to export the job configuration to a job directory,
then run:

.. code-block:: bash

   nvflare simulator -w workspace -n <number_of_clients> -t <threads> -gpu <number_of_GPUs> <job_dir>

Alternatively, you can call the simulator directly from Python using the ``SimulatorRunner.run()`` method.

.. note::

   The simulator behavior may be updated with the upcoming Collaborative API release.


How to Run Jobs in PoC Mode
===========================

Running Job Recipe in PocEnv (Recommended)
------------------------------------------

To run a simulation in PoC mode, we recommend using Job Recipe with the PoC Environment.

.. code-block:: python

   recipe = FedAvgRecipe(
       name="hello-pt",
       min_clients=n_clients,
       num_rounds=num_rounds,
       initial_model=SimpleNetwork(),
       train_script="client.py",
       train_args=f"--batch_size {batch_size}",
   )

   env = POCEnv(num_clients=2)
   recipe.execute(env=env)

The ``POCEnv`` class represents the PoC environment. If the environment doesn't exist, it will
automatically create the PoC directory, start the clients and server, and then run the job.


Using PoC CLI Directly
----------------------

You can use the FLARE PoC CLI commands to prepare, start, and stop the PoC environment:

.. code-block:: bash

   nvflare poc prepare -n N    # Prepare PoC with N clients
   nvflare poc start           # Start the PoC environment
   nvflare job submit -j <job_dir>  # Submit a job
   nvflare poc stop            # Stop the PoC environment


References
==========

- :ref:`job_recipe` - Job Recipe API documentation
- :ref:`fl_simulator` - FLARE Simulator reference
- :ref:`poc` - Proof of Concept (PoC) guide
- :ref:`poc_command` - PoC CLI commands
- :ref:`nvflare_cli` - FLARE CLI reference
