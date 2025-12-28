.. _simulation_guide:

###################################
How to Run Simulation Code in FLARE
###################################

NVIDIA FLARE provides several different types of simulations:

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

Here, ``SimEnv()`` represents the Simulation Environment. The recipe will execute in that environment.


Using FLARE Simulator Directly
------------------------------

If you prefer to use the command line, you can use the FLARE CLI to run the simulator.
Use the ``recipe.export(job_dir)`` method to export the job configuration to a job directory.

**FLARE CLI**

.. code-block:: bash

   nvflare simulator -w workspace -n <number_of_clients> -t <threads> -gpu <number_of_GPUs> <job_dir>

**Calling from Python**

You can also call the simulator directly from Python code using the ``SimulatorRunner.run()`` class method.

.. note::

   The simulator behavior might change with the Collaborative API release (coming soon).


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

Here, ``POCEnv()`` represents the PoC Environment. The recipe will execute in that environment.
If the PoC environment doesn't exist, it will create the PoC directory, start the clients and servers,
and then run the job.


Using PoC CLI Directly
----------------------

You can use the FLARE PoC CLI commands to prepare, start, and stop the PoC environment.

**FLARE CLI**

.. code-block:: bash

   nvflare poc prepare -n N
   nvflare poc start
   nvflare poc stop
   nvflare job submit -j <job_dir>


Additional Resources
====================

- Job Recipe API: :ref:`job_recipe`
- FLARE Simulator: :ref:`fl_simulator`
- Proof of Concept (PoC): :ref:`poc`
- PoC CLI Commands: :ref:`poc_command`
- FLARE CLI Reference: :ref:`nvflare_cli`
