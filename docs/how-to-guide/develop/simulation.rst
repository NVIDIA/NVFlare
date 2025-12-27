.. _simulation_guide:

###################################
How to run simulation code in FLARE
###################################

NVIDIA FLARE provide several different type of simulations:

- Simulate FLARE System in local host: Simulator
  Running client and server directly with multi-thread and processes without worry about the actual deployment

- Simulate FLARE deployment in local host: Proof of Concept (PoC) mode.
  Simulate actual deployment with provisioning, where clients and server are separated into different folders (startup kits)
  User can interact with FLARE system just like user did with production FLARE system.


How to run simulation with Simulator
=====================================

Running JobRecipe in SimEnv (Recommended)
-----------------------------------------
To run simulation, we recommend to use JobRecipe with Simulation Env.

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

  Here the SimEnv() represents the Simulation env. The recipe will execute in that environment.


Directly using FLARE Simulator
------------------------------

If user prefers to use commandline, one can use FLARE CLI to run simulator.
User can use recipe.export(job_dir) method export job configuration to Job_dir

  **FLARE CLI**

.. code-block:: text
   nvflare simulator -w workspace -n <number of clients> -t <threads> -gpu <number of GPUs> <job_dir>

   **Call from Python**
   use can also call directly from python code to `SimulatorRunner.run()` class method

.. note::
   The simulator behavior might change with Collaborator API release (coming soon)

How to run job in POC mode
==========================

Running JobRecipe in PocEnv (Recommended)
-----------------------------------------

To run simulation, we recommend to use JobRecipe with PoC Env.

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

Here the POCEnv() represents the PoC env. The recipe will execute in that environment,
If PoC env doesn't exists, if will create PoC directory and start the clients and servers in PoC, then run the job.


Directly using Running in POC
-----------------------------
You can use FLARE PoC Cli command to prepare, start/stop PoC env.

  **FLARE CLI**

.. code-block:: text
   nvflare poc prepare -n N
   nvflare poc start
   nvflare poc stop
   nvflare job submit -j <job_dir>








