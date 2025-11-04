
.. _job_recipe:

NVFlare Job Recipe
==================

This tutorial covers how to use Job Recipes in NVFlare to simplify federated learning job creation and execution. 
Job Recipes provide a simplified abstraction that hides the complexity of low-level job configurations while exposing only the key arguments users should care about.

.. note::
   This is a technical preview. Not all algorithms are currently implemented with recipes.


Motivation for Using JobRecipe
------------------------------

The **Job API** provides a powerful and flexible way to define FLARE FL workflows and configurations in Python without manually editing configuration files. While the API simplified the process compared to previous approaches, it is not simple enough. For new users and data scientists working with standard pipelines, learning detailed concepts such as controllers, executors, workflows, and how to wire them together is unnecessary.

To address this, NVFlare introduces the concept of **Job Recipes**. A ``JobRecipe`` is a simplified abstraction designed to provide a high-level API with:

* **Only the key arguments** a data scientist should care about, such as the number of clients, number of rounds, training scripts, and model definition.
* **Consistent entry points** for common federated learning patterns such as **FedAvg** and **Cyclic Training**.
* **Execution environments** from simulation to production for the same job.

This makes ``JobRecipe`` particularly useful as a **first touchpoint** for new users and data scientists working with standard pipelines:

* Instead of learning the entire Job API, users can start with a recipe and focus only on high-level parameters (e.g., ``min_clients``, ``num_rounds``).
* Recipes encapsulate the necessary job structure and execution logic, ensuring correctness while reducing the chance of misconfiguration.
* If necessary, users can later progress to customizing the full Job API once they are comfortable with the basics.

Basic Example
-------------

Let's start with a simple example using the ``FedAvgRecipe`` for PyTorch. This recipe automatically handles all the complexity of setting up a federated averaging workflow.

We use our existing training network under ``../hello-world/hello-pt/model.py`` and script ``client.py`` to generate the recipe:

.. code-block:: python

   import os
   import sys
   sys.path.append("../hello-world/hello-pt")

   from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
   from model import SimpleNetwork

   # Create a FedAvg recipe
   recipe = FedAvgRecipe(
       name="hello-pt",
       min_clients=2,
       num_rounds=3,
       initial_model=SimpleNetwork(),
       train_script="client.py",
       train_args="--batch_size 32",
   )

   print("Recipe created successfully!")
   print(f"Recipe name: {recipe.name}")
   print(f"Min clients: {recipe.min_clients}")
   print(f"Number of rounds: {recipe.num_rounds}")

Execution Environments
----------------------

A **Job Recipe** defines *what* to run in a federated learning setting, but it also needs to know *where* to run. NVFlare provides several **execution environments** that allow the same recipe to be executed in different contexts:

* **Simulation (** ``SimEnv`` **)** – For local testing and experimentation on a single machine
* **Proof-of-Concept (** ``PocEnv`` **)** – For small-scale, multi-process setups that mimic real-world deployment on a single machine
* **Production (** ``ProdEnv`` **)** – For full-scale distributed deployments across multiple organizations and sites

This separation enables users to **prototype once and deploy anywhere** without modifying the core job definition.

SimEnv – Simulation Environment
-------------------------------

Runs all clients and the server as **threads** within a single process. This is lightweight and easy to set up with no networking required. Best suited for:

* Quick experiments
* Debugging scripts and models
* Educational use cases

**Arguments:**

* ``num_clients`` (int): Number of simulated clients
* ``clients``: A list of client names (length needs to match ``num_clients`` if both are provided)
* ``num_threads``: Number of threads to use to run simulated clients
* ``gpu_config`` (str): List of GPU device IDs, comma separated
* ``log_config`` (str): Log config mode (``'concise'``, ``'full'``, ``'verbose'``), filepath, or level

Now let's test running the prepared recipe with ``SimEnv``:

.. code-block:: python

   from nvflare.recipe.sim_env import SimEnv
   # Create a simulation environment
   env = SimEnv(
       num_clients=2, 
       num_threads=2,
   )
   # Execute the recipe
   run = recipe.execute(env=env)
   run.get_status()
   run.get_result()

The result is stored under ``/tmp/nvflare/simulation/hello-pt``.

PocEnv – Proof-of-Concept Environment
-------------------------------------

Runs server and clients as **separate processes** on the same machine. This simulates real-world deployment within a single node, with server and clients running in different processes. More realistic than ``SimEnv``, but still lightweight enough for a single node.

Best suited for:

* Demonstrations
* Small-scale validation before production deployment
* Debugging orchestration logic

**Arguments:**

* ``num_clients`` (int, optional): Number of clients to use in POC mode. Defaults to 2.
* ``clients`` (List[str], optional): List of client names. If ``None``, will generate ``site-1``, ``site-2``, etc.
* ``gpu_ids`` (List[int], optional): List of GPU IDs to assign to clients. If ``None``, uses CPU only.
* ``auto_stop`` (bool, optional): Whether to automatically stop POC services after job completion.
* ``use_he`` (bool, optional): Whether to use HE. Defaults to ``False``.
* ``docker_image`` (str, optional): Docker image to use for POC.
* ``project_conf_path`` (str, optional): Path to the project configuration file.

Let's first set the path to the POC environment:

.. code-block:: shell

   %env NVFLARE_POC_WORKSPACE=/tmp/nvflare/poc

.. code-block:: python

   from nvflare.recipe.poc_env import POCEnv

   # Create a POC environment
   env = POCEnv(
       num_clients=2
   )
   # Execute the recipe
   run = recipe.execute(env=env)
   run.get_status()
   run.get_result()

The result is stored under the directory ``/tmp/nvflare/poc``.

ProdEnv – Production Environment
--------------------------------

We assume a system with a server and clients is up and running across **multiple machines and sites**. This environment uses secure communication channels and real-world NVFlare deployment infrastructure. ``ProdEnv`` utilizes the admin's startup package to communicate with an existing NVFlare system to execute and monitor job execution.

Best suited for:

* Enterprise federated learning deployments
* Multi-institution collaborations
* Production-scale workloads

**Arguments:**

* ``startup_kit_location`` (str): The directory that contains the startup kit of the admin (generated by nvflare provisioning)
* ``login_timeout`` (float): Timeout value for the admin to login to the system
* ``monitor_job_duration`` (int): Duration to monitor the job execution. ``None`` means no monitoring at all

Let's first provision a startup kit:

.. code-block:: shell

   !nvflare provision -p project.yml -w /tmp/nvflare/prod_workspaces

Let's then start all parties (from terminal, rather than running the below script directly within notebook):

.. code-block:: shell

   bash /tmp/nvflare/prod_workspaces/example_project/prod_00/start_all.sh

Now let's go ahead with environment creation and recipe execution.

.. code-block:: python

   from nvflare.recipe.prod_env import ProdEnv
   import os
   import sys
   sys.path.append("../hello-world/hello-pt")

   from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
   from model import SimpleNetwork

   # Create a FedAvg recipe
   recipe = FedAvgRecipe(
       name="hello-pt",
       min_clients=2,
       num_rounds=3,
       initial_model=SimpleNetwork(),
       train_script="client.py",
       train_args="--batch_size 32",
   )
   # Create a Prod environment
   env = ProdEnv(
       startup_kit_location="/tmp/nvflare/prod_workspaces/example_project/prod_00/admin@nvidia.com"
   )
   # Execute the recipe
   run = recipe.execute(env=env)
   run.get_status()
   run.get_result()

Benefits of Environment Abstraction
-----------------------------------

* **Consistency** – A recipe defined once can be reused across all environments without modification.
* **Progressive workflow** – Start in ``SimEnv`` for prototyping, move to ``PocEnv`` for validation, and finally deploy with ``ProdEnv``.
* **Scalability** – The same training logic scales from a laptop experiment to a global production deployment.

Special Considerations for Edge Applications
---------------------------------------------

Edge applications running with the new hierarchical system are not supported by the simulator and at the current version must run with ``ProdEnv``. Please see more detailed examples `here <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/edge>`_. In particular, see the edge recipe preparation and experimental run in `this example <https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/edge/jobs/pt_job_adv.py>`_.

Best Practices
--------------

1. **Develop in** ``SimEnv`` to iterate quickly.
2. **Validate in** ``PocEnv`` to test multi-process orchestration.
3. **Deploy in** ``ProdEnv`` for real-world federated learning.
4. **Start simple** with basic recipes before customizing.
5. **Use consistent naming** for your recipes and experiments.
6. **Monitor execution** to understand the federated learning process.

Summary
-------

Job Recipes, combined with execution environments, provide a **unified abstraction** for defining and running federated learning jobs:

* **Recipes define how training should proceed** (e.g., FedAvg, FedOpt, Swarm Learning)
* **Environments define where and how the job runs** (simulation, proof-of-concept, production)

This separation ensures that the same recipe can seamlessly transition from **local testing** to **enterprise-scale production** without requiring code changes.

The goal of Job Recipes is to create a simple entry point into NVFlare that is most intuitive for new users and data scientists running standard FL pipelines, while still allowing for growth into more complex and customizable workflows.

Examples
--------
To see more examples of Job Recipe in action, check out the quick start series :ref:`quickstart`, where several job recipes are demonstrated.

