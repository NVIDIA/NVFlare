
.. _job_recipe:

NVFlare Job Recipe
==================

This tutorial covers how to use Job Recipes in NVFlare to simplify federated learning job creation and execution. 
Job Recipes provide a simplified abstraction that hides the complexity of low-level job configurations while exposing only the key arguments users should care about.

.. note::
   This is a technical preview. Not all algorithms are currently implemented with recipes.

For the stable public Recipe surface, see :ref:`recipe_api`.


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

Model Input Options
-------------------

Recipes accept model input in two formats, each with different trade-offs:

**Option 1: Class Instance (Recommended for simplicity)**

.. code-block:: python

   from nvflare.app_opt.pt.recipes import FedAvgRecipe
   from model import SimpleNetwork

   recipe = FedAvgRecipe(
       name="hello-pt",
       model=SimpleNetwork(),  # Instantiated model
       train_script="client.py",
       ...
   )

**Option 2: Dictionary Configuration (Recommended for large models)**

.. code-block:: python

   recipe = FedAvgRecipe(
       name="hello-pt",
       model={
           "class_path": "model.SimpleNetwork",
           "args": {"num_classes": 10, "hidden_dim": 256}
       },
       train_script="client.py",
       ...
   )

.. important::

   **Understanding Model Serialization**

   When you pass a class instance (e.g., ``SimpleNetwork()``), NVFlare does **not** ship the Python object directly.
   Instead, the model is converted to a configuration file before job submission. The actual model is re-instantiated
   on the server/clients from this configuration.

   This means:

   * **Large models**: Instantiating a large model (e.g., LLM with billions of parameters) just to create a recipe
     is inefficient. Use the dictionary format to avoid unnecessary instantiation time and memory usage.
   * **Non-serializable state**: If your model carries state that cannot be reconstructed from JSON configuration
     (e.g., loaded data, open file handles), that state will be lost.
   * **TensorFlow/Keras class instances**: Use a user-defined subclass (for example, subclassing
     ``tf.keras.Model`` or ``tf.keras.Sequential``) so the model can be reconstructed from class path and args.
     Passing raw inline Keras model objects may fail during job export.
   * **Trade-off**: Class instance is more Pythonic and catches errors early; dictionary format is more performant
     for large models.

Pre-trained Checkpoint Path
---------------------------

Use ``initial_ckpt`` to specify a path to pre-trained model weights:

.. code-block:: python

   recipe = FedAvgRecipe(
       name="hello-pt",
       model=SimpleNetwork(),
       initial_ckpt="/data/models/pretrained_model.pt",  # Absolute path
       train_script="client.py",
       ...
   )

.. important::

   **Checkpoint Path Requirements**

   * **Absolute path required**: The path must be an absolute path (e.g., ``/data/models/model.pt``), not relative.
   * **May not exist locally**: The checkpoint file does **not** need to exist on the machine where you create
     the recipe. It only needs to exist on the **server** when the model is actually loaded during job execution.
   * **PyTorch requires model architecture**: For PyTorch, you must provide ``model`` (class instance or
     dict config) along with ``initial_ckpt``, because PyTorch checkpoints contain only weights, not architecture.
   * **PyTorch update schema**: The server-side PyTorch model or checkpoint defines the accepted
     ``state_dict()`` key schema for client updates. A client may return only the subset of keys it trained,
     but every returned key must already exist in the server schema. New client-only keys are rejected.
   * **TensorFlow/Keras can use checkpoint alone**: Keras ``.h5`` or SavedModel formats contain both architecture
     and weights, so ``initial_ckpt`` can be used without ``model``. If ``model`` is provided, use a subclassed
     Keras class instance (or dict config).

**Example: Resume training from pre-trained weights**

.. code-block:: python

   # PyTorch: requires both model and checkpoint
   recipe = FedAvgRecipe(
       model=SimpleNetwork(),
       initial_ckpt="/server/path/to/pretrained.pt",
       ...
   )

   # TensorFlow: checkpoint alone works (Keras saves full model)
   recipe = FedAvgRecipe(
       initial_ckpt="/server/path/to/pretrained.h5",
       framework=FrameworkType.TENSORFLOW,
       ...
   )

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
       model=SimpleNetwork(),
       train_script="client.py",
       train_args="--batch_size 32",
   )

   print("Recipe created successfully!")
   print(f"Recipe name: {recipe.name}")
   print(f"Min clients: {recipe.min_clients}")
   print(f"Number of rounds: {recipe.num_rounds}")

Metrics Artifacts
-----------------

Training aggregation recipes write standard metrics artifacts when their server
workflow reports round-level aggregation metrics. See
:ref:`recipe_metrics_artifacts` for the schema, security behavior, and how
tools locate the artifacts.

Per-Site Configuration
----------------------

Some recipes accept site-keyed configuration so that each site can use different
arguments, scripts, or data loaders. Call ``set_per_site_config`` immediately
after constructing the recipe, before adding client configuration, files,
filters, components, or tracking:

.. code-block:: python

   from nvflare.recipe import SimEnv, set_per_site_config

   set_per_site_config(
       recipe,
       {
           "site-1": {"train_args": "--data_path xxx --batch_size 4"},
           "site-2": {"train_args": "--data_path yyy --batch_size 2"},
       },
   )

   env = SimEnv(clients=recipe.configured_sites())

The helper validates and stores the mapping; it does not build and then replace
client apps. The recipe materializes its client topology once, before the first
client-targeted customization or before export or execution. Built-in FedAvg
recipes and ``FedEvalRecipe`` create one app per configured site directly. If
per-site configuration is omitted, they create the default ``@ALL`` app at that
same preparation point. XGBoost bagging, horizontal, and vertical recipes add
the required data loader and executor components to each configured site and
must be configured before client customization, export, or execution. The
mapping must be non-empty and define at least ``min_clients`` sites. Reserved
targets such as ``server`` and ``@ALL`` are not site names.

``configured_sites()`` returns the configured top-level site names. It does not
infer sites from recipe metadata, indicate which clients are connected, validate
production enrollment, or replace the execution environment.

.. important::

   Each site's dictionary is recipe-specific. FedAvg recipes support
   ``train_script``, ``train_args``, ``launch_external_process``, ``command``,
   ``framework``, ``server_expected_format``, ``params_transfer_type``,
   ``launch_once``, and ``shutdown_timeout``. ``FedEvalRecipe`` supports the
   corresponding ``eval_script`` and ``eval_args`` fields plus its launch,
   command, and exchange-format overrides. XGBoost recipes require a
   ``data_loader`` for every site; bagging also accepts ``lr_scale``.

   The older ``per_site_config=...`` constructor argument remains temporarily
   available for compatibility, emits ``FutureWarning``, and delegates to this
   helper behavior. New code should use ``set_per_site_config``.

No Secrets In Recipe Parameters
-------------------------------

Recipe parameters are job definition, not secret storage. Values such as
``train_args``, ``task_args``, ``eval_args``, ``per_site_config``, config
override dictionaries, execution parameters, and dictionaries passed to
``add_client_config`` or ``add_server_config`` can be serialized in clear text
into the generated job. They must never contain actual passwords, API keys,
tokens, private keys, or other credentials.

Recipes emit ``PotentialSecretWarning`` when a supplied value looks like an
actual secret, but this heuristic check cannot prove that a value is safe.
Keep the value at the executing site. Use ``secret_ref`` for a site environment
variable or ``secret_file_ref`` for a mounted secret file only at a supported
runtime boundary. See :ref:`recipe_secrets` for the supported locations,
examples, and deployment guidance.

Recipe Metadata
---------------

Use ``set_recipe_meta`` to add generated job metadata from a recipe without
mutating nested generated-job metadata directly. The helper sets one ``JobMetaKey``
metadata entry at a time:

.. code-block:: python

   from nvflare.apis.job_def import JobMetaKey
   from nvflare.recipe import set_recipe_meta

   set_recipe_meta(
       recipe,
       JobMetaKey.SCOPE,
       "private",
   )
   set_recipe_meta(
       recipe,
       JobMetaKey.RESOURCE_SPEC,
       {
           "site-1": {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 4},
           "site-2": {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 2},
       },
   )
   set_recipe_meta(
       recipe,
       JobMetaKey.JOB_LAUNCHER_SPEC,
       {
           "site-1": {"docker": {"image": "nvflare-site1:latest"}},
           "site-2": {"docker": {"image": "nvflare-site2:latest"}},
       },
   )

The settable keys are exactly the members of
:data:`nvflare.apis.job_def.USER_SETTABLE_JOB_META_KEYS`; other enum members
and raw strings are not accepted. Each key expects a specific value shape:

* ``JobMetaKey.RESOURCE_SPEC`` (``resource_spec``): per-site resource
  requirements -- a dict keyed by site name with dict values.
* ``JobMetaKey.JOB_LAUNCHER_SPEC`` (``launcher_spec``): per-site launcher
  requirements -- a dict keyed by site name with dict values.
* ``JobMetaKey.SCOPE`` (``scope``): job scope name -- a string.
* ``JobMetaKey.CUSTOM_PROPS`` (``custom_props``): nested custom metadata -- a
  dict.

Two groups of keys are intentionally **not** settable through this helper:

* Keys with dedicated ``FedJob`` constructor fields -- ``min_clients`` and
  ``mandatory_clients``. Set them when constructing the recipe/``FedJob``
  (e.g. ``FedJob(..., min_clients=2, mandatory_clients=[...])``) so the
  controller, scheduler, and generated metadata all use the same value;
  setting them through ``meta_props`` would only change the metadata and
  diverge from the value the recipe already used to build its controller.
* ``study``: the server assigns it from the admin session's active study at
  job submission, so a recipe-set value would be silently overwritten. Select
  the study through the execution environment instead (e.g.
  ``PocEnv(study=...)`` or ``ProdEnv(study=...)``, described below).

Dict values, including all nested dictionary and list contents, must be
JSON-serializable; dictionary keys are coerced to strings as they will appear
in ``meta.json``, and non-finite floating-point values such as ``NaN`` and
``Infinity`` are rejected. The helper writes the key/value pair through
``meta_props``. If the generated ``meta.json`` also contains that key, the
``meta_props`` value is written last by the job generator.

.. note::

   Per-site resource specs may also exist on the underlying generated job
   (registered through the lower-level job object's ``add_resource_spec``,
   an internal path -- prefer ``set_recipe_meta`` in recipe scripts; see
   :ref:`recipe_api`). If you set ``RESOURCE_SPEC`` through
   ``set_recipe_meta``, the ``meta_props`` value replaces those per-site
   specs in the generated ``meta.json``; a warning is emitted for specs
   already registered when the helper is called, but specs added afterwards
   are overridden without one.

If the same key already exists in ``meta_props``, ``set_recipe_meta`` replaces
that value.

The helper does not validate runtime resource availability, production
enrollment, or whether sites named in metadata are present for a run. The
execution environment and deployment still determine which sites are present.

For a complete production example, see the
:github_nvflare_link:`Recipe job on Kubernetes clients <examples/advanced/recipe-k8s>`.
It uses ``ProdEnv`` to submit a PyTorch CIFAR-10 job to ``site-1`` and
``site-2`` in separate Kubernetes clusters, keeps GPU requirements in
``resource_spec``, and places the per-cluster job images and container
settings in ``launcher_spec``.

Execution Environments
----------------------

A **Job Recipe** defines *what* to run in a federated learning setting, but it also needs to know *where* to run. NVFlare provides several **execution environments** that allow the same recipe to be executed in different contexts:

* **Simulation (** ``SimEnv`` **)** – For local testing and experimentation on a single machine or in one batch job
* **Proof-of-Concept (** ``PocEnv`` **)** – For small-scale, multi-process setups that mimic real-world deployment on a single machine
* **Production (** ``ProdEnv`` **)** – For full-scale distributed deployments across multiple organizations and sites

This separation enables users to **prototype once and deploy anywhere** without modifying the core job definition.

SimEnv – Simulation Environment
-------------------------------

Runs the job with the local FL simulator backend: no provisioned project or
long-running server/client daemons. Simulated clients use local worker
processes; ``num_threads`` is the historical name for the worker-process
concurrency. Best suited for:

* Quick experiments
* Debugging scripts and models
* Educational use cases
* Batch-scheduled experiments where one submitted job should run the complete
  federated workflow and then exit

**Arguments:**

* ``num_clients`` (int): Number of simulated clients
* ``clients``: A list of client names (length needs to match ``num_clients`` if both are provided)
* ``num_threads``: Number of concurrent simulated client worker processes
* ``gpu_config`` (str): List of GPU device IDs, comma separated
* ``log_config`` (str): Log config mode (``'concise'``, ``'full'``, ``'verbose'``), filepath, or level
* ``workspace_root`` (str): Root directory for simulation artifacts; defaults to ``/tmp/nvflare/simulation``

.. note::

   ``NVFLARE_SIMULATOR_WORKSPACE_ROOT`` is a process-level orchestration
   override. When it is set, ``SimEnv`` uses it instead of ``workspace_root``,
   including an explicitly supplied constructor value. Auto-FL uses this
   override only in each trial's child process to prevent concurrent simulator
   runs from sharing artifacts. ``SimEnv`` emits a ``RuntimeWarning`` when the
   override changes the configured path. Normal Recipe applications should
   leave it unset and configure ``workspace_root`` directly.

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
* ``docker_image`` (str, optional): SP/CP Docker image for Docker POC mode
  prepared with the deploy Docker preparation path. Jobs submitted in this mode
  must specify their SJ/CJ Docker image in ``launcher_spec``.
* ``project_conf_path`` (str, optional): Path to the project configuration file.
* ``study`` (str, optional): The study context for this execution environment. Jobs will be submitted and monitored within this study. Defaults to ``"default"``. Named studies require ``project_conf_path`` to point to a project with ``api_version: 4`` and ``studies:``. See :ref:`multi_study_guide`.

Let's first set the path to the POC environment:

.. code-block:: shell

   %env NVFLARE_POC_WORKSPACE=/tmp/nvflare/poc

.. code-block:: python

   from nvflare.recipe.poc_env import PocEnv

   # Create a POC environment
   env = PocEnv(
       num_clients=2
   )
   # Execute the recipe
   run = recipe.execute(env=env)
   run.get_status()
   run.get_result()

The result is stored under the directory ``/tmp/nvflare/poc``.

To use a named study, point ``PocEnv`` to a custom project file that defines ``studies:``:

.. code-block:: python

   env = PocEnv(
       num_clients=2,
       project_conf_path="/tmp/nvflare/poc_project.yml",
       study="cancer-research"  # omit for the default study
   )

If ``project_conf_path`` is not specified, or if the project does not define ``studies:``, the POC deployment behaves as single-tenant and only the ``default`` study is valid.

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
* ``study`` (str): The study context for this execution environment. Jobs will be submitted and monitored within this study. Defaults to ``"default"``. See :ref:`multi_study_guide`.

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
       model=SimpleNetwork(),
       train_script="client.py",
       train_args="--batch_size 32",
   )
   # Create a Prod environment
   env = ProdEnv(
       startup_kit_location="/tmp/nvflare/prod_workspaces/example_project/prod_00/admin@nvidia.com",
       study="cancer-research"  # omit for the default study
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
