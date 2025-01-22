**************************
What's New in FLARE v2.4.0
**************************

Usability Improvements
======================

Client API
----------
We introduce the new Client API, which streamlines the conversion process from centralized to federated deep learning code.
Using the Client API only requires a few lines of code changes, without the need to restructure the code or implement a new class.
Users can modify their pre-existing centralized deep learning code with these small changes to easily transform into federated learning code.
For PyTorch-Lightning, we provide a tight integration which requires even fewer lines of code changes.
Furthermore, the Client API significantly reduces the need for users to delve into FLARE specific concepts, helping to simplify the overall user experience.

Here is a brief example of a common pattern when using the Client API for a client trainer:

.. code-block:: python

    # import nvflare client API
    import nvflare.client as flare

    # initialize NVFlare client API
    flare.init()

    # run continuously when launching once
    while flare.is_running():

      # receive FLModel from NVFlare
      input_model = flare.receive()

      # loads model from NVFlare
      net.load_state_dict(input_model.params)

      # perform local training and evaluation on received model
      {existing centralized deep learning code} ...

      # construct output FLModel
      output_model = flare.FLModel(
          params=net.cpu().state_dict(),
          metrics={"accuracy": accuracy},
          meta={"NUM_STEPS_CURRENT_ROUND": steps},
      )

      # send model back to NVFlare
      flare.send(output_model)

For more in-depth information on the Client API, refer to the :ref:`client_api` documentation and :github_nvflare_link:`examples <examples/hello-world/ml-to-fl>`.

The 3rd-Party Integration Pattern
---------------------------------
In certain scenarios, users face challenges when attempting to move the training logic to the FLARE client side due to pre-existing ML/DL training system infrastructure.
In the 2.4.0 release, we introduce the Third-Party Integration Pattern, which allows the FLARE system and a third-party external training system to seamlessly exchange model parameters without requiring a tightly integrated system.

See the :ref:`3rd_party_integration` documentation for more details.


Job Templates and CLI
---------------------
The newly added Job Templates serve as pre-defined Job configurations designed to improve the process of creating and adjusting Job configurations.
Using the new Job CLI, users can easily leverage existing Job Templates, modify them according to their needs, and generate new ones.
Furthermore, the Job CLI also offers users a convenient method for submitting jobs directly from the command line, without the need for starting the Admin console.

``nvflare job list_templates|create|submit|show_variables``

Also explore the continuously growing :github_nvflare_link:`Job Template directory <job_templates>` we have created for commonly used configurations.
For more in-depth information on Job Templates and the Job CLI, refer to the :ref:`job_cli` documentation and :github_nvflare_link:`tutorials <examples/tutorials/job_cli.ipynb>`.

ModelLearner
------------
The ModelLearner is introduced for a simplified user experience in cases requiring a Learner-pattern.
Users exclusively interact with the FLModel object, which includes weights, optimizer, metrics, and metadata, while FLARE-specific concepts remain hidden to users.
The ModelLearner defines standard learning functions, such as ``train()``, ``validate()``, and ``submit_model()`` that can be subclassed for easy adaptation.

See the :ref:`model_learner` documentation and API definitions of :github_nvflare_link:`ModelLearner <nvflare/app_common/abstract/model_learner.py>` and
:github_nvflare_link:`FLModel <nvflare/app_common/abstract/fl_model.py>` for more detail.

Step-by-Step Example Series
---------------------------
To help users quickly get started with FLARE, we've introduced a comprehensive :github_nvflare_link:`step-by-step example series <examples/hello-world/step-by-step>` using Jupyter Notebooks.
Unlike traditional examples, each step-by-step example utilizes only two datasets for consistency— CIFAR10 for image data and the HIGGS dataset for tabular data.
Each example will build upon previous ones to showcase different features, workflows, or APIs, allowing users to gain a comprehensive understanding of FLARE functionalities.

**CIFAR10 Examples:**

- image_stats: federated statistics (histograms) of CIFAR10.
- sag: scatter and gather (SAG) workflow with PyTorch with Client API.
- sag_deploy_map: scatter and gather workflow with deploy_map configuration, for deployment of apps to different sites using the Client API.
- sag_model_learner: scatter and gather workflow illustrating how to write client code using the ModelLearner.
- sag_executor: scatter and gather workflow demonstrating show to write client-side executors.
- sag_mlflow: MLflow experiment tracking logs with the Client API in scatter & gather workflows.
- sag_he: homomorphic encryption using Client API and POC -he mode.
- cse: cross-site evaluation using the Client API.
- cyclic: cyclic weight transfer workflow with server-side controller.
- cyclic_ccwf: client-controlled cyclic weight transfer workflow with client-side controller.
- swarm: swarm learning and client-side cross-site evaluation with Client API.

**HIGGS Examples:**

- tabular_stats: federated statistics tabular histogram calculation.
- scikit_learn: federated linear model (logistic regression on binary classification) learning on tabular data.
- sklearn_svm: federated SVM model learning on tabular data.
- sklearn_kmeans: federated k-Means clustering on tabular data.
- xgboost: federated horizontal xgboost learning on tabular data with bagging collaboration.

Streaming APIs
==============
To support large language models (LLMs), the 2.4.0 release introduces the streaming API to facilitate the transfer of objects exceeding the 2 GB size limit imposed by gRPC.
The addition of a new streaming layer designed to handle large objects allows us to divide the large model into 1M chunks and stream them to the target.
We provide built-in streamers for Objects, Bytes, Files, and Blobs, providing a versatile solution for efficient object streaming between different endpoints.

Refer to the :mod:`nvflare.fuel.f3.stream_cell` api for more details, and the :ref:`notes_on_large_models` documentation for insights on working with large models in FLARE.

Expanding Federated Learning Workflows
======================================
In the 2.4.0 release, we introduce :ref:`client_controlled_workflows` as an alternative to the existing server-side controlled workflows.

Server-side controlled workflow
-------------------------------

- Server is trusted by all clients to handle the training process, job management as well as final model weights
- Server controller manages the job lifecycle (eg. health of client sites, monitoring of job status)
- Server controller manages the training process (eg. task assignment, model initialization, aggregation, and obtaining the distributed final model)

Client-side controlled workflow
-------------------------------

- Clients do not trust the server to handle the training process. Instead task assignment, model initialization, aggregation, and final model distribution are handled by clients.
- Server controller still manages the job lifecycle (eg. health of client sites, monitoring of job status)
- **Secure Messaging:** Peer-to-Peer clients exchange messages using TLS encryption where sender uses the public key of the receiver from certificates received, and encrypts messages with AES256 key.
  Only the sender and client can view the message. In the case that there is no direction connection between clients and the message is routed via the server, the server will be unable to decrypt the message.

Three commonly used types of client-side controlled workflows are provided:

- :ref:`ccwf_cyclic_learning`: the model is passed from client to client.
- :ref:`ccwf_swarm_learning`: randomly select clients as client-side controller and aggregators, where then Scatter and Gather with FedAvg is performed.
- :ref:`ccwf_cross_site_evaluation`: allow clients to evaluate other sites' models.

See :github_nvflare_link:`swarm learning <examples/advanced/swarm_learning>` and :github_nvflare_link:`client-controlled cyclic <examples/hello-world/step-by-step/cifar10/cyclic_ccwf>` for examples using these client-controlled workflows.

MLFlow and Weights & Biases Experiment Tracking Support
=======================================================
We expand our experiment tracking support with MLFLow and Weights & Biases systems.
The detailed documentation on these features can be found in :ref:`experiment_tracking`, and examples can be found at FL Experiment Tracking with
:github_nvflare_link:`MLFlow <examples/advanced/experiment-tracking/mlflow>` and
:github_nvflare_link:`wandb <examples/advanced/experiment-tracking/wandb>`.

Configuration Enhancements
==========================

Multi Configuration File Formats
--------------------------------
In the 2.4.0 release, we have added support for multiple configuration formats.
Prior to this release, the sole configuration file format was JSON, which although flexible, was lacking in useful features such as comments, variable substitution, and inheritance.

We added two new configuration formats:

- `Pyhocon <https://github.com/chimpler/pyhocon>`_ - a JSON variant and HOCON (Human-Optimized Config Object Notation) parser for Python, with many desired features
- `OmegaConf <https://omegaconf.readthedocs.io/en/2.3_branch/>`_ - a YAML based hierarchical configuration

Users have the flexibility to use a single format or combine several formats, as exemplified by config_fed_client.conf and config_fed_server.json.
If multiple configuration formats coexist, then their usage will be prioritized based on the following search order: .json -> .conf -> .yml -> .yaml

Improved Job Configuration File Processing
------------------------------------------
- Variable Resolution - for user-defined variable definitions and variable references in config files
- Built-in System Variables - for pre-defined system variables available to use in config files
- OS Environment Variables - OS environment variables can be referenced via the dollar sign
- Parameterized Variable Definition - for creating configuration templates that can be reused and resolved into different concrete configurations

See more details in the :ref:`configurations` documentation.

POC Command Upgrade
===================
We have expanded the POC command to bring users one step closer to the real deployment process.
The changes allow users to experiment with deployment options locally, and use the same project.yaml file for both experimentation and in production.

The POC command mode has been changed from "local, non-secure" to "local, secure, production" to better reflect the production environment simulation.
Lastly, the POC command is now more aligned with common syntax,
``nvflare poc -<action>`` => ``nvflare poc <action>``

See more details in the :ref:`poc_command` documentation or :github_nvflare_link:`tutorial <examples/tutorials/setup_poc.ipynb>`.

Security Enhancements
=====================

Unsafe component detection
--------------------------
Users now have the capability to define an unsafe component checker, and the checker will be invoked to validate the component to be built.
The checker raises UnsafeJob exception if it fails to validate the component, which will cause the job to be aborted.

For more details, refer to the :ref:`unsafe_component_detection` documentation.

Event-based security plug-in
----------------------------
We have introduced additional FL events that can be used to build plug-ins for job-level function authorizations.

For more details, refer to the :ref:`site_specific_auth` documentation as well as the
:github_nvflare_link:`custom authentication example <examples/advanced/custom_authentication>` for more details about these capabilites.

FL HUB: Hierarchical Unification Bridge
=======================================
The FL HUB is a new experimental feature designed to support multiple FLARE systems working together in a hierarchical manner.
In Federated Computing, the number of edge devices is usually large with often just a single server, which can cause performance issues.
A solution to this problem is to use a hierarchical FLARE system, where tiered FLARE systems connect together to form a tree-like structure.
Each leaf of clients (edge devices) only connect to its server, where this server also serves as the client for the parent tier FLARE system.

One potential use case is with global studies, where the client machine may be located across different regions.
Rather than requiring every region's client machines connect to only a single FL server in that region, the FL HUB could enable a more performant tiered multi-server setup.

Learn more about the FL Hub in the :ref:`hierarchy_unification_bridge` documentation and the :github_nvflare_link:`code <nvflare/app_common/hub>`.

Misc. Features
==============
- FLARE API Parity

  - FLARE API now has the same set of APIs as the Admin Client.
  - Allows users to use almost all of the commands from python API or notebooks.

- Docker Support

  - NVFLARE cloud CSP startup scripts now support deployment with docker containers in addition to VM deployment.
  - provision command now supports detached docker run, in addition to the interactive docker run.

- Flare Dashboard

  - Prior to the 2.4.0, the Flare dashboard can only run within a docker container.
  - In the 2.4.0, the Flare dashboard can now run locally without docker for development.

- Run Model Evaluation Without Training

  - In the 2.4.0 release, users can now run cross-validation without having to re-run the training.
  - See the example for :github_nvflare_link:`run cross-site validation without training <examples/hello-world/hello-numpy-cross-val#run-cross-site-validation-using-the-previous-trained-results>`.

- Communication Enhancements

  - We added the application layer ping between Client Job process and Server parent process to replace the gRPC timeout.
    Previously, we noticed if the gRPC timeout is set too long, the cloud provider (eg. Azure Cloud) will kill the connection after 4 minutes.
    If the timeout setup is too short (such as 2 minutes), the underlying gRPC will report too many pings.
    The application level ping will avoid both issues to make sure the server/client is aware of the status of the processes.
  - FLARE provides two drivers for gRPC based communication- asyncio (AIO) and regular (non-AIO) versions of gRPC library.
    One notable benefit of the AIO gRPC is its ability to handle many more concurrent connections on the server side.
    However, the AIO gRPC may crash under challenging network conditions on the client side, whereas the non-AIO gRPC is more stable.
    Hence in FLARE 2.4.0, the default configuration uses the non-AIO gRPC library version for better stability.

    - In order to change the driver selection, users can update ``comm_config.json`` in the local directory of the workspace,
      and set the ``use_aio_grpc`` config variable.

New Examples
============

Federated Large Language Model (LLM) examples
---------------------------------------------

We've added several examples to demonstrate how to work with federated LLM:

- :github_nvflare_link:`Parameter Efficient Fine Turning <integration/nemo/examples/peft>` utilizing NeMo's PEFT methods to adapt a LLM to a downstream task.
- :github_nvflare_link:`Prompt-Tuning Example <integration/nemo/examples/prompt_learning>` for using FLARE with NeMo for prompt learning.
- :github_nvflare_link:`Supervised Fine Tuning (SFT) <integration/nemo/examples/supervised_fine_tuning>` to fine-tune all parameters of a LLM on supervised data.
- :github_nvflare_link:`LLM Tuning via HuggingFace SFT Trainer <examples/advanced/llm_hf>` for using FLARE with a HuggingFace trainer for LLM tuning tasks.

Vertical Federated XGBoost
--------------------------
With the 2.0 release of `XGBoost <https://github.com/dmlc/xgboost>`_, we are able to demonstrate the :github_nvflare_link:`vertical xgboost example <examples/advanced/vertical_xgboost>`.
We use Private Set Intersection and XGBoost's new federated learning support to perform classification on vertically split HIGGS data (where sites share overlapping data samples but contain different features).

Graph Neural Networks (GNNs)
----------------------------
We added two examples using GraphSage to demonstrate how to train :github_nvflare_link:`Federated GNN on Graph Dataset using Inductive Learning <examples/advanced/gnn#federated-gnn-on-graph-dataset-using-inductive-learning>`.

**Protein Classification:** to classify protein roles based on their cellular functions from gene ontology.
The dataset we are using is PPI (`protein-protein interaction <http://snap.stanford.edu/graphsage/#code>`_) graphs, where each graph represents a specific human tissue.
Protein-protein interaction (PPI) dataset is commonly used in graph-based machine-learning tasks, especially in the field of bioinformatics.
This dataset represents interactions between proteins as graphs, where nodes represent proteins and edges represent interactions between them.

**Financial Transaction Classification:** to classify whether a given transaction is licit or illicit.
For this financial application, we use the `Elliptic++ <https://github.com/git-disl/EllipticPlusPlus>`_ dataset which
consists of 203k Bitcoin transactions and 822k wallet addresses to enable both the detection of fraudulent transactions and the detection of illicit
addresses (actors) in the Bitcoin network by leveraging graph data. For more details, please refer to this `paper <https://arxiv.org/pdf/2306.06108.pdf>`_.

Financial Application Examples
------------------------------
To demonstrate how to perform Fraud Detection in financial applications, we introduced an :github_nvflare_link:`example <examples/advanced/finance>` illustrating how to use XGBoost in various ways
to train a model in a federated manner with a `finance dataset <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>`_.
We illustrate both vertical and horizontal federated learning with XGBoost, along with histogram and tree-based approaches.

KeyCloak Site Authentication Integration
----------------------------------------
FLARE is agnostic to the 3rd party authentication mechanism, and each client can have its own authentication system.
We demonstrate FLARE's support of site-specific authentication using KeyCloak.
The :github_nvflare_link:`KeyCloak Site Authentication Integration <examples/advanced/keycloak-site-authentication>` example is configured so the admin user will need additional user authentication to submit and run a job.


**********************************
Migration to 2.4.0: Notes and Tips
**********************************

FLARE 2.4.0 introduces a few API and behavior changes. This migration guide will help you to migrate from the previous NVFLARE version to the current version.

Job Format: meta.json
=====================
In FLARE 2.4.0, users must have a meta.json configuration file defined in their jobs.
Legacy app definitions should be updated to the job format to include a meta.json file with a deployment map and any number of app folders (containing config/ and custom/).
Here is a basic job structure with a single app:

.. code-block:: shell

  ├── my_job
  │   ├── app
  │   │   ├── config
  │   │   │   ├── config_client.json
  │   │   │   └── config_server.json
  │   │   └── custom
  │   └── meta.json

Here is the default meta.json which can be edited accordingly:

.. code-block:: json

  {
    "name": "my_job",
    "resource_spec": {},
    "min_clients" : 2,
    "deploy_map": {
      "app": [
        "@ALL"
      ]
    }
  }

FLARE API Parity
================
In FLARE 2.3.0, an initial version of the FLARE API was implemented as a redesigned FLAdminAPI, however we only included a subset of the functions.
In FLARE 2.4.0, the FLARE API has been enhanced to include the remaining functions of the FLAdminAPI, so that the FLAdminAPI can sunset.

See the :ref:`migrating_to_flare_api` for more details on the added functions.

Timeout Handling
----------------

In the 2.4.0 release, improvements have been to made to the timeout handling for commands involving Admin Server communication with FL Clients and awaiting responses.
Previously, a fixed global timeout value was used on the Admin Server, however this value was sometimes not enough if a command took a long time
(e.g. ``cat server log.txt`` command may take time to transfer the large log file).
In this case, the user could use the ``set_timeout`` command to change the default timeout value of the Admin Server, however this command had the drawback of being global, and would affect all users.
The global effect of this command meant one user setting a very small timeout value could cause all user commands to fail.

To address this, the ``set_timeout`` command has been changed to be session specific.
Additionally a new ``unset_timeout`` command has been added to revert to use the Admin Server's default timeout for the session.

Changes to ``show_stats`` and ``show_errors``
---------------------------------------------

The old structure puts the server's result dict directly at the top level of the overall result dict, while each client's result dict is placed as an item keyed on the client name.
To make it consistent between server and client results, we've change to put the server's result as an item keyed on "server".
If any code is based on the old return structure of FLAdminAPI, please update it accordingly.

.. code-block:: json

    {
      "server": { # new "server" key for server result dict
        "ScatterAndGather": {
          "tasks": {
            "train": [
              "site-1",
              "site-2"
            ]
          },
          "phase": "train",
          "current_round": 2,
          "num_rounds": 50
        },
        "ServerRunner": {
          "job_id": "3ad5bdef-db12-4ffb-9362-0ff163973f7d",
          "status": "started",
          "workflow": "scatter_and_gather"
        }
      },
      "site-1": {
        "ClientRunner": {
          "job_id": "3ad5bdef-db12-4ffb-9362-0ff163973f7d",
          "current_task_name": "None",
          "status": "started"
        }
      },
      "site-2": {
        "ClientRunner": {
          "job_id": "3ad5bdef-db12-4ffb-9362-0ff163973f7d",
          "current_task_name": "train",
          "status": "started"
        }
      }
    }

POC Command Upgrade
===================
The POC command has been upgraded in 2.4.0:

- Remove ``--`` for action commands, change to subcommands
- POC is now using "production mode", the admin user name is now "admin@nvidia.com" instead of "admin" from previous releases.
- new ``-d`` docker and ``-he`` Homomorphic encryption options
- ``nvflare poc prepare`` generates ``.nvflare/config.conf`` to store location of POC workspace, takes precedent over environment variable ``NVFLARE_POC_WORKSPACE``
- In the previous version, the startup kits are located directly under default POC workspace at ``/tmp/nvflare/poc``. In the 2.4.0, the startup kit is now under ``/tmp/nvflare/poc/example_project/prod_00/`` to follow the production provision default structure.
- Multi-org and multi-role support

.. code-block:: none

  nvflare poc -h
  usage: nvflare poc [-h] [--prepare] [--start] [--stop] [--clean] {prepare,prepare-jobs-dir,start,stop,clean} ...

  optional arguments:
    -h, --help            show this help message and exit
    --prepare             deprecated, suggest use 'nvflare poc prepare'
    --start               deprecated, suggest use 'nvflare poc start'
    --stop                deprecated, suggest use 'nvflare poc stop'
    --clean               deprecated, suggest use 'nvflare poc clean'

  poc:
    {prepare,prepare-jobs-dir,start,stop,clean}
                          poc subcommand
      prepare             prepare poc environment by provisioning local project
      prepare-jobs-dir    prepare jobs directory
      start               start services in poc mode
      stop                stop services in poc mode
      clean               clean up poc workspace

Refer to :ref:`poc_command` for more details.

Secure Messaging
================

A new ``secure`` argument has been added for ``send_aux_request()`` in :class:`ServerEngineSpec<nvflare.apis.server_engine_spec.ServerEngineSpec>`,
and :class:`ClientEngineExecutorSpec<nvflare.private.fed.client.client_engine_executor_spec.ClientEngineExecutorSpec>`.

``secure`` is an optional boolean to determine whether the aux request should be sent in a secure way.
One such use case is for secure peer-to-peer messaging, such as in the client-controlled workflows.

.. code-block:: python

   @abstractmethod
    def send_aux_request(
        self,
        targets: Union[None, str, List[str]],
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        optional=False,
        secure: bool = False,
    ) -> dict:
        """Send a request to Server via the aux channel.
        Implementation: simply calls the ClientAuxRunner's send_aux_request method.
        Args:
            targets: aux messages targets. None or empty list means the server.
            topic: topic of the request
            request: request to be sent
            timeout: number of secs to wait for replies. 0 means fire-and-forget.
            fl_ctx: FL context
            optional: whether the request is optional
            secure: should the request sent in the secure way
        Returns:
            a dict of reply Shareable in the format of:
                { site_name: reply_shareable }
        """
        pass

Stats Result Format
===================
In :class:`StatisticsController<nvflare.app_common.workflows.statistics_controller.StatisticsController>`,
the result dictionary format originally concatenated "site" and "dataset" to support visualization.
In 2.4.0 this has now been changed so "site" and "dataset" have their own keys in the result dictionary.

``result = {feature: {statistic: {site-dataset: value}}}``

to

``result =  feature: {statistic: {site: {dataset: value}}}}``

To continue to support the visualization needs, the site-dataset concatenation logic has instead been moved to
:class:`Visualization<nvflare.app_opt.statistics.visualization.statistics_visualization.Visualization>`.
