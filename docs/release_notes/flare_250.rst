**************************
What's New in FLARE v2.5.0
**************************

User Experience Improvements
============================
NVFlare 2.5.0 offers several new sets of APIs that allows for end-to-end ease of use that can greatly improve researcher and data
scientists' experience working with FLARE. The new API covers client, server and job construction with end-to-end pythonic user experience.

Model Controller API
~~~~~~~~~~~~~~~~~~~~
The new :ref:`model_controller` greatly simplifies the experience of developing new federated learning workflows. Users can simply subclass
the ModelController to develop new workflows. The new API doesn't require users to know the details of NVFlare constructs except for FLModel
class, where it is simply a data structure that contains model weights, optimization parameters and metadata. 

You can easily construct a new workflow with basic python code, and when ready, the send_and_wait() communication function is all you need for
communication between clients and server. 

Client API
~~~~~~~~~~
We introduced another :ref:`client_api` implementation,
:class:`InProcessClientAPIExecutor<nvflare.app_common.executors.in_process_client_api_executor.InProcessClientAPIExecutor>`.
This has the same interface and syntax of the previous Client API using
:class:`SubprocessLauncher<nvflare.app_common.launchers.subprocess_launcher.SubprocessLauncher>`, except all communication is in memory. 

Using this in-process client API, we build a :class:`ScriptExecutor<nvflare.app_common.executors.script_executor.ScriptExecutor>`,
which is directly used in the new Job API.

Compared with SubProcessLauncherClientAPI, the in-process client API offers better efficiency and is easier to configure. All
the operations will be carried out within the memory space of the executor.  

SubProcessLauncherClientAPI can be used for cases where a separate training process is required.

Job API
~~~~~~~
The new Job API, or :ref:`fed_job_api`, combined with Client API and Model Controller API, will give users an end-to-end pythonic
user experience. The Job configuration, required prior to the current release, can now be directly generated automatically, so the
user doesn't need to edit the configuration files manually. 

We provide many examples to demonstrate the power of the new Job APIs making it very easy to experiment with new federated
learning algorithms or create new applications. 

Flower Integration
------------------
Integration between NVFlare and the `Flower <https://flower.ai/>`_ framework aims to provide researchers the ability to leverage
the strengths of both frameworks by enabling Flower projects to seamlessly run on top of NVFlare. Through the seamless
integration of Flower and FLARE, applications crafted within the Flower framework can effortlessly operate within the FLARE runtime
environment without necessitating any modifications. This initial integration streamlines the process, eliminating complexities and
ensuring smooth interoperability between the two platforms, thus enhancing the overall efficiency and accessibility of FL applications.
Please find details `here <https://arxiv.org/abs/2407.00031>`__. A hello-world example is available
:github_nvflare_link:`here <examples/hello-world/hello-flower>`.

Secure XGBoost
--------------
The latest features from XGBoost introduced the support for secure federated learning via homomorphic encryption. For vertical federated
XGBoost learning, the gradients of each sample are protected by encryption such that the label information
will not be leaked to unintended parties; while for horizontal federated XGBoost learning, the local gradient histograms will not be
learnt by the central aggregation server. 

With our encryption plugins working with XGBoost, NVFlare now supports all secure federated schemes for XGBoost model training, with
both CPU and GPU.

Please check `federated xgboost with nvflare user guide <https://nvflare.readthedocs.io/en/2.5/user_guide/federated_xgboost.html>`
and the :github_nvflare_link:`example <examples/advanced/xgboost_secure>`

Tensorflow support
------------------
With community contributions, we add FedOpt, FedProx and Scaffold algorithms using Tensorflow.
You can check the code :github_nvflare_link:`here <nvflare/app_opt/tf>` and the :github_nvflare_link:`example <examples/getting_started/tf>`

FOBS Auto Registration
----------------------
FOBS, the secure mechanism NVFlare uses for message serialization and deserialization, is enhanced with new auto registration features.
These changes will reduce the number of decomposers that users have to register. The changes are:

  - Auto registering of decomposers on deserialization. The decomposer class is stored in the serialized data and the decomposers are
    registered automatically when deserializing. If a component only receives serialized data but it doesn't perform serialization,
    decomposer registering is not needed anymore.

  - Data Class decomposer auto registering on serialization. If a decomposer is not found for a class, FOBS will try to treat the class
    as a Data Class and register DataClassDecomposer for it. This works in most cases but not all.


New Examples
------------
Secure Federated Kaplan-Meier Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :github_nvflare_link:`Secure Federated Kaplan-Meier Analysis via Time-Binning and Homomorphic Encryption example <examples/advanced/kaplan-meier-he>`
illustrates two features:

  - How to perform Kaplan-Meier survival analysis in a federated setting without and with secure features via time-binning and Homomorphic Encryption (HE).
  - How to use the Flare ModelController API to contract a workflow to facilitate HE under simulator mode.

BioNemo example for Drug Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`BioNeMo <https://www.nvidia.com/en-us/clara/bionemo/>`_ is NVIDIA's generative AI platform for drug discovery.
We included several examples of running BioNeMo in a federated learning environment using NVFlare:

  - The :github_nvflare_link:`task fitting example <examples/advanced/bionemo/task_fitting/README.md>` includes a notebook that shows how to obtain protein-learned representations in the form of embeddings using the ESM-1nv pre-trained model.
  - The :github_nvflare_link:`downstream example <examples/advanced/bionemo/downstream/README.md>` shows three different downstream tasks for fine-tuning a BioNeMo ESM-style model.

Federated Logistic Regression with NR optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :github_nvflare_link:`Federated Logistic Regression with Second-Order Newton-Raphson optimization example <examples/advanced/lr-newton-raphson>`
shows how to implement a federated binary classification via logistic regression with second-order Newton-Raphson optimization.

Hierarchical Federated Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:github_nvflare_link:`Hierarchical Federated Statistics <examples/advanced/federated-statistics/hierarchical_stats>` is helpful when there
are multiple organizations involved.  For example, in the medical device applications, the medical devices usage statistics can be
viewed from both device, device-hosting site, and hospital or manufacturers' point of views.
Manufacturers would like to see the usage stats of their product (device) in different sites and hospitals. Hospitals
may like to see overall stats of devices including different products from different manufacturers. In such a case, the hierarchical
federated stats will be very helpful.

FedAvg Early Stopping Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :github_nvflare_link:`FedAvg Early Stopping example <examples/hello-world/hello-fedavg>` tries to demonstrate that with the new server-side model
controller API, it is very easy to change the control conditions and adjust workflows with a few lines of python code.

Tensorflow Algorithms & Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FedOpt, FedProx, Scaffold implementation for Tensorflow.

FedBN: Federated Learning on Non-IID Features via Local Batch Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :github_nvflare_link:`FedBN example <research/fed-bn>` showcases a federated learning algorithm designed
to address the feature shift problem when aggregating models across different data distributions.

In this work, we propose an effective method that uses local batch normalization to alleviate the feature shift before averaging models.
The resulting scheme, called FedBN, outperforms both classical FedAvg and FedProx in our extensive experiments. These empirical results
are supported by a convergence analysis that shows in a simplified setting that FedBN has a faster convergence rate than FedAvg.


End-to-end Federated XGBoost examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In :github_nvflare_link:`this example <examples/advanced/finance-end-to-end/xgboost.ipynb>`,
we try to show that end-to-end process of feature engineering, pre-processing and training in federated settings. You
can use FLARE to perform federated ETL and then training. 

Developer Tutorial Page
-----------------------
To let users quickly learn Federated Learning with FLARE, we developed a `tutorial web page <https://nvidia.github.io/NVFlare>`_ with
both code and video to interactively learn how to convert and run FL in a few minutes. We also
created a tutorial catalog to help you easily search and find the examples you are interested in.

**********************************
Migration to 2.5.0: Notes and Tips
**********************************

FLARE 2.5.0 introduces some API and behavior changes. This migration guide will help you to migrate from the previous NVFlare version
to the current version.

Deprecate "name" to only use "path"
-----------------------------------
In 2.5.0, the "name" field in configurations is deprecated. You need to change the "name" field to "path" and use the full path. For
example,

.. code-block:: json

  "name": "TBAnalyticsReceiver"

needs to be updated to:

.. code-block:: json

  "path": "nvflare.app_opt.tracking.tb.tb_receiver.TBAnalyticsReceiver"

XGBoost v1 - v2
---------------

XGBoost support is enhanced in 2.5.0 to support secure training using Homomorphic Encryption (HE). The user interface is also simplified by
setting the XGBoost parameters in the controller so all clients get the same parameters. 

The main changes are:

  - The xgboost params have been moved from the client configuration to server.
  - New split_mode and secure_training parameters
  - New :class:`CSVDataLoader<nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader.CSVDataLoader>`

Sample configuration files for 2.5.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

config_fed_server.json
""""""""""""""""""""""

.. code-block:: json

  {
      "format_version": 2,
      "num_rounds": 3,
      "workflows": [
          {
              "id": "xgb_controller",
              "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController",
              "args": {
                  "num_rounds": "{num_rounds}",
                  "split_mode": 1,
                  "secure_training": false,
                  "xgb_options": {
                      "early_stopping_rounds": 2
                  },
                  "xgb_params": {
                      "max_depth": 3,
                      "eta": 0.1,
                      "objective": "binary:logistic",
                      "eval_metric": "auc",
                      "tree_method": "hist",
                      "nthread": 1
                  },
                  "client_ranks": {
                      "site-1": 0,
                      "site-2": 1
                  },
                  "in_process": true 
              }
          }
      ]
  }

config_fed_client.json
""""""""""""""""""""""

.. code-block:: json

  {
      "format_version": 2,
      "executors": [
          {
              "tasks": [
                  "config",
                  "start"
              ],
              "executor": {
                  "id": "Executor",
                  "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_executor.FedXGBHistogramExecutor",
                  "args": {
                      "data_loader_id": "dataloader",
                      "in_process": true
                  }
              }
          }
      ],
      "components": [
          {
              "id": "dataloader",
              "path": "nvflare.app_opt.xgboost.histogram_based_v2.secure_data_loader.SecureDataLoader",
              "args": {
                  "rank": 0,
                  "folder": "/tmp/nvflare/dataset/vertical_xgb_data"
              }
          }
      ]
  }

Simulator workspace structure
-----------------------------

In 2.4.0, the server and all the clients shared the same simulator workspace root of ``simulate_job``. The server and each client had
their own app_XXXX job definition, but the same root folder for the workspace may result in conflicting model file locations.

.. raw:: html

   <details>
   <summary><a>Example folder structure for 2.4.0</a></summary>

.. code-block:: none

  simulator/
  ├── local
  │   └── log.config
  ├── simulate_job
  │   ├── app_server
  │   │   ├── FL_global_model.pt
  │   │   ├── __init__.py
  │   │   ├── config
  │   │   │   ├── config_fed_client.json
  │   │   │   ├── config_fed_server.json
  │   │   │   ├── config_train.json
  │   │   │   ├── config_validation.json
  │   │   │   ├── dataset_0.json
  │   │   │   └── environment.json
  │   │   ├── custom
  │   │   │   ├── __init__.py
  │   │   │   ├── add_shareable_parameter.py
  │   │   │   ├── client_aux_handler.py
  │   │   │   ├── client_send_aux.py
  │   │   │   ├── client_trainer.py
  │   │   │   ├── fed_avg_responder.py
  │   │   │   ├── model_shareable_manager.py
  │   │   │   ├── print_shareable_parameter.py
  │   │   │   ├── server_aux_handler.py
  │   │   │   ├── server_send_aux.py
  │   │   │   └── supervised_fitter.py
  │   │   ├── docs
  │   │   │   ├── Readme.md
  │   │   │   └── license.txt
  │   │   ├── eval
  │   │   └── models
  │   ├── app_site-1
  │   │   ├── __init__.py
  │   │   ├── config
  │   │   │   ├── config_fed_client.json
  │   │   │   ├── config_fed_server.json
  │   │   │   ├── config_train.json
  │   │   │   ├── config_validation.json
  │   │   │   ├── dataset_0.json
  │   │   │   └── environment.json
  │   │   ├── custom
  │   │   │   ├── __init__.py
  │   │   │   ├── add_shareable_parameter.py
  │   │   │   ├── client_aux_handler.py
  │   │   │   ├── client_send_aux.py
  │   │   │   ├── client_trainer.py
  │   │   │   ├── fed_avg_responder.py
  │   │   │   ├── model_shareable_manager.py
  │   │   │   ├── print_shareable_parameter.py
  │   │   │   ├── server_aux_handler.py
  │   │   │   ├── server_send_aux.py
  │   │   │   └── supervised_fitter.py
  │   │   ├── docs
  │   │   │   ├── Readme.md
  │   │   │   └── license.txt
  │   │   ├── eval
  │   │   ├── log.txt
  │   │   └── models
  │   ├── app_site-2
  │   │   ├── __init__.py
  │   │   ├── config
  │   │   │   ├── config_fed_client.json
  │   │   │   ├── config_fed_server.json
  │   │   │   ├── config_train.json
  │   │   │   ├── config_validation.json
  │   │   │   ├── dataset_0.json
  │   │   │   └── environment.json
  │   │   ├── custom
  │   │   │   ├── __init__.py
  │   │   │   ├── add_shareable_parameter.py
  │   │   │   ├── client_aux_handler.py
  │   │   │   ├── client_send_aux.py
  │   │   │   ├── client_trainer.py
  │   │   │   ├── fed_avg_responder.py
  │   │   │   ├── model_shareable_manager.py
  │   │   │   ├── print_shareable_parameter.py
  │   │   │   ├── server_aux_handler.py
  │   │   │   ├── server_send_aux.py
  │   │   │   └── supervised_fitter.py
  │   │   ├── docs
  │   │   │   ├── Readme.md
  │   │   │   └── license.txt
  │   │   ├── eval
  │   │   ├── log.txt
  │   │   └── models
  │   ├── log.txt
  │   ├── meta.json
  │   └── pool_stats
  │       └── simulator_cell_stats.json
  └── startup
      ├── client_context.tenseal
      └── server_context.tenseal

.. raw:: html

   </details>
   <br />

In 2.5.0, the server and all the clients will have their own workspace subfolder under the simulator workspace. The ``simulator_job``
is within the workspace of each site. This results in the total isolation of each site, with no model files conflicting. This workspace
structure is consistent with the format of the POC real world application.

.. raw:: html

   <details>
   <summary><a>Example folder structure for 2.5.0</a></summary>

.. code-block:: none

  simulator/
  ├── server
  │   ├── local
  │   │   └── log.config
  │   ├── log.txt
  │   ├── pool_stats
  │   │   └── simulator_cell_stats.json
  │   ├── simulate_job
  │   │   ├── app_server
  │   │   │   ├── FL_global_model.pt
  │   │   │   └── config
  │   │   │       ├── config_fed_client.conf
  │   │   │       └── config_fed_server.conf
  │   │   ├── artifacts
  │   │   │   ├── 39d0b7edb17b437dbf77da2e402b2a4d
  │   │   │   │   └── artifacts
  │   │   │   │       └── running_loss_reset.txt
  │   │   │   └── b10ff3e54b0d464c8aab8cf0b751f3cf
  │   │   │       └── artifacts
  │   │   │           └── running_loss_reset.txt
  │   │   ├── cross_site_val
  │   │   │   ├── cross_val_results.json
  │   │   │   ├── model_shareables
  │   │   │   │   ├── SRV_FL_global_model.pt
  │   │   │   │   ├── site-1
  │   │   │   │   └── site-2
  │   │   │   └── result_shareables
  │   │   │       ├── site-1_SRV_FL_global_model.pt
  │   │   │       ├── site-1_site-1
  │   │   │       ├── site-1_site-2
  │   │   │       ├── site-2_SRV_FL_global_model.pt
  │   │   │       ├── site-2_site-1
  │   │   │       └── site-2_site-2
  │   │   ├── meta.json
  │   │   ├── mlruns
  │   │   │   ├── 0
  │   │   │   │   └── meta.yaml
  │   │   │   └── 470289463842501388
  │   │   │       ├── 39d0b7edb17b437dbf77da2e402b2a4d
  │   │   │       │   ├── artifacts
  │   │   │       │   ├── meta.yaml
  │   │   │       │   ├── metrics
  │   │   │       │   │   ├── running_loss
  │   │   │       │   │   ├── train_loss
  │   │   │       │   │   └── validation_accuracy
  │   │   │       │   ├── params
  │   │   │       │   │   ├── learning_rate
  │   │   │       │   │   ├── loss
  │   │   │       │   │   └── momentum
  │   │   │       │   └── tags
  │   │   │       │       ├── client
  │   │   │       │       ├── job_id
  │   │   │       │       ├── mlflow.note.content
  │   │   │       │       ├── mlflow.runName
  │   │   │       │       └── run_name
  │   │   │       ├── b10ff3e54b0d464c8aab8cf0b751f3cf
  │   │   │       │   ├── artifacts
  │   │   │       │   ├── meta.yaml
  │   │   │       │   ├── metrics
  │   │   │       │   │   ├── running_loss
  │   │   │       │   │   ├── train_loss
  │   │   │       │   │   └── validation_accuracy
  │   │   │       │   ├── params
  │   │   │       │   │   ├── learning_rate
  │   │   │       │   │   ├── loss
  │   │   │       │   │   └── momentum
  │   │   │       │   └── tags
  │   │   │       │       ├── client
  │   │   │       │       ├── job_id
  │   │   │       │       ├── mlflow.note.content
  │   │   │       │       ├── mlflow.runName
  │   │   │       │       └── run_name
  │   │   │       ├── meta.yaml
  │   │   │       └── tags
  │   │   │           └── mlflow.note.content
  │   │   └── tb_events
  │   │       ├── site-1
  │   │       │   ├── events.out.tfevents.1724447288.yuhongw-mlt.86138.3
  │   │       │   ├── metrics_running_loss
  │   │       │   │   └── events.out.tfevents.1724447288.yuhongw-mlt.86138.5
  │   │       │   └── metrics_train_loss
  │   │       │       └── events.out.tfevents.1724447288.yuhongw-mlt.86138.4
  │   │       └── site-2
  │   │           ├── events.out.tfevents.1724447288.yuhongw-mlt.86138.0
  │   │           ├── metrics_running_loss
  │   │           │   └── events.out.tfevents.1724447288.yuhongw-mlt.86138.2
  │   │           └── metrics_train_loss
  │   │               └── events.out.tfevents.1724447288.yuhongw-mlt.86138.1
  │   └── startup
  ├── site-1
  │   ├── local
  │   │   └── log.config
  │   ├── log.txt
  │   ├── simulate_job
  │   │   ├── app_site-1
  │   │   │   └── config
  │   │   │       ├── config_fed_client.conf
  │   │   │       └── config_fed_server.conf
  │   │   ├── meta.json
  │   │   └── models
  │   │       └── local_model.pt
  │   └── startup
  ├── site-2
  │   ├── local
  │   │   └── log.config
  │   ├── log.txt
  │   ├── simulate_job
  │   │   ├── app_site-2
  │   │   │   └── config
  │   │   │       ├── config_fed_client.conf
  │   │   │       └── config_fed_server.conf
  │   │   ├── meta.json
  │   │   └── models
  │   │       └── local_model.pt
  │   └── startup
  └── startup

.. raw:: html

   </details>
   <br />

Allow Simulator local resources configuration
----------------------------------------------
In 2.4.0, we only support the ``log.config`` setting file within the simulator workspace ``startup`` folder to be used to change the log format.

In 2.5.0, we enable the full ``local`` and ``startup`` contents to be configured under the simulator workspace. All the POC real world application
local settings can be placed within the ``workspace/local`` folder and be deployed to each site. The ``log.config`` file is also moved to
this ``workspace/local`` folder.
