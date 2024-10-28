**************************
What's New in FLARE v2.3.0
**************************

Cloud Deployment Support
========================
The Dashboard UI and FL entities have expanded support for :ref:`cloud_deployment` for both Azure and AWS.
Simple CLI commands now exist to create the infrastructure, deploy, and start the Dashboard UI,
FL Server, and FL Client(s):

.. code-block:: bash

    nvflare dashboard --cloud azure | aws
    <server-startup-kit>/start.sh --cloud azure | aws
    <client-startup-kit>/start.sh --cloud azure | aws

These start scripts can automatically create the needed resources, VMs, networking, security groups, and deploy FLARE
to the newly created infrastructure and start the FLARE system.

Python Version Support
======================
FLARE is now supported for Python 3.9 and Python 3.10, so FLARE 2.3.0 will support Python versions 3.8, 3.9, 3.10.
Python 3.7 is no longer actively supported and tested.

New FLARE API to provide better user experience 
===============================================
The new FLARE API is an improved version of the FLAdminAPI with better ease of use. FLARE API currently supports selected commands. See
:ref:`migrating_to_flare_api` for details on migrating to the new FLARE API. For now, the FLAdminAPI should still remain functional.
For details on the FLARE API, you can see this notebook: https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/flare_api.ipynb.

Job Signing for Improved Security
=================================
Before a job is submitted to the server, the submitter's private key is used to sign each file's digest to ensure that custom code is signed.
Each folder has one signature file, which maps file names to the signatures of all files inside that folder. The signer's certificate is also
included for signature verification. The verification is performed at deployment time, rather than submission time, as the clients do not receive
the job until the job is deployed.

Client-Side Model Initialization
================================
Prior to FLARE 2.3.0, model initialization was performed on the server-side.
The model was either initialized from a model file or custom model initiation code. Pre-defining a model file required extra steps of pre-generating
and saving the model file and then sending it over to the server. Running custom model initialization code on server could be a security risk.

FLARE 2.3.0 introduces another way to initialize the model on the client side. The FL Server can select
the initial model based on a user-chosen strategy. Here is an example using client-side model initialization: https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-pt.
You can read more about this feature in :ref:`initialize_global_weights_workflow`.

Traditional Machine Learning Examples
=====================================
Several new examples have been added to support using traditional machine learning algorithms in federated learning:
   - :github_nvflare_link:`Linear model <examples/advanced/sklearn-linear>` using scikit-learn library via
     `iterative SGD training <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_.
     Linear and logistic regressions can be implemented following this iterative example by adopting different loss functions.
   - :github_nvflare_link:`SVM <examples/advanced/sklearn-svm>` using scikit-learn library. In this two-step process, the server performs an additional round of SVM over the collected supporting vectors from clients.
   - :github_nvflare_link:`K-Means <examples/advanced/sklearn-kmeans>` using scikit-learn library via
     `mini-batch K-Means method <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html>`_.
     In this iterative process, each client performs mini-batch K-Means and the server syncs the updates for the global model.
   - :github_nvflare_link:`Random Forest <examples/advanced/random_forest>` using XGBoost library with
     `random forest functionality <https://xgboost.readthedocs.io/en/stable/tutorials/rf.html>`_. In this two-step process, clients
     construct sub-forests on their local data, and the server ensembles all collected sub-forests to produce the global random forest.

Vertical Learning
=================

Federated Private Set Intersection (PSI)
----------------------------------------
In order to support vertical learning use cases such as secure user-id matching and feature
over-lapping discovery, we have developed a multi-party private set intersection (PSI) operator
that allows for the secure discovery of data intersections. Our approach leverages OpenMined's two-party
`Private Set Intersection Cardinality protocol <https://github.com/OpenMined/PSI>`_, which is based on ECDH and Bloom Filters, and we have
made this protocol available for multi-party use. More information on our approach and how to use the
PSI operator can be found in the :github_nvflare_link:`PSI Example <examples/advanced/psi/README.md>`.

It is worth noting that PSI is used as a pre-processing step in the split learning example, which can be found in this
:github_nvflare_link:`notebook <examples/advanced/vertical_federated_learning/cifar10-splitnn/README.md>`.

Split Learning
--------------
Split Learning can allow the training of deep neural networks on vertically separated data. With this release, we include an [example](https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/vertical_federated_learning/cifar10-splitnn/README.md)
on how to run [split learning](https://arxiv.org/abs/1810.06060) using the CIFAR-10 dataset assuming one client holds the images and the other client holds the labels to compute losses and accuracy metrics.

Activations and corresponding gradients are being exchanged between the clients using FLARE's new communication API.

New Example for NLP
===================
The new :github_nvflare_link:`NLP-NER Example <examples/advanced/nlp-ner/README.md>` illustrates both `BERT <https://github.com/google-research/bert>`_ and
`GPT-2 <https://github.com/openai/gpt-2>`__ models from `Hugging Face <https://huggingface.co/>`_ (`BERT-base-uncased <https://huggingface.co/bert-base-uncased>`_, `GPT-2 <https://huggingface.co/gpt2>`__)
on a Named Entity Recognition (NER) task using the `NCBI disease dataset <https://pubmed.ncbi.nlm.nih.gov/24393765/>`_.

Research Areas
==============

FedSM
-----
The :github_nvflare_link:`FedSM example <research/fed-sm/README.md>` illustrates the personalized federated learning algorithm `FedSM <https://arxiv.org/abs/2203.10144>`_
accepted to CVPR 2022. It bridges the different data distributions across clients via a SoftPull mechanism and utilizes
a Super Model. A model selector is trained to predict the belongings of a particular sample to any of the clients'
personalized models or global model. The training of this model also illustrates a challenging federated learning scenario
with extreme label-imbalance, where each local training is only based on a single label towards the optimization for
classification of a number of classes equivalent to the number of clients. In this case, the higher-order moments of the
Adam optimizer are also averaged and synced together with model updates.

Auto-FedRL
----------
The :github_nvflare_link:`Auto-FedRL example <research/auto-fed-rl/README.md>` implements the automated machine learning solution described in
`Auto-FedRL: Federated Hyperparameter Optimization for Multi-institutional Medical Image Segmentation <https://arxiv.org/abs/2203.06338>`_ accepted to ECCV 2022.
Conventional hyperparameter optimization algorithms are often impractical in real-world FL applications as they involve numerous training trials,
which are often not affordable with limited computing budgets.
Auto-FedRL proposes an efficient reinforcement learning (RL)-based federated hyperparameter optimization algorithm,
in which an online RL agent can dynamically adjust the hyperparameters of each client based on the current training progress.

Quantifying Data Leakage in Federated Learning
----------------------------------------------
This research :github_nvflare_link:`example <research/quantifying-data-leakage/README.md>` contains the tools necessary to recreate the chest X-ray experiments described in
`Do Gradient Inversion Attacks Make Federated Learning Unsafe? <https://arxiv.org/abs/2202.06924>`_, accepted to IEEE Transactions on Medical Imaging.
It presents new ways to measure and visualize potential data leakage in FL using a new FLARE filter
that can quantify the data leakage for each client and visualize it as a function of the FL training rounds.
Quantifying the data leakage in FL can help determine the optimal tradeoffs between privacy-preserving techniques, such as differential privacy, and model accuracy based on quantifiable metrics.

Communication Framework Upgrades
================================
There should be no visible changes in terms of the configuration and usage patterns for the end user, but the underlying communication
layer has been improved to allow for greater flexibility and performance. These new communication features will be made generally available in next release.

**********************************
Migration to 2.3.0: Notes and Tips
**********************************
2.3.0 introduces a few API and behavior changes. This migration guide will help you to migrate from the previous NVFLARE version to the current version.

1. FLARE API
============
FLARE API is the FLAdminAPI redesigned for a better user experience in version 2.3. To understand the FLARE API usage, the relationship to
the FLAdmin API, and migration steps, please refer to :ref:`migrating_to_flare_api`.

2. Enhancements to the ``list_jobs`` command
============================================
The ``list_jobs`` command now has an option ``-r`` to display the results in reverse chronological order by submitted time. A ``-m`` option
has been added to limit the maximum number of jobs returned.

3. Redesign of communication layer
==================================
NVFLARE 2.3.0 comes with a new communication layer. Although the full-fledged features will not be generally available until the next release, the
underlying communication engine is already replaced, and you might see changes in logging.

As such, we have to change a few communication related APIs in :class:`ClientEngineExecutorSpec<nvflare.private.fed.client.client_engine_executor_spec.ClientEngineExecutorSpec>`:


FLARE 2.2.x

.. code-block:: python

    @abstractmethod
    def send_aux_request(self, topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> Shareable:
      """Send a request to Server via the aux channel.

      Implementation: simply calls the ClientAuxRunner's send_aux_request method.

      Args:
          topic: topic of the request
          request: request to be sent
          timeout: number of secs to wait for replies. 0 means fire-and-forget.
          fl_ctx: FL context

      Returns: a reply Shareable

      """
      pass

FLARE 2.3.0

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

      Returns:
          a dict of reply Shareable in the format of:
              { site_name: reply_shareable }

      """

4. Controller behavior changes
==============================
Inside :class:`ControllerSpec<nvflare.apis.controller_spec.ControllerSpec>`, the usage of ``wait_time_after_min_received``
has been changed to no longer wait if all responses are received.

.. code-block:: python

    class ControllerSpec(ABC):

        def broadcast(
          self,
          task: Task,
          fl_ctx: FLContext,
          targets: Union[List[Client], List[str], None] = None,
          min_responses: int = 0,
          wait_time_after_min_received: int = 0,
        ):

Prior to release 2.3.0,

Wait_time_after_min_received: this means after min_response received, we will wait wait_time_after_min_received.

In Release 2.3.0: 

Wait_time_after_min_received: If min_response received, but not all responses are received, we will wait wait_time_after_min_received.
If all responses are received, there is no wait.

5. Behavior changes to POC ``–stop``
====================================
In 2.2.x version, the POC stop will try to kill the process directly regardless the system state. 

In 2.3.0 version, the stop command will try with the following:

  #. Connect to the server
  #. If server can be connected, then list active jobs
  #. Abort all active jobs
  #. Call system shutdown, and wait for system to gradually shutdown
  #. Wait for system to shut down with max_timeout of 30 seconds
  #. After that, we try kill the process (this was the entirety of the 2.2.x behavior)

6. Scatter and Gather Controller API changes
============================================
A new argument has been added to :class:`ScatterAndGather<nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather>`. ``allow_empty_global_weights`` is
an optional boolean to determine whether or not to allow empty global weights and defaults to False.

Some pipelines can have empty global weights at the first round, such that clients start training from scratch without any global info.

7. Updates to the Job Scheduler Configuration
=============================================
See :ref:`job_scheduler_configuration` for information on how the Job Scheduler can be configured with different arguments.
