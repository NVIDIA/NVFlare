.. _key_features:

############
Key Features
############

Key Features in FLARE v2.2 - What's New
=======================================

With FLARE v2.2, the primary goals are to:
 - Accelerate the federated learning workflow
 - Simplify deploying a federated learning project in the real-world
 - Support federated data science
 - Enable integration with other platforms

To accomplish these goals, a set of key new tools and features were developed, including:
 - FL Simulator
 - FLARE Dashboard
 - Site-policy management
 - Federated XGboost <https://github.com/NVIDIA/NVFlare/tree/main/examples/xgboost>
 - Federated Statistics <https://github.com/NVIDIA/NVFlare/tree/main/examples/federated_statistics>
 - MONAI Integration <https://github.com/NVIDIA/NVFlare/tree/main/integration/monai>

The sections below provide an overview of these features.  For more detailed documentation and usage information, refer to the :ref:`User Guide <user_guide>` and :ref:`Programming Guide <programming_guide>`.

FL Simulator
------------
The :ref:`FL Simulator <fl_simulator>` is a lightweight tool that allows you to build, debug, and run a FLARE
application locally without explicitly deploying a provisioned FL system.  The FL Simulator provides both a CLI for
interactive use and an API for developing workflows programmatically. Clients are implemented using threads for each
client. If running in an environment with limited resources, multiple clients can be run sequentially using single
threads (or GPUs). This allows for testing the scalability of an application even with limited resources.

Users can run the FL Simulator in a python environment to debug FLARE application code directly. Jobs can be submitted
directly to the simulator without debugging, just as in a production FLARE deployment.  This allows you to build, debug,
and test in an interactive environment, and then deploy the same application in production without modification.

POC mode upgrade
----------------
For researchers who prefer to use :ref:`POC (proof of concept) <poc_command>` mode, the usage has been improved for
provisioning and starting a server and clients locally.

FLARE Dashboard and Provisioning
--------------------------------
The :ref:`FLARE Dashboard <nvflare_dashboard_ui>` provides a web UI that allows a project administrator to configure a
project and distribute client startup kits without the need to gather client information up-front, or manually configure
the project using the usual ``project.yml`` configuration.  Once the details of the project have been configured,
:ref:`provisioning <provisioning>` of client systems and FLARE Console users, is done on the fly. The web UI allows users to
register, and once approved, download project startup kits on-demand.  For those who wish to provision manually, the
provisioning CLI is still included in the main nvflare CLI:

.. code-block:: shell

  nvflare provision -h

The CLI method of provisioning has also been enhanced to allow for :ref:`dynamic provisioning <dynamic_provisioning>`,
allowing the addition of new sites or users without the need to re-provision existing sites.

In addition to these enhancements to the provisioning workflow, we provide some new tools to simplify local deployment
and troubleshoot client connectivity.  First is a ``docker-compose`` :ref:`utility <docker_compose>` that allows the
administrator to provision a set of local startup kits, and issue ``docker-compose up`` to start the server and connect
all clients.

We also provide a new :ref:`pre-flight check <preflight_check>` to help remote sites troubleshoot potential environment
and connectivity issues before attempting to connect to the FL Server.

.. code-block:: shell

  nvflare preflight-check -h

This command will examine all available provisioned packages (server, admin, clients, overseers) to check connections
between the different components (server, clients, overseers), ports, dns, storage access, etc., and provide suggestions
for how to fix any potential issues.

Federated Data Science
----------------------

Federated XGBoost
"""""""""""""""""

XGBoost is a popular machine learning method used by applied data scientists in a wide variety of applications. In FLARE v2.2,
we introcuce federated XGBoost integration, with a controller and executor that run distributed XGBoost training among a group
of clients.  See the `hello-xgboost example <https://github.com/NVIDIA/NVFlare/tree/main/examples/xgboost>`_ to get started.

Federated Statistics
""""""""""""""""""""
Before implementing a federated training application, a data scientist often performs a process of data exploration,
analysis, and feature engineering. One method of data exploration is to explore the statistical distribution of a dataset.
With FLARE v2.2, we indroduce federated statistics operators - a server controller and client executor.  With these
pre-defined operators, users define the statistics to be calculated locally on each client dataset, and the workflow
controller generates an output json file that contains global as well as individual site statistics.  This data can be
visualized to allow site-to-site and feature-to-feature comparison of metrics and histograms across the set of clients.

Site Policy Management and Security
-----------------------------------

Although the concept of client authorization and security policies are not new in FLARE, version 2.2 has shifted to
federated :ref:`site policy management <site_policy_management>`. In the past, authorization policies were defined by the
project administrator at time of provisioning, or in the job specification.  The shift to federated site policy allows
individual sites to control:

 - Site security policy
 - Resource management
 - Data privacy

With these new federated controls, the individual site has full control over authorization policies, what resources are
available to the client workflow, and what security filters are applied to incoming and outgoing traffic.

In addition to the federated site policy, FLARE v2.2 also introduces secure logging and security auditing.  Secure
logging, when enabled, limits client output to only file and line numbers in the event of an error, rather than a full
traceback, preventing unintentionally disclosing site-specific information to the project administrator.  Secure
auditing keeps a site-specific log of all access and commands performed by the project admin.


Key Features in FLARE 2.1
=========================
    - :ref:`High Availability (HA) <high_availability>` supports multiple FL Servers and automatically cuts
      over to another server when the currently active server becomes unavailable.
    - :ref:`Multi-Job Execution <multi_job>` supports resource-based multi-job execution by allowing for concurrent runs
      provided resources required by the jobs are satisfied.

NVIDIA FLARE provides a set of commonly-used algorithms to illustrate best practices and allow simplified development of
common Federated Learning Workflows.

Key Features of the FLARE Platform
==================================

Training Workflows
------------------
    - :ref:`Scatter and Gather (SAG) <scatter_and_gather_workflow>` is a reference implementation of the default
      workflow in previous versions of NVIDIA FLARE.  SAG implements a hub and spoke model in which the central server
      Controller broadcasts Tasks to be Executed on the client Workers.  After the client Executors return their Task's
      Shareable result (e.g., client model weights from DL training), the server Controller aggregates the results, for
      example with a federated weighted average.
    - :ref:`Cyclic <cyclic>` is a reference implementation of a cyclic workflow, in which the central server issues a
      series of tasks to be scheduled for cyclic execution among a group of clients.  The client worker Executor passes
      the Task's Shareable result to the next client for further execution, and so on, until the final client returns
      the final Shareable to the server.

Evaluation Workflows
--------------------
    - :ref:`Cross site model validation <cross_site_model_evaluation>` is a workflow that allows validation of each
      client model and the server global model against each client dataset.

      Data is not shared, rather the collection of models is distributed to each client site to run local validation.

      The results of local validation are collected by the server to construct an all-to-all matrix of
      model performance vs. client dataset.

    - :ref:`Global model evaluation <cross_site_model_evaluation>` is a subset of cross-site model validation in which
      the server's global model is distributed to each client for evaluation on the client's local dataset.

Privacy Preservation Algorithms
-------------------------------
Privacy preserving algorithms in NVIDIA FLARE are implemented as :ref:`filters <filters_for_privacy>`
that can be applied as data is sent or received between peers.

    - Differential privacy:

        - Exclude specific variables (:class:`ExcludeVars<nvflare.app_common.filters.exclude_vars.ExcludeVars>`)
        - truncate weights by percentile (:class:`PercentilePrivacy<nvflare.app_common.filters.percentile_privacy.PercentilePrivacy>`)
        - apply sparse vector techniques (:class:`SVTPrivacy<nvflare.app_common.filters.svt_privacy.SVTPrivacy>`).

    - Homomorphic encryption: NVIDIA FLARE provides homomorphic encryption and decryption
      filters that can be used by clients to encrypt Shareable data before sending it to a peer.

      The server does not have a decryption key but using HE can operate on the encrypted data to aggregate
      and return the encrypted aggregated data to clients.

      Clients can then decrypt the data with their local key and continue local training.

Learning Algorithms
-------------------

    - Fed average (implemented through the :ref:`scatter_and_gather_workflow`) - In the federated averaging workflow,
      a set of initial weights is distributed to client Workers who perform local training.  After local training,
      clients return their local weights as a Shareables that are aggregated (averaged).  This new set of global average
      weights is redistributed to clients and the process repeats for the specified number of rounds.
    - `FedProx <https://arxiv.org/abs/1812.06127>`_ (example configuration can be found in cifar10_fedprox of `CIFAR-10 example <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_) -
      implements a :class:`Loss function <nvflare.app_common.pt.pt_fedproxloss.PTFedProxLoss>` to penalize a client's
      local weights based on deviation from the global model.
    - `FedOpt <https://arxiv.org/abs/2003.00295>`_ (example configuration can be found in cifar10_fedopt of `CIFAR-10 example <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_) -
      implements a :class:`ShareableGenerator <nvflare.app_common.pt.pt_fedopt.PTFedOptModelShareableGenerator>` that
      can use a specified Optimizer and Learning Rate Scheduler when updating the global model.

Example Applications
---------------------

NVIDIA FLARE provide a rich set of :ref:`example applications <example_applications>` to walk your through the whole
process.
