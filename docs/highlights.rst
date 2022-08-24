.. _highlights:

##########
Highlights
##########

New in NVIDIA FLARE 2.1.0
=========================
    - :ref:`High Availability (HA) <high_availability>` supports multiple FL Servers and automatically cuts
      over to another server when the currently active server becomes unavailable.
    - :ref:`Multi-Job Execution <multi_job>` supports resource-based multi-job execution by allowing for concurrent runs
      provided resources required by the jobs are satisfied.

NVIDIA FLARE Key Features
=========================
NVIDIA FLARE provides a set of commonly-used algorithms to illustrate best practices and allow simplified development of
common Federated Learning Workflows.

Training workflows
------------------
    - :ref:`Scatter and Gather (SAG) <scatter_and_gather_workflow>` is a reference implementation of the default
      workflow in previous versions of NVIDIA FLARE.  SAG implements a hub and spoke model in which the central server
      Controller broadcasts Tasks to be Executed on the client Workers.  After the client Executors return their Task’s
      Shareable result (e.g., client model weights from DL training), the server Controller aggregates the results, for
      example with a federated weighted average.
    - :ref:`Cyclic <cyclic>` is a reference implementation of a cyclic workflow, in which the central server issues a
      series of tasks to be scheduled for cyclic execution among a group of clients.  The client worker Executor passes
      the Task's Shareable result to the next client for further execution, and so on, until the final client returns
      the final Shareable to the server.

Evaluation workflows
--------------------
    - :ref:`Cross site model validation <cross_site_model_evaluation>` is a workflow that allows validation of each
      client model and the server global model against each client dataset.

      Data is not shared, rather the collection of models is distributed to each client site to run local validation.

      The results of local validation are collected by the server to construct an all-to-all matrix of
      model performance vs. client dataset.

    - :ref:`Global model evaluation <cross_site_model_evaluation>` is a subset of cross-site model validation in which
      the server’s global model is distributed to each client for evaluation on the client’s local dataset.

Privacy preservation algorithms
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

Learning algorithms
-------------------

    - Fed average (implemented through the :ref:`scatter_and_gather_workflow`) - In the federated averaging workflow,
      a set of initial weights is distributed to client Workers who perform local training.  After local training,
      clients return their local weights as a Shareables that are aggregated (averaged).  This new set of global average
      weights is redistributed to clients and the process repeats for the specified number of rounds.
    - `FedProx <https://arxiv.org/abs/1812.06127>`_ (example configuration can be found in cifar10_fedprox of `CIFAR-10 example <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_) -
      implements a :class:`Loss function <nvflare.app_common.pt.pt_fedproxloss.PTFedProxLoss>` to penalize a client’s
      local weights based on deviation from the global model.
    - `FedOpt <https://arxiv.org/abs/2003.00295>`_ (example configuration can be found in cifar10_fedopt of `CIFAR-10 example <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_) -
      implements a :class:`ShareableGenerator <nvflare.app_common.pt.pt_fedopt.PTFedOptModelShareableGenerator>` that
      can use a specified Optimizer and Learning Rate Scheduler when updating the global model.

Examples
---------

Nvidia FLARE provide a rich set of :ref:`example applications <example_applications>` to walk your through the whole
process.
