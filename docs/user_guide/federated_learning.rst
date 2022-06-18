Federated Learning
==================

.. _fl_algorithms:

Federated Learning Algorithms
-----------------------------

Federated Averaging
^^^^^^^^^^^^^^^^^^^
In NVIDIA FLARE, FedAvg is implemented through the :ref:`scatter_and_gather_workflow`. In the federated averaging workflow,
a set of initial weights is distributed to client workers who perform local training.  After local training, clients
return their local weights as a Shareables that are aggregated (averaged).  This new set of global average weights is
redistributed to clients and the process repeats for the specified number of rounds.

FedProx
^^^^^^^
`FedProx <https://arxiv.org/abs/1812.06127>`_ implements a :class:`Loss function <nvflare.app_common.pt.pt_fedproxloss.PTFedProxLoss>`
to penalize a client’s local weights based on deviation from the global model. An example configuration can be found in
cifar10_fedprox of the `CIFAR-10 example <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_.

FedOpt
^^^^^^
`FedOpt <https://arxiv.org/abs/2003.00295>`_ implements a :class:`ShareableGenerator <nvflare.app_common.pt.pt_fedopt.PTFedOptModelShareableGenerator>`
that can use a specified Optimizer and Learning Rate Scheduler when updating the global model. An example configuration
can be found in cifar10_fedopt of `CIFAR-10 example <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_.

SCAFFOLD
^^^^^^^^
`SCAFFOLD <https://arxiv.org/abs/1910.06378>`_ uses a slightly modified version of the CIFAR-10 Learner implementation,
namely the `CIFAR10ScaffoldLearner`, which adds a correction term during local training following the `implementation <https://github.com/Xtra-Computing/NIID-Bench>`_
as described in `Li et al. <https://arxiv.org/abs/2102.02079>`_

Ditto
^^^^^
`Ditto <https://arxiv.org/abs/2012.04221>`_ uses a slightly modified version of the prostate Learner implementation,
namely the `ProstateDittoLearner`, which decouples local personalized model from global model via an additional model
training and a controllable prox term. See the `prostate segmentation example <https://github.com/NVIDIA/NVFlare/tree/main/examples/prostate>`_
for an example with ditto in addition to FedProx, FedAvg, and centralized training.

Federated Analytics
-------------------
Federated analytics may be used to gather information about the data at various sites. An example can be found in the
`Federated Analysis example <https://github.com/NVIDIA/NVFlare/tree/main/examples/federated_analysis>`_.
