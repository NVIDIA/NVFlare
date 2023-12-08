.. _fl_algorithms:

********************
FL Algorithms
********************

Federated Averaging
-------------------
In NVIDIA FLARE, FedAvg is implemented through the :ref:`scatter_and_gather_workflow`. In the federated averaging workflow,
a set of initial weights is distributed to client workers who perform local training.  After local training, clients
return their local weights as a Shareables that are aggregated (averaged).  This new set of global average weights is
redistributed to clients and the process repeats for the specified number of rounds. 

FedProx
-------
`FedProx <https://arxiv.org/abs/1812.06127>`_ implements a :class:`Loss function <nvflare.app_common.pt.pt_fedproxloss.PTFedProxLoss>`
to penalize a client's local weights based on deviation from the global model. An example configuration can be found in
cifar10_fedprox of the :github_nvflare_link:`CIFAR-10 example <examples/advanced/cifar10>`.

FedOpt
------
`FedOpt <https://arxiv.org/abs/2003.00295>`_ implements a :class:`ShareableGenerator <nvflare.app_common.pt.pt_fedopt.PTFedOptModelShareableGenerator>`
that can use a specified Optimizer and Learning Rate Scheduler when updating the global model. An example configuration
can be found in cifar10_fedopt of :github_nvflare_link:`CIFAR-10 example <examples/advanced/cifar10>`.

SCAFFOLD
--------
`SCAFFOLD <https://arxiv.org/abs/1910.06378>`_ uses a slightly modified version of the CIFAR-10 Learner implementation,
namely the `CIFAR10ScaffoldLearner`, which adds a correction term during local training following the `implementation <https://github.com/Xtra-Computing/NIID-Bench>`_
as described in `Li et al. <https://arxiv.org/abs/2102.02079>`_. An example configuration can be found in cifar10_scaffold of :github_nvflare_link:`CIFAR-10 example <examples/advanced/cifar10>`.

Ditto
-----
`Ditto <https://arxiv.org/abs/2012.04221>`_ uses a slightly modified version of the prostate Learner implementation,
namely the `ProstateDittoLearner`, which decouples local personalized model from global model via an additional model
training and a controllable prox term. See the :github_nvflare_link:`prostate segmentation example <examples/advanced/prostate>`
for an example with ditto in addition to FedProx, FedAvg, and centralized training.

Federated XGBoost
-----------------

NVFlare supports federated learning using popular gradient boosting library XGBoost.
It uses XGBoost library with federated plugin (xgboost version >= 1.7.0rc1) to perform the learning.

Using XGBoost with NVFlare has following benefits compared with running federated XGBoost directly,

* XGBoost instance's life-cycle is managed by NVFlare. Both XGBoost client and server
  are started/stopped automatically by NVFlare workflow.
* For histogram-based XGBoost federated server can be configured automatically with auto-assigned port number.
* When mutual TLS is used, the certificates are managed by NVFlare using existing
  provisioning process.
* No need to manually configure each instance. Instance specific parameters
  like code:`rank` are assigned automatically by the NVFlare controller.

* :github_nvflare_link:`Federated Horizontal XGBoost (GitHub) <examples/advanced/xgboost>` - Includes examples of histogram-based and tree-based algorithms. Tree-based algorithms also includes bagging and cyclic approaches
* :github_nvflare_link:`Federated Vertical XGBoost (GitHub) <examples/advanced/vertical_xgboost>` - Example using Private Set Intersection and XGBoost on vertically split HIGGS data.

Federated Analytics
-------------------

* :github_nvflare_link:`Federated Statistics for medical imaging (Github) <examples/advanced/federated-statistics/image_stats/README.md>` - Example of gathering local image histogram to compute the global dataset histograms.
* :github_nvflare_link:`Federated Statistics for tabular data with DataFrame (Github) <examples/advanced/federated-statistics/df_stats/README.md>` - Example of gathering local statistics summary from Pandas DataFrame to compute the global dataset statistics.
* :github_nvflare_link:`Federated Statistics with Monai Statistics integration for Spleen CT Image (Github) <integration/monai/examples/spleen_ct_segmentation/README.md>` - Example demonstrated Monai statistics integration and few other features in federated statistics

