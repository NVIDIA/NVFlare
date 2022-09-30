.. _example_applications:

####################
Example Applications
####################
NVIDIA FLARE has several examples to help you get started with federated learning and to explore certain features in
`the examples directory <https://github.com/NVIDIA/NVFlare/tree/main/examples>`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   examples/hello_scatter_and_gather
   examples/hello_cross_val
   examples/hello_pt 
   examples/hello_pt_tb
   examples/hello_tf2
   Hello Cyclic Weight Transfer (GitHub) <https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-cyclic>
   Federated Analysis (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/federated_analysis>
   Federated Learning with CIFAR-10 (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>
   Hello MONAI (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-monai>
   Differential Privacy for BraTS18 Segmentation (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/brats18>
   Prostate Segmentation from Multi-source Data (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/prostate>


The following quickstart guides walk you through some of these examples:

  * **Basic Examples**
  
    * :ref:`Hello Scatter and Gather <hello_scatter_and_gather>` - Example using the Scatter And Gather (SAG) workflow with a Numpy trainer
    * :ref:`Hello Cross-Site Validation <hello_cross_val>` - Example using the Cross Site Model Eval workflow with a Numpy trainer
    * :ref:`Hello PyTorch <hello_pt>` - Example image classifier using FedAvg and PyTorch as the deep learning training framework
    * :ref:`Hello PyTorch with TensorBoard <hello_pt_tb>` - Example building on Hello PyTorch with TensorBoard streaming from clients to server
    * :ref:`Hello TensorFlow <hello_tf2>` - Example image classifier using FedAvg and TensorFlow as the deep learning training frameworks
    * `Hello Cyclic Weight Transfer (GitHub) <https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-cyclic>`_ - Example using the CyclicController workflow to implement `Cyclic Weight Transfer <https://pubmed.ncbi.nlm.nih.gov/29617797/>`_ with TensorFlow as the deep learning training framework

  * **Advanced examples**
    
    * `Federated Analysis (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/federated_analysis>`_ - Example of gathering local data summary statistics to compute global dataset statistics
    * `Federated Learning with CIFAR-10 (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_ - Includes examples of using FedAvg, FedProx, FedOpt, SCAFFOLD, homomorphic encryption, and streaming of TensorBoard metrics to the server during training
    * `Hello MONAI (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-monai>`_ - Example medical image analysis using FedAvg and `MONAI <https://monai.io/>`_
    * `Differential Privacy for BraTS18 segmentation (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/brats18>`_ - Illustrates the use of differential privacy for training brain tumor segmentation models using federated learning
    * `Prostate Segmentation from Multi-source Data (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/prostate>`_ - Example of training a multi-institutional prostate segmentation model using `FedAvg <https://arxiv.org/abs/1602.05629>`_, `FedProx <https://arxiv.org/abs/1812.06127>`_, and `Ditto <https://arxiv.org/abs/2012.04221>`_

For the complete collection of example applications, see https://github.com/NVIDIA/NVFlare/tree/main/examples.

Custom Code in Example Apps
===========================
There are several ways to make :ref:`custom code <custom_code>` available to clients when using NVIDIA FLARE.
Most hello-* examples use a custom folder within the FL application.
Note that using a custom folder in the app needs to be :ref:`allowed <troubleshooting_byoc>` when using secure provisioning.
By default, this option is disabled in the secure mode. POC mode, however, will work with custom code by default.

In contrast, the `CIFAR-10 <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_,
`prostate segmentation <https://github.com/NVIDIA/NVFlare/tree/main/examples/prostate>`_,
and `BraTS18 segmentation <https://github.com/NVIDIA/NVFlare/tree/main/examples/brats18>`_ examples assume that the
learner code is already installed on the client's system and available in the PYTHONPATH.
Hence, the app folders do not include the custom code there.
The PYTHONPATH is set in the ``run_poc.sh`` or ``run_secure.sh`` scripts of the example.
Running these scripts as described in the README will make the learner code available to the clients.


.. _fl_algorithms:

Federated Learning Algorithms
=============================

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
