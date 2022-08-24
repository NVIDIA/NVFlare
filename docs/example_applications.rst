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
