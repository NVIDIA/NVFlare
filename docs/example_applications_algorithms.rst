.. _example_applications:

####################
Example Applications
####################
NVIDIA FLARE has several tutorials and examples to help you get started with federated learning and to explore certain features in
`the examples directory <https://github.com/NVIDIA/NVFlare/tree/main/examples>`_.

.. toctree::
   :maxdepth: -1
   :hidden:

   examples/hello_world_examples
   examples/tutorial_notebooks
   examples/fl_algorithms
   examples/traditional_ml_examples
   examples/medical_image_analysis
   examples/federated_statistics
   examples/federated_site_policies
   examples/experiment_tracking
   examples/federated_statistics
   examples/federated_site_policies
   Federated Policies (GitHub) <https://github.com/NVIDIA/NVFlare/blob/main/examples/federated-policies>
   examples/tensorboard_streaming


The following tutorials and quickstart guides walk you through some of these examples:

  1. **Hello World Examples** which can be run from the `hello_world notebook <https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-world/hello_world.ipynb>`_.

     1.1. Workflows
          * :ref:`Hello Scatter and Gather <hello_scatter_and_gather>` - Example using the Scatter And Gather (SAG) workflow with a Numpy trainer
          * :ref:`Hello Cross-Site Validation <hello_cross_val>` - Example using the Cross Site Model Eval workflow with a Numpy trainer
          * `Hello Cyclic Weight Transfer (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-cyclic>`_ - Example using the CyclicController workflow to implement `Cyclic Weight Transfer <https://pubmed.ncbi.nlm.nih.gov/29617797/>`_ with TensorFlow as the deep learning training framework

     1.2. Deep Learning
          * :ref:`Hello PyTorch <hello_pt>` - Example image classifier using FedAvg and PyTorch as the deep learning training framework
          * :ref:`Hello TensorFlow <hello_tf2>` - Example image classifier using FedAvg and TensorFlow as the deep learning training frameworks

  2. **Tutorial notebooks**

    * `Intro to the FL Simulator <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/flare_simulator.ipynb>`_ - Shows how to use the :ref:`fl_simulator` to run a local simulation of an NVFLARE deployment to test and debug an application without provisioning a real FL project.
    * `Hello FLARE API <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/flare_api.ipynb>`_ - Goes through the different commands of the :ref:`flare_api` to show the syntax and usage of each.
    * `NVFLARE in POC Mode <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/setup_poc.ipynb>`_ - Shows how to use :ref:`POC mode <poc_command>` to test the features of a full FLARE deployment on a single machine.
    * `Provision and Start NVFLARE <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/provision.ipynb>`_ - Shows how to provision and start a secure FL system.

  3. **FL algorithms**

    * `Federated Learning with CIFAR-10 (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_ - Includes examples of using FedAvg, FedProx, FedOpt, SCAFFOLD, homomorphic encryption, and streaming of TensorBoard metrics to the server during training
    * :ref:`Federated XGBoost <federated_xgboost>` - Includes examples of histogram-based and tree-based algorithms. Tree-based algorithms also includes bagging and cyclic approaches

  4. **Traditional ML examples**

    * `Federated Linear Model with Scikit-learn (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/sklearn-linear>`_ - For an example of using NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_, a widely used open-source machine learning library that supports supervised and unsupervised learning.
    * `Federated K-Means Clustering with Scikit-learn (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/sklearn-kmeans>`_ - NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_ and k-Means.
    * `Federated SVM with Scikit-learn (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/sklearn-svm>`_ - NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_ and `SVM <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    * `Federated Learning for Random Forest based on XGBoost (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/random_forest>`_ - Example of using NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_ and `Random Forest <https://xgboost.readthedocs.io/en/stable/tutorials/rf.html>`_.

  5. **Medical Image Analysis**

    * `MONAI Integration (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/integration/monai>`_ - For an example of using NVIDIA FLARE to train a 3D medical image analysis model using federated averaging (FedAvg) and MONAI Bundle `MONAI <https://monai.io/>`_
    * `Federated Learning with Differential Privacy for BraTS18 segmentation (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/brats18>`_ - Illustrates the use of differential privacy for training brain tumor segmentation models using federated learning
    * `Federated Learning for Prostate Segmentation from Multi-source Data (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/prostate>`_ - Example of training a multi-institutional prostate segmentation model using `FedAvg <https://arxiv.org/abs/1602.05629>`_, `FedProx <https://arxiv.org/abs/1812.06127>`_, and `Ditto <https://arxiv.org/abs/2012.04221>`_

  6. **Federated Statistics**

    * :ref:`Federated Statistic Overview <federated_statistics>` - Discuss the overall federated statistics features
    * `Federated Statistics for medical imaging (Github) <https://github.com/NVIDIA/NVFlare/tree/main/examples/federated_statistics/image_stats/README.md>`_ - Example of gathering local image histogram to compute the global dataset histograms.
    * `Federated Statistics for tabular data with DataFrame (Github) <https://github.com/NVIDIA/NVFlare/tree/main/examples/federated_statistics/df_stats/README.md>`_ - Example of gathering local statistics summary from Pandas DataFrame to compute the global dataset statistics.
    * `Federated Statistics with Monai Statistics integration for Spleen CT Image (Github) <https://github.com/NVIDIA/NVFlare/tree/main/integration/monai/examples/spleen_ct_segmentation/README.md>`_ - Example demonstrated Monai statistics integration and few other features in federated statistics

  7. **Federated Site Policies**

    * `Federated Policies (Github) <https://github.com/NVIDIA/NVFlare/blob/dev/examples/federated-policies/README.rst>`_ - Discuss the federated site policies for authorization, resource and data privacy management

  8. **Experiment tracking**

    * :ref:`TensorBoard Streaming <tensorboard_streaming>` - Example building on Hello PyTorch with TensorBoard streaming from clients to server


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
to penalize a client's local weights based on deviation from the global model. An example configuration can be found in
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

Federated XGBoost
^^^^^^^^^^^^^^^^^

* `Federated XGBoost (GitHub) <https://github.com/NVIDIA/NVFlare/tree/main/examples/xgboost>`_ - Includes examples of histogram-based and tree-based algorithms. Tree-based algorithms also includes bagging and cyclic approaches

Federated Analytics
^^^^^^^^^^^^^^^^^^^

* `Federated Statistics for medical imaging (Github) <https://github.com/NVIDIA/NVFlare/tree/main/examples/federated_statistics/image_stats/README.md>`_ - Example of gathering local image histogram to compute the global dataset histograms.
* `Federated Statistics for tabular data with DataFrame (Github) <https://github.com/NVIDIA/NVFlare/tree/main/examples/federated_statistics/df_stats/README.md>`_ - Example of gathering local statistics summary from Pandas DataFrame to compute the global dataset statistics.
* `Federated Statistics with Monai Statistics integration for Spleen CT Image (Github) <https://github.com/NVIDIA/NVFlare/tree/main/integration/monai/examples/spleen_ct_segmentation/README.md>`_ - Example demonstrated Monai statistics integration and few other features in federated statistics
