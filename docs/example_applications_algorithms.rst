.. _example_applications:

####################
Example Applications
####################
NVIDIA FLARE has several tutorials and examples to help you get started with federated learning and to explore certain features in the
:github_nvflare_link:`examples directory <examples>`.

1. Hello World Examples
=======================
Can be run from :github_nvflare_link:`hello_world notebook <examples/hello-world/hello_world.ipynb>`.

.. toctree::
  :maxdepth: 1
  :hidden:

  examples/hello_world_examples

1.1. Deep Learning to Federated Learning
----------------------------------------

  * :github_nvflare_link:`Deep Learning to Federated Learning (GitHub) <examples/hello-world/ml-to-fl>` - Example for converting Deep Learning (DL) to Federated Learning (FL) using the Client API.

1.2. Workflows
--------------

  * :ref:`Hello FedAvg with NumPy <hello_fedavg_numpy>` - Example using the FedAvg workflow with a NumPy trainer
  * :ref:`Hello Cross-Site Validation <hello_cross_val>` - Example using the Cross Site Eval workflow, also demonstrates running cross site validation using the previous training results.
  * :github_nvflare_link:`Hello Cyclic Weight Transfer (GitHub) <examples/hello-world/hello-cyclic>` - Example using the CyclicController workflow to implement `Cyclic Weight Transfer <https://pubmed.ncbi.nlm.nih.gov/29617797/>`_ with TensorFlow as the deep learning training framework
  * :github_nvflare_link:`Swarm Learning <examples/advanced/swarm_learning>` - Example using Swarm Learning and Client-Controlled Cross-site Evaluation workflows.
  * :github_nvflare_link:`Client-Controlled Cyclic Weight Transfer <examples/hello-world/step-by-step/cifar10/cyclic_ccwf>` - Example using Client-Controlled Cyclic workflow using Client API.

1.3. Deep Learning
------------------

  * :ref:`Hello PyTorch <hello_pt_job_api>` - Example image classifier using FedAvg and PyTorch as the deep learning training framework
  * :ref:`Hello TensorFlow <hello_tf_job_api>` - Example image classifier using FedAvg and TensorFlow as the deep learning training frameworks


2. Step-By-Step Example Series
==============================

:github_nvflare_link:`Step-by-Step Examples (GitHub) <examples/hello-world/step-by-step/>` - Step-by-step examples series with CIFAR-10 (image data) and HIGGS (tabular data) to showcase different FLARE features, workflows, and APIs.

2.1 CIFAR-10 Image Data Examples
--------------------------------

  * :github_nvflare_link:`image_stats <examples/hello-world/step-by-step/cifar10/stats/image_stats.ipynb>` - federated statistics (histograms) of CIFAR10.
  * :github_nvflare_link:`sag <examples/hello-world/step-by-step/cifar10/sag/sag.ipynb>` - scatter and gather (SAG) workflow with PyTorch with Client API.
  * :github_nvflare_link:`sag_deploy_map <examples/hello-world/step-by-step/cifar10/sag_deploy_map/sag_deploy_map.ipynb>` - scatter and gather workflow with deploy_map configuration for deployment of apps to different sites using the Client API.
  * :github_nvflare_link:`sag_model_learner <examples/hello-world/step-by-step/cifar10/sag_model_learner/sag_model_learner.ipynb>` - scatter and gather workflow illustrating how to write client code using the ModelLearner.
  * :github_nvflare_link:`sag_executor <examples/hello-world/step-by-step/cifar10/sag_executor/sag_executor.ipynb>` - scatter and gather workflow demonstrating show to write client-side executors.
  * :github_nvflare_link:`sag_mlflow <examples/hello-world/step-by-step/cifar10/sag_mlflow/sag_mlflow.ipynb>` - MLflow experiment tracking logs with the Client API in scatter & gather workflows.
  * :github_nvflare_link:`sag_he <examples/hello-world/step-by-step/cifar10/sag_he/sag_he.ipynb>` - homomorphic encyption using Client API and POC -he mode.
  * :github_nvflare_link:`cse <examples/hello-world/step-by-step/cifar10/cse/cse.ipynb>` - cross-site evaluation using the Client API.
  * :github_nvflare_link:`cyclic <examples/hello-world/step-by-step/cifar10/cyclic/cyclic.ipynb>` - cyclic weight transfer workflow with server-side controller.
  * :github_nvflare_link:`cyclic_ccwf <examples/hello-world/step-by-step/cifar10/cyclic_ccwf/cyclic_ccwf.ipynb>` - client-controlled cyclic weight transfer workflow with client-side controller.
  * :github_nvflare_link:`swarm <examples/hello-world/step-by-step/cifar10/swarm/swarm.ipynb>` - swarm learning and client-side cross-site evaluation with Client API.

2.2 HIGGS Tabular Data Examples
-------------------------------

  * :github_nvflare_link:`tabular_stats <examples/hello-world/step-by-step/higgs/stats/tabular_stats.ipynb>`- federated stats tabular histogram calculation.
  * :github_nvflare_link:`sklearn_linear <examples/hello-world/step-by-step/higgs/sklearn-linear/sklearn_linear.ipynb>`- federated linear model (logistic regression on binary classification) learning on tabular data.
  * :github_nvflare_link:`sklearn_svm <examples/hello-world/step-by-step/higgs/sklearn-svm/sklearn_svm.ipynb>`- federated SVM model learning on tabular data.
  * :github_nvflare_link:`sklearn_kmeans <examples/hello-world/step-by-step/higgs/sklearn-kmeans/sklearn_kmeans.ipynb>`- federated k-Means clustering on tabular data.
  * :github_nvflare_link:`xgboost <examples/hello-world/step-by-step/higgs/xgboost/xgboost_horizontal.ipynb>`- federated horizontal xgboost learning on tabular data with bagging collaboration.


3. Tutorial Notebooks
=====================

  * :github_nvflare_link:`Intro to the FL Simulator <examples/tutorials/flare_simulator.ipynb>` - Shows how to use the :ref:`fl_simulator` to run a local simulation of an NVFLARE deployment to test and debug an application without provisioning a real FL project.
  * :github_nvflare_link:`Hello FLARE API <examples/tutorials/flare_api.ipynb>` - Goes through the different commands of the :ref:`flare_api` to show the syntax and usage of each.
  * :github_nvflare_link:`NVFLARE in POC Mode <examples/tutorials/setup_poc.ipynb>` - Shows how to use :ref:`POC mode <poc_command>` to test the features of a full FLARE deployment on a single machine.
  * :github_nvflare_link:`Job CLI Tutorial <examples/tutorials/job_cli.ipynb>` - Walks through the different commands of the Job CLI and showcases syntax and example usages.

4. Federated Learning Algorithms
================================

  * :github_nvflare_link:`Federated Learning with CIFAR-10 (GitHub) <examples/advanced/cifar10>` - Includes examples of using FedAvg, FedProx, FedOpt, SCAFFOLD, homomorphic encryption, and streaming of TensorBoard metrics to the server during training

  .. toctree::
    :maxdepth: 2

    examples/fl_algorithms

5. Privacy Preserving Algorithms
================================
Privacy preserving algorithms in NVIDIA FLARE are implemented as :ref:`filters <filters_for_privacy>` that can be applied as data is sent or received between peers.

  * :github_nvflare_link:`Federated Learning with CIFAR-10 (GitHub) <examples/advanced/cifar10>` - Includes examples of using FedAvg, FedProx, FedOpt, SCAFFOLD, homomorphic encryption, and streaming of TensorBoard metrics to the server during training
  * :github_nvflare_link:`Differential Privacy for BraTS18 segmentation (GitHub) <examples/advanced/brats18>`- Example using SVT Differential Privacy for BraTS18 segmentation.

6. Traditional ML examples
==========================

  * :github_nvflare_link:`Federated Linear Model with Scikit-learn (GitHub) <examples/advanced/sklearn-linear>` - For an example of using NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_, a widely used open-source machine learning library that supports supervised and unsupervised learning.
  * :github_nvflare_link:`Federated K-Means Clustering with Scikit-learn (GitHub) <examples/advanced/sklearn-kmeans>` - NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_ and k-Means.
  * :github_nvflare_link:`Federated SVM with Scikit-learn (GitHub) <examples/advanced/sklearn-svm>` - NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_ and `SVM <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
  * :github_nvflare_link:`Federated Horizontal XGBoost (GitHub) <examples/advanced/xgboost>` - Includes examples of histogram-based and tree-based algorithms. Tree-based algorithms also includes bagging and cyclic approaches
  * :github_nvflare_link:`Federated Learning for Random Forest based on XGBoost (GitHub) <examples/advanced/random_forest>` - Example of using NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_ and `Random Forest <https://xgboost.readthedocs.io/en/stable/tutorials/rf.html>`_.
  * :github_nvflare_link:`Federated Vertical XGBoost (GitHub) <examples/advanced/vertical_xgboost>` - Example using Private Set Intersection and XGBoost on vertically split HIGGS data.

7. Medical Image Analysis
=========================

  * :github_nvflare_link:`MONAI Integration (GitHub) <integration/monai>` - For an example of using NVIDIA FLARE to train a 3D medical image analysis model using federated averaging (FedAvg) and MONAI Bundle `MONAI <https://monai.io/>`_
  * :github_nvflare_link:`Federated Learning with Differential Privacy for BraTS18 segmentation (GitHub) <examples/advanced/brats18>` - Illustrates the use of differential privacy for training brain tumor segmentation models using federated learning
  * :github_nvflare_link:`Federated Learning for Prostate Segmentation from Multi-source Data (GitHub) <examples/advanced/prostate>` - Example of training a multi-institutional prostate segmentation model using `FedAvg <https://arxiv.org/abs/1602.05629>`_, `FedProx <https://arxiv.org/abs/1812.06127>`_, and `Ditto <https://arxiv.org/abs/2012.04221>`_

8. Federated Statistics
=======================

  * :ref:`Federated Statistic Overview <federated_statistics>` - Discuss the overall federated statistics features.
  * :github_nvflare_link:`Federated Statistics for medical imaging (Github) <examples/advanced/federated-statistics/image_stats/README.md>` - Example of gathering local image histogram to compute the global dataset histograms.
  * :github_nvflare_link:`Federated Statistics for tabular data with DataFrame (Github) <examples/advanced/federated-statistics/df_stats/README.md>` - Example of gathering local statistics summary from Pandas DataFrame to compute the global dataset statistics.
  * :github_nvflare_link:`Federated Statistics with Monai Statistics integration for Spleen CT Image (Github) <integration/monai/examples/README.md>` - Example demonstrated Monai statistics integration and few other features in federated statistics
  
  .. toctree::
    :maxdepth: 1
    :hidden:

    examples/federated_statistics_overview

9. Federated Site Policies
==========================

  * :github_nvflare_link:`Federated Policies (Github) <examples/advanced/federated-policies/README.rst>` - Discuss the federated site policies for authorization, resource and data privacy management
  * :github_nvflare_link:`Custom Authentication (Github) <examples/advanced/custom_authentication/README.rst>` - Show the custom authentication policy and secure mode.
  * :github_nvflare_link:`Job-Level Authorization (Github) <examples/advanced/job-level-authorization/README.md>` - Show the job-level authorization policy and secure mode.
  * :github_nvflare_link:`KeyCloak Site Authentication Integration (Github) <examples/advanced/keycloak-site-authentication/README.md>` - Demonstrate KeyCloak integration for supporting site-specific authentication.

10. Experiment Tracking
=======================

  * :github_nvflare_link:`FL Experiment Tracking with TensorBoard Streaming <examples/advanced/experiment-tracking/tensorboard>` - :ref:`(documentation) <tensorboard_streaming>` - Example building on Hello PyTorch with TensorBoard streaming from clients to server
  * :github_nvflare_link:`FL Experiment Tracking with MLflow <examples/advanced/experiment-tracking/mlflow>` - :ref:`(documentation) <experiment_tracking_mlflow>`- Example integrating Hello PyTorch with MLflow with streaming from clients to server
  * :github_nvflare_link:`FL Experiment Tracking with Weights and Biases <examples/advanced/experiment-tracking/wandb>` - Example integrating Hello PyTorch with Weights and Biases streaming capability from clients to server.
  * :github_nvflare_link:`MONAI FLARE Integration Experiment Tracking <integration/monai/examples/spleen_ct_segmentation_local#51-experiment-tracking-with-mlflow>` - Example using FLARE and MONAI integration with experiment tracking streaming from clients to server.

  .. toctree::
    :maxdepth: 1
    :hidden:

    examples/tensorboard_streaming
    examples/fl_experiment_tracking_mlflow

11.  Natural Language Processing (NLP)
======================================

  * :github_nvflare_link:`NLP-NER (Github) <examples/advanced/nlp-ner/README.md>` - Illustrates both `BERT <https://github.com/google-research/bert>`_ and `GPT-2 <https://github.com/openai/gpt-2>`_ models from `Hugging Face <https://huggingface.co/>`_ (`BERT-base-uncased <https://huggingface.co/bert-base-uncased>`_, `GPT-2 <https://huggingface.co/gpt2>`_) on a Named Entity Recognition (NER) task using the `NCBI disease dataset <https://pubmed.ncbi.nlm.nih.gov/24393765/>`_.

12. FL Hierarchical Unification Bridge (HUB)
============================================

  * :github_nvflare_link:`FL HUB <examples/advanced/fl_hub>` - Example for FL HUB allowing hierarchical interaction between several levels of FLARE FL systems.

13. Federated Large Language Model (LLM)
========================================

  * :github_nvflare_link:`Parameter Efficient Fine Turning <integration/nemo/examples/peft>` - Example utilizing NeMo's PEFT methods to adapt a LLM to a downstream task.
  * :github_nvflare_link:`Prompt-Tuning Example <integration/nemo/examples/prompt_learning>` - Example for using FLARE with NeMo for prompt learning.
  * :github_nvflare_link:`Supervised Fine Tuning (SFT) <integration/nemo/examples/supervised_fine_tuning>` - Example to fine-tune all parameters of a LLM on supervised data.
  * :github_nvflare_link:`LLM Tuning via HuggingFace SFT Trainer <examples/advanced/llm_hf>` - Example for using FLARE with a HuggingFace trainer for LLM tuning tasks.


14. Graph Neural Network (GNN)
==============================

  * :github_nvflare_link:`Protein Classification <examples/advanced/gnn#federated-gnn-on-graph-dataset-using-inductive-learning>` - Example using GNNs for Protein Classification using `PPI <http://snap.stanford.edu/graphsage/#code>`_ dataset using GraphSAGE.
  * :github_nvflare_link:`Financial Transaction Classification <examples/advanced/gnn#federated-gnn-on-graph-dataset-using-inductive-learning>` - Example using GNNs for Financial Transaction Classification with `Elliptic++ <https://github.com/git-disl/EllipticPlusPlus>`_ dataset using GraphSAGE.

15. Financial Applications
==========================

  * :github_nvflare_link:`Financial Application with Federated XGBoost Methods <examples/advanced/finance>` Example using XGBoost in various ways to train a federated model to perform fraud detection with a finance dataset.


Setting up a virtual environment for examples and notebooks
===========================================================
It is recommended to set up a virtual environment before installing the dependencies for the examples. Install dependencies for a virtual environment with:

.. code-block:: bash

    python3 -m pip install --user --upgrade pip
    python3 -m pip install --user virtualenv


Once venv is installed, you can use it to create a virtual environment with:

.. code-block:: shell

    $ python3 -m venv nvflare_example

This will create the ``nvflare_example`` directory in current working directory if it doesn't exist,
and also create directories inside it containing a copy of the Python interpreter,
the standard library, and various supporting files.


Activate the virtualenv by running the following command:

.. code-block:: shell

    $ source nvflare_example/bin/activate

Installing required packages
----------------------------
In each example folder, install required packages for training:

.. code-block:: bash

    pip install --upgrade pip
    pip install -r requirements.txt

(optional) some examples contain scripts for plotting the TensorBoard event files, if needed, please also install the additional requirements in the example folder:

.. code-block:: bash

    pip install -r plot-requirements.txt


JupyterLab with your virtual environment for Notebooks
------------------------------------------------------
To run examples including notebooks, we recommend using `JupyterLab <https://jupyterlab.readthedocs.io>`_.

After activating your virtual environment, install JupyterLab:

.. code-block:: bash

  pip install jupyterlab

If you need to register the virtual environment you created so it is usable in JupyterLab, you can register the kernel with:

.. code-block:: bash

  python -m ipykernel install --user --name="nvflare_example"

Start a Jupyter Lab:

.. code-block:: bash

  jupyter lab .

When you open a notebook, select the kernel you registered, "nvflare_example", using the dropdown menu at the top right.

Custom Code in Example Apps
===========================
There are several ways to make :ref:`custom code <custom_code>` available to clients when using NVIDIA FLARE.
Most hello-* examples use a custom folder within the FL application.
Note that using a custom folder in the app needs to be :ref:`allowed <troubleshooting_byoc>` when using secure provisioning.
By default, this option is disabled in the secure mode. POC mode, however, will work with custom code by default.

In contrast, the :github_nvflare_link:`CIFAR-10 <examples/advanced/cifar10>`,
:github_nvflare_link:`prostate segmentation <examples/advanced/prostate>`,
and :github_nvflare_link:`BraTS18 segmentation <examples/advanced/brats18>` examples assume that the
learner code is already installed on the client's system and available in the PYTHONPATH.
Hence, the app folders do not include the custom code there.
The PYTHONPATH is set in the ``run_poc.sh`` or ``run_secure.sh`` scripts of the example.
Running these scripts as described in the README will make the learner code available to the clients.
