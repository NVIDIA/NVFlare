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

1.1. Workflows
--------------

  * :ref:`Hello NumPy <hello_numpy>` - Example using the FedAvg workflow with a NumPy trainer
  * :ref:`Hello Cross-Site Validation <hello_cross_val>` - Example using the Cross Site Eval workflow, also demonstrates running cross site validation using the previous training results.
  * :github_nvflare_link:`Hello Cyclic Weight Transfer (GitHub) <examples/hello-world/hello-cyclic>` - Example using the CyclicController workflow to implement `Cyclic Weight Transfer <https://pubmed.ncbi.nlm.nih.gov/29617797/>`_ with TensorFlow as the deep learning training framework
  * :github_nvflare_link:`Swarm Learning <examples/advanced/swarm_learning>` - Example using Swarm Learning and Client-Controlled Cross-site Evaluation workflows.

1.2. Deep Learning
------------------

  * :ref:`Hello PyTorch <hello_pt_job_api>` - Example image classifier using FedAvg and PyTorch as the deep learning training framework
  * :ref:`Hello TensorFlow <hello_tf_job_api>` - Example image classifier using FedAvg and TensorFlow as the deep learning training frameworks


2. Tutorial Notebooks
=====================

  * :github_nvflare_link:`Intro to the FL Simulator <examples/tutorials/flare_simulator.ipynb>` - Shows how to use the :ref:`fl_simulator` to run a local simulation of an NVFLARE deployment to test and debug an application without provisioning a real FL project.
  * :github_nvflare_link:`Hello FLARE API <examples/tutorials/flare_api.ipynb>` - Goes through the different commands of the :ref:`flare_api` to show the syntax and usage of each.
  * :github_nvflare_link:`NVFLARE in POC Mode <examples/tutorials/setup_poc.ipynb>` - Shows how to use :ref:`POC mode <poc_command>` to test the features of a full FLARE deployment on a single machine.
  * :github_nvflare_link:`Job CLI Tutorial <examples/tutorials/job_cli.ipynb>` - Walks through the different commands of the Job CLI and showcases syntax and example usages.
  * :github_nvflare_link:`Job Recipe <examples/tutorials/job_recipe.ipynb>` - Introduces Job Recipes to simplify federated learning job creation and execution with a high-level API.
  * :github_nvflare_link:`FLARE Logging <examples/tutorials/logging.ipynb>` - Covers how to configure logging in FLARE for different use cases and modes.

3. Federated Learning Algorithms
================================

  * :github_nvflare_link:`Federated Learning with CIFAR-10 (GitHub) <examples/advanced/cifar10>` - Includes examples of using FedAvg, FedProx, FedOpt, SCAFFOLD, homomorphic encryption, and streaming of TensorBoard metrics to the server during training

  .. toctree::
    :maxdepth: 2

    examples/fl_algorithms

4. Privacy Preserving Algorithms
================================
Privacy preserving algorithms in NVIDIA FLARE are implemented as :ref:`filters <filters_for_privacy>` that can be applied as data is sent or received between peers.

  * :github_nvflare_link:`Federated Learning with CIFAR-10 (GitHub) <examples/advanced/cifar10>` - Includes examples of using FedAvg, FedProx, FedOpt, SCAFFOLD, homomorphic encryption, and streaming of TensorBoard metrics to the server during training

5. Traditional ML examples
==========================

  * :github_nvflare_link:`Federated Linear Model with Scikit-learn (GitHub) <examples/advanced/sklearn-linear>` - For an example of using NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_, a widely used open-source machine learning library that supports supervised and unsupervised learning.
  * :github_nvflare_link:`Federated K-Means Clustering with Scikit-learn (GitHub) <examples/advanced/sklearn-kmeans>` - NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_ and k-Means.
  * :github_nvflare_link:`Federated SVM with Scikit-learn (GitHub) <examples/advanced/sklearn-svm>` - NVIDIA FLARE with `scikit-learn <https://scikit-learn.org/>`_ and `SVM <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
  * :github_nvflare_link:`Federated XGBoost (GitHub) <examples/advanced/xgboost>` - Includes examples of histogram-based and tree-based algorithms. Tree-based algorithms also includes bagging and cyclic approaches. Also includes an example of vertical federated XGBoost.

6. Medical Image Analysis
=========================

  * :github_nvflare_link:`MONAI Integration (GitHub) <integration/monai>` - For an example of using NVIDIA FLARE to train a 3D medical image analysis model using federated averaging (FedAvg) and MONAI Bundle `MONAI <https://project-monai.github.io/>`_

7. Federated Statistics
=======================

  * :ref:`Federated Statistic Overview <federated_statistics>` - Discuss the overall federated statistics features.
  * :github_nvflare_link:`Federated Statistics for medical imaging (Github) <examples/advanced/federated-statistics/image_stats/README.md>` - Example of gathering local image histogram to compute the global dataset histograms.
  * :github_nvflare_link:`Federated Statistics for tabular data with DataFrame (Github) <examples/advanced/federated-statistics/df_stats/README.md>` - Example of gathering local statistics summary from Pandas DataFrame to compute the global dataset statistics.
  * :github_nvflare_link:`Federated Statistics with Monai Statistics integration for Spleen CT Image (Github) <integration/monai/examples/README.md>` - Example demonstrated Monai statistics integration and few other features in federated statistics
  
  .. toctree::
    :maxdepth: 1
    :hidden:

    examples/federated_statistics_overview

8. Federated Site Policies
==========================

  * :github_nvflare_link:`Federated Policies (Github) <examples/advanced/federated-policies/README.rst>` - Discuss the federated site policies for authorization, resource and data privacy management
  * :github_nvflare_link:`Custom Authentication (Github) <examples/advanced/custom_authentication/README.rst>` - Show the custom authentication policy and secure mode.
  * :github_nvflare_link:`Job-Level Authorization (Github) <examples/advanced/job-level-authorization/README.md>` - Show the job-level authorization policy and secure mode.
  * :github_nvflare_link:`KeyCloak Site Authentication Integration (Github) <examples/advanced/keycloak-site-authentication/README.md>` - Demonstrate KeyCloak integration for supporting site-specific authentication.

9. Experiment Tracking
=======================

  * :github_nvflare_link:`FL Experiment Tracking with TensorBoard Streaming <examples/advanced/experiment-tracking/tensorboard>` - :ref:`(documentation) <tensorboard_streaming>` - Example building on Hello PyTorch with TensorBoard streaming from clients to server
  * :github_nvflare_link:`FL Experiment Tracking with MLflow <examples/advanced/experiment-tracking/mlflow>` - :ref:`(documentation) <experiment_tracking_mlflow>`- Example integrating Hello PyTorch with MLflow with streaming from clients to server
  * :github_nvflare_link:`FL Experiment Tracking with Weights and Biases <examples/advanced/experiment-tracking/wandb>` - Example integrating Hello PyTorch with Weights and Biases streaming capability from clients to server.

  .. toctree::
    :maxdepth: 1
    :hidden:

    examples/tensorboard_streaming
    examples/fl_experiment_tracking_mlflow

10.  Natural Language Processing (NLP)
======================================

  * :github_nvflare_link:`NLP-NER (Github) <examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-8_federated_LLM_training/08.1_fed_bert/federated_nlp_with_bert.ipynb>` - Illustrates both `BERT <https://github.com/google-research/bert>`_ and `GPT-2 <https://github.com/openai/gpt-2>`_ models from `Hugging Face <https://huggingface.co/>`_ (`BERT-base-uncased <https://huggingface.co/bert-base-uncased>`_, `GPT-2 <https://huggingface.co/gpt2>`__) on a Named Entity Recognition (NER) task using the `NCBI disease dataset <https://pubmed.ncbi.nlm.nih.gov/24393765/>`_.

11. Federated Large Language Model (LLM)
========================================

  * :github_nvflare_link:`Parameter Efficient Fine Turning <integration/nemo/examples/peft>` - Example utilizing NeMo's PEFT methods to adapt a LLM to a downstream task.
  * :github_nvflare_link:`Prompt-Tuning Example <integration/nemo/examples/prompt_learning>` - Example for using FLARE with NeMo for prompt learning.
  * :github_nvflare_link:`Supervised Fine Tuning (SFT) <integration/nemo/examples/supervised_fine_tuning>` - Example to fine-tune all parameters of a LLM on supervised data.
  * :github_nvflare_link:`LLM Tuning via HuggingFace SFT Trainer <examples/advanced/llm_hf>` - Example for using FLARE with a HuggingFace trainer for LLM tuning tasks.


12. Graph Neural Network (GNN)
==============================

  * :github_nvflare_link:`Protein Classification <examples/advanced/gnn>` - Example using GNNs for Protein Classification using `PPI <http://snap.stanford.edu/graphsage/#code>`_ dataset using GraphSAGE.
  * :github_nvflare_link:`Financial Transaction Classification <examples/advanced/gnn>` - Example using GNNs for Financial Transaction Classification with `Elliptic++ <https://github.com/git-disl/EllipticPlusPlus>`_ dataset using GraphSAGE.

13. Financial Applications
==========================

  * :github_nvflare_link:`Financial Application with Federated XGBoost Methods <examples/advanced/finance>` Example using XGBoost in various ways to train a federated model to perform fraud detection with a finance dataset.
  * :github_nvflare_link:`Financial Transaction Classification <examples/advanced/gnn>` - Example using GNNs for Financial Transaction Classification with `Elliptic++ <https://github.com/git-disl/EllipticPlusPlus>`_ dataset using GraphSAGE.


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

In contrast, the :github_nvflare_link:`CIFAR-10 <examples/advanced/cifar10>` example assumes that the
learner code is already installed on the client's system and available in the PYTHONPATH.
Hence, the app folders do not include the custom code there.
The PYTHONPATH is set in the ``run_poc.sh`` or ``run_secure.sh`` scripts of the example.
Running these scripts as described in the README will make the learner code available to the clients.
