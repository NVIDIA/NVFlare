.. _quickstart:

###################
Quick Start Series
###################

Welcome to the NVIDIA FLARE Quick Start Series! This guide provides a set of hello-world examples to help you quickly learn how to build federated learning programs using NVIDIA FLARE.

Make sure you have completed the :ref:`installation` steps before proceeding.

Run Modes
=========

FLARE supports three modes for different stages of your workflow:

- **Simulator** (:ref:`fl_simulator`) -- Runs jobs on a single system for fast testing and algorithm development.
- **POC** (:ref:`poc_command`) -- Simulates deployment on one host with separate processes for clients and server.
- **Production** (:ref:`provisioned_setup`) -- Distributed deployment using startup kits from provisioning.

Start with the **Simulator** for development, then validate with **POC** before going to **Production**.

.. seealso::

    For a complete list of ready-to-use federated learning recipes with code examples, see :ref:`available_recipes`.

Hello-world Examples
====================

The following hello-world examples demonstrate different federated learning algorithms and workflows. Each example includes instructions and code to help you get started.

1. **Hello PyTorch** - Federated averaging with PyTorch models and training loops. :doc:`hello-world/hello-pt/index`

2. **Hello Lightning** - Example using PyTorch Lightning for streamlined model training. :doc:`hello-world/hello-lightning/index`

3. **Hello Differential Privacy** - `Federated learning with differential privacy using PyTorch and Opacus for privacy-preserving training. <hello-world/hello-dp/index.html>`_

4. **Hello TensorFlow** - `Federated averaging using TensorFlow models. <hello-world/hello-tf/index.html>`_

5. **Hello Logistic Regression** - `Federated logistic regression example using scikit-learn. <hello-world/hello-lr/index.html>`_

6. **Hello Cyclic** - `Cyclic federated learning workflow example. <hello-world/hello-cyclic/index.html>`_

7. **Hello Tabular Statistics** - `Federated statistics computation example. <hello-world/hello-tabular-stats/index.html>`_

8. **Hello Flower** - `Running Flower apps in FLARE. <hello-world/hello-flower/index.html>`_

9. **Hello XGBoost** - `Federated XGBoost example demonstrating gradient boosting for tabular data in a federated setting. <hello-world/hello-xgboost/index.html>`_

Let's start with Hello PyTorch: :doc:`hello-world/hello-pt/index`

.. toctree::
   :maxdepth: 1
   :hidden:

   hello-world/hello-pt/index
   hello-world/hello-tf/index
   hello-world/hello-lightning/index
   hello-world/hello-xgboost/index
   hello-world/hello-dp/index
   hello-world/hello-flower/index
   hello-world/hello-lr/index
   hello-world/hello-tabular-stats/index
   hello-world/hello-cyclic/index