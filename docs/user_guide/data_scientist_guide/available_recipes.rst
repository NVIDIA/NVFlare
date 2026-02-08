.. _available_recipes:

##################
Available Recipes
##################

NVFlare provides a variety of pre-built recipes for common federated learning algorithms and workflows.
Recipes are high-level, declarative APIs that simplify job configuration and execution.

.. contents:: Table of Contents
   :local:
   :depth: 2

Common Recipe Parameters
========================

Most training recipes accept the following model-related parameters:

``model``
    The model to use for federated training. Accepts:

    * **Class instance**: e.g., ``MyModel()`` - convenient and Pythonic
    * **Dict config**: e.g., ``{"class_path": "module.MyModel", "args": {"param": value}}`` - better for large models

    .. note::
       Class instances are converted to configuration files before job submission. For large models,
       use dict config to avoid unnecessary instantiation overhead.

``initial_ckpt``
    Absolute path to a pre-trained checkpoint file. The file may not exist locally but must exist
    on the server when the model is loaded during job execution.

    * PyTorch: Requires ``model`` for architecture (checkpoint has weights only)
    * TensorFlow/Keras: Can use ``initial_ckpt`` alone (Keras saves full model)

See :ref:`job_recipe` for detailed explanations of these options.

Federated Averaging (FedAvg)
============================

The most fundamental federated learning algorithm that aggregates model updates from multiple clients
by computing a weighted average.

PyTorch FedAvg
--------------

.. code-block:: python

    from nvflare.app_opt.pt.recipes import FedAvgRecipe
    from nvflare.recipe import SimEnv

    recipe = FedAvgRecipe(
        name="fedavg-pt",
        min_clients=2,
        num_rounds=5,
        model=MyModel(),
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/hello-world/hello-pt <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-pt>`_
- `examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedavg <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedavg>`_

TensorFlow FedAvg
-----------------

.. code-block:: python

    from nvflare.app_opt.tf.recipes import FedAvgRecipe
    from nvflare.recipe import SimEnv

    recipe = FedAvgRecipe(
        name="fedavg-tf",
        min_clients=2,
        num_rounds=5,
        model=MyTFModel(),
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/hello-world/hello-tf <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-tf>`_
- `examples/advanced/cifar10/tf/cifar10_fedavg <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/tf/cifar10_fedavg>`_

NumPy FedAvg
------------

For framework-agnostic or NumPy-based models.

.. code-block:: python

    from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
    from nvflare.recipe import SimEnv

    recipe = NumpyFedAvgRecipe(
        name="fedavg-numpy",
        min_clients=2,
        num_rounds=5,
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/hello-world/hello-numpy <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-numpy>`_

Sklearn FedAvg
--------------

For scikit-learn based models.

.. code-block:: python

    from nvflare.app_opt.sklearn.recipes import SklearnFedAvgRecipe
    from nvflare.recipe import SimEnv

    recipe = SklearnFedAvgRecipe(
        name="fedavg-sklearn",
        min_clients=2,
        num_rounds=5,
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/sklearn-linear <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/sklearn-linear>`_

FedAvg with Homomorphic Encryption
----------------------------------

FedAvg with secure aggregation using homomorphic encryption.

.. code-block:: python

    from nvflare.app_opt.pt.recipes import FedAvgRecipeWithHE
    from nvflare.recipe import SimEnv

    recipe = FedAvgRecipeWithHE(
        name="fedavg-he",
        min_clients=2,
        num_rounds=5,
        model=MyModel(),
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/kaplan-meier-he <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/kaplan-meier-he>`_


FedProx
=======

FedProx is FedAvg with a proximal term added to the client loss function to handle data heterogeneity.
It uses the standard FedAvgRecipe with the FedProx loss helper on the client side.

PyTorch FedProx
---------------

.. code-block:: python

    from nvflare.app_opt.pt.recipes import FedAvgRecipe
    from nvflare.recipe import SimEnv

    # FedProx uses FedAvgRecipe with FedProxLoss in the client training script
    recipe = FedAvgRecipe(
        name="fedprox-pt",
        min_clients=2,
        num_rounds=5,
        model=MyModel(),
        train_script="client.py",
        train_args="--fedproxloss_mu 0.01",  # Pass mu parameter to client
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

In your client training script, use the FedProxLoss helper:

.. code-block:: python

    from nvflare.app_opt.pt import PTFedProxLoss

    # In training loop:
    fedprox_loss = PTFedProxLoss(mu=fedproxloss_mu)
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        ce_loss = criterion(output, target)
        # Add FedProx regularization term
        prox_loss = fedprox_loss(model)
        loss = ce_loss + prox_loss
        loss.backward()
        optimizer.step()

**Examples:**

- `examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedprox <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedprox>`_

TensorFlow FedProx
------------------

.. code-block:: python

    from nvflare.app_opt.tf.recipes import FedAvgRecipe
    from nvflare.recipe import SimEnv

    recipe = FedAvgRecipe(
        name="fedprox-tf",
        min_clients=2,
        num_rounds=5,
        model=MyTFModel(),
        train_script="client.py",
        train_args="--fedproxloss_mu 0.01",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

In your client training script, use the TensorFlow FedProxLoss:

.. code-block:: python

    from nvflare.app_opt.tf.fedprox_loss import TFFedProxLoss

    fedprox_loss = TFFedProxLoss(mu=fedproxloss_mu)
    # Use in training loop

**Examples:**

- `examples/advanced/cifar10/tf/cifar10_fedprox <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/tf/cifar10_fedprox>`_


FedOpt (Federated Optimization)
===============================

Federated optimization with server-side optimizer (e.g., SGD, Adam).

PyTorch FedOpt
--------------

.. code-block:: python

    from nvflare.app_opt.pt.recipes import FedOptRecipe
    from nvflare.recipe import SimEnv

    recipe = FedOptRecipe(
        name="fedopt-pt",
        min_clients=2,
        num_rounds=5,
        model=MyModel(),
        train_script="client.py",
        optimizer_args={"path": "torch.optim.SGD", "args": {"lr": 1.0, "momentum": 0.6}},
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedopt <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedopt>`_

TensorFlow FedOpt
-----------------

.. code-block:: python

    from nvflare.app_opt.tf.recipes import FedOptRecipe
    from nvflare.recipe import SimEnv

    recipe = FedOptRecipe(
        name="fedopt-tf",
        min_clients=2,
        num_rounds=5,
        model=MyTFModel(),
        train_script="client.py",
        optimizer_args={"path": "tensorflow.keras.optimizers.SGD", "args": {"learning_rate": 1.0}},
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/cifar10/tf/cifar10_fedopt <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/tf/cifar10_fedopt>`_


SCAFFOLD
========

SCAFFOLD algorithm for handling data heterogeneity with control variates.

PyTorch SCAFFOLD
----------------

.. code-block:: python

    from nvflare.app_opt.pt.recipes import ScaffoldRecipe
    from nvflare.recipe import SimEnv

    recipe = ScaffoldRecipe(
        name="scaffold-pt",
        min_clients=2,
        num_rounds=5,
        model=MyModel(),
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/cifar10/pt/cifar10-sim/cifar10_scaffold <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/pt/cifar10-sim/cifar10_scaffold>`_

TensorFlow SCAFFOLD
-------------------

.. code-block:: python

    from nvflare.app_opt.tf.recipes import ScaffoldRecipe
    from nvflare.recipe import SimEnv

    recipe = ScaffoldRecipe(
        name="scaffold-tf",
        min_clients=2,
        num_rounds=5,
        model=MyTFModel(),
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/cifar10/tf/cifar10_scaffold <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/tf/cifar10_scaffold>`_


Cyclic Learning
===============

Sequential training across clients in a cyclic order.

PyTorch Cyclic
--------------

.. code-block:: python

    from nvflare.app_opt.pt.recipes import CyclicRecipe
    from nvflare.recipe import SimEnv

    recipe = CyclicRecipe(
        name="cyclic-pt",
        min_clients=2,
        num_rounds=5,
        model=MyModel(),
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/hello-world/hello-cyclic <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-cyclic>`_

TensorFlow Cyclic
-----------------

.. code-block:: python

    from nvflare.app_opt.tf.recipes import CyclicRecipe
    from nvflare.recipe import SimEnv

    recipe = CyclicRecipe(
        name="cyclic-tf",
        min_clients=2,
        num_rounds=5,
        model=MyTFModel(),
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)


XGBoost Recipes
===============

Federated XGBoost for tree-based models.

XGBoost Horizontal (Histogram-based)
------------------------------------

Histogram-based federated XGBoost for horizontal data partitioning.

.. code-block:: python

    from nvflare.app_opt.xgboost.recipes import XGBHorizontalRecipe
    from nvflare.recipe import SimEnv

    recipe = XGBHorizontalRecipe(
        name="xgb-horizontal",
        min_clients=2,
        num_rounds=10,
        xgb_params={"max_depth": 6, "eta": 0.1, "objective": "binary:logistic"},
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/xgboost/fedxgb <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost/fedxgb>`_

XGBoost Bagging (Tree-based)
----------------------------

Tree-based federated XGBoost using bagging.

.. code-block:: python

    from nvflare.app_opt.xgboost.recipes import XGBBaggingRecipe
    from nvflare.recipe import SimEnv

    recipe = XGBBaggingRecipe(
        name="xgb-bagging",
        min_clients=2,
        training_mode="bagging",
        num_rounds=10,
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/xgboost/fedxgb (job_tree.py) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost/fedxgb>`_

XGBoost Vertical
----------------

Federated XGBoost for vertical data partitioning.

.. code-block:: python

    from nvflare.app_opt.xgboost.recipes import XGBVerticalRecipe
    from nvflare.recipe import SimEnv

    recipe = XGBVerticalRecipe(
        name="xgb-vertical",
        min_clients=2,
        num_rounds=10,
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/xgboost/fedxgb (job_vertical.py) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost/fedxgb>`_
- `examples/advanced/xgboost/fedxgb_secure (job_vertical.py) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost/fedxgb_secure>`_


Sklearn Specialized Recipes
===========================

K-Means FedAvg
--------------

Federated K-Means clustering.

.. code-block:: python

    from nvflare.app_opt.sklearn.recipes import KMeansFedAvgRecipe
    from nvflare.recipe import SimEnv

    recipe = KMeansFedAvgRecipe(
        name="kmeans",
        min_clients=2,
        num_rounds=5,
        n_clusters=3,
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/sklearn-kmeans <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/sklearn-kmeans>`_

SVM FedAvg
----------

Federated Support Vector Machine.

.. code-block:: python

    from nvflare.app_opt.sklearn.recipes import SVMFedAvgRecipe
    from nvflare.recipe import SimEnv

    recipe = SVMFedAvgRecipe(
        name="svm",
        min_clients=2,
        num_rounds=5,
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/sklearn-svm <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/sklearn-svm>`_

Logistic Regression FedAvg
--------------------------

Federated Logistic Regression.

.. code-block:: python

    from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe
    from nvflare.recipe import SimEnv

    recipe = FedAvgLrRecipe(
        name="lr",
        min_clients=2,
        num_rounds=5,
        train_script="client.py",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/hello-world/hello-lr <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-lr>`_


Federated Statistics
====================

Compute federated statistics across distributed data.

.. code-block:: python

    from nvflare.recipe import SimEnv
    from nvflare.recipe.fedstats import FedStatsRecipe

    recipe = FedStatsRecipe(
        name="stats",
        stats_output_path="./output",
        sites=["site-1", "site-2"],
        statistic_configs={"count": {}, "mean": {}, "stddev": {}},
        stats_generator=my_stats_generator,
    )
    env = SimEnv(clients=["site-1", "site-2"])
    run = recipe.execute(env)

**Examples:**

- `examples/hello-world/hello-tabular-stats <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-tabular-stats>`_
- `examples/advanced/federated-statistics/df_stats <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/federated-statistics/df_stats>`_
- `examples/advanced/federated-statistics/image_stats <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/federated-statistics/image_stats>`_


Federated Evaluation
====================

Evaluate a pre-trained model across multiple sites.

PyTorch FedEval
---------------

Evaluate a pre-trained PyTorch model by sending it to all clients for evaluation on their local data.

.. code-block:: python

    from nvflare.app_opt.pt.recipes.fedeval import FedEvalRecipe
    from nvflare.recipe import SimEnv

    recipe = FedEvalRecipe(
        name="eval_job",
        model=MyModel(),
        eval_ckpt="/path/to/pretrained_model.pt",
        min_clients=2,
        eval_script="client.py",
        eval_args="--batch_size 32",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

.. note::
   ``eval_ckpt`` is **required**. It can be either:

   * an absolute path on the server to the pre-trained checkpoint (.pt, .pth), or
   * a relative or absolute path to a local checkpoint file that will be bundled with the job
     (for example, via utilities such as ``prepare_initial_ckpt``).

   When specifying an absolute server-side path, the checkpoint file may not exist locally when
   building the job.
**Examples:**

- `examples/hello-world/hello-lightning-eval <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-lightning-eval>`_


Cross-Site Evaluation
=====================

Evaluate models across all client sites (compare each client's model against all datasets).

.. code-block:: python

    from nvflare.app_common.np.recipes import NumpyCrossSiteEvalRecipe
    from nvflare.recipe import SimEnv

    recipe = NumpyCrossSiteEvalRecipe(
        name="cross-eval",
        min_clients=2,
        eval_script="evaluate.py",
        eval_args="--data_root /path/to/data",
        initial_ckpt="/path/to/pretrained_model.npy",  # Optional: evaluate specific model
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

.. note::
   - Use ``eval_script`` to specify custom evaluation logic. If not provided, uses a built-in
     dummy validator (for testing only).
   - Use ``initial_ckpt`` to evaluate a specific pre-trained model. If not provided, the recipe
     evaluates models from the training run directory.

**Examples:**

- `examples/hello-world/hello-numpy-cross-val <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-numpy-cross-val>`_


Private Set Intersection (PSI)
==============================

Compute intersection of private sets across clients.

.. code-block:: python

    from nvflare.app_common.psi.recipes import DhPSIRecipe
    from nvflare.recipe import SimEnv

    recipe = DhPSIRecipe(
        name="psi",
        min_clients=2,
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/advanced/psi/user_email_match <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/psi/user_email_match>`_


Flower Integration
==================

Run Flower-based federated learning jobs.

.. code-block:: python

    from nvflare.app_opt.flower.recipe import FlowerRecipe
    from nvflare.recipe import SimEnv

    recipe = FlowerRecipe(
        name="flower-job",
        min_clients=2,
        flower_app="path/to/flower/app",
    )
    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

**Examples:**

- `examples/hello-world/hello-flower <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/hello-flower>`_


Swarm Learning
==============

Decentralized federated learning without a central server.

.. code-block:: python

    from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe
    from nvflare.recipe import SimEnv

    recipe = SimpleSwarmLearningRecipe(
        name="swarm",
        model=MyModel(),
        num_rounds=5,
        train_script="client.py",
        initial_ckpt="/path/to/pretrained.pt",  # Optional: pre-trained weights
    )
    env = SimEnv(num_clients=3)
    run = recipe.execute(env)

.. note::
   ``SimpleSwarmLearningRecipe`` is also available from the original location for backward compatibility:
   ``from nvflare.app_common.ccwf.recipes.swarm import SimpleSwarmLearningRecipe``


Edge Recipes
============

Recipes for edge device federated learning.

EdgeFedBuffRecipe
-----------------

.. code-block:: python

    from nvflare.edge.tools.edge_fed_buff_recipe import (
        EdgeFedBuffRecipe,
        ModelManagerConfig,
        DeviceManagerConfig,
    )

    recipe = EdgeFedBuffRecipe(
        job_name="edge-fedavg",
        model=MyModel(),
        model_manager_config=ModelManagerConfig(max_num_active_model_versions=3, max_model_version=20),
        device_manager_config=DeviceManagerConfig(device_selection_size=100),
        initial_ckpt="/path/to/pretrained.pt",  # Optional: pre-trained weights
    )

**Examples:**

- `examples/advanced/edge/jobs <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/edge/jobs>`_


Utility Functions
=================

Add Experiment Tracking
-----------------------

Add experiment tracking (MLflow, TensorBoard, W&B) to any recipe.

.. code-block:: python

    from nvflare.recipe.utils import add_experiment_tracking

    add_experiment_tracking(recipe, tracking_type="tensorboard")
    # or
    add_experiment_tracking(recipe, tracking_type="mlflow")
    # or
    add_experiment_tracking(recipe, tracking_type="wandb")

Add Cross-Site Evaluation
-------------------------

Add cross-site evaluation to any training recipe.

.. code-block:: python

    from nvflare.recipe.utils import add_cross_site_evaluation

    add_cross_site_evaluation(recipe)


Execution Environments
======================

Recipes can be executed in different environments:

SimEnv (Simulation)
-------------------

Run locally for development and testing.

.. code-block:: python

    from nvflare.recipe import SimEnv

    env = SimEnv(num_clients=2)
    run = recipe.execute(env)

PocEnv (Proof of Concept)
-------------------------

Run with multiple processes on a single machine.

.. code-block:: python

    from nvflare.recipe import PocEnv

    env = PocEnv(num_clients=2)
    run = recipe.execute(env)

ProdEnv (Production)
--------------------

Deploy to production NVFlare infrastructure.

.. code-block:: python

    from nvflare.recipe import ProdEnv

    env = ProdEnv(startup_kit_location="/path/to/startup_kit")
    run = recipe.execute(env)

