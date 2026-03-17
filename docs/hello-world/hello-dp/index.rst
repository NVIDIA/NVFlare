Hello Differential Privacy
===========================

This example demonstrates how to use NVIDIA FLARE with PyTorch and **Differential Privacy (DP)** to train a fraud detection model using federated averaging (FedAvg) with privacy guarantees. The example uses `Opacus <https://opacus.ai>`_ to implement DP-SGD (Differentially Private Stochastic Gradient Descent) during local client training on each client. This achieves sample-level differential privacy. The complete example code can be found in the `hello-dp directory <examples/hello-world/hello-dp/>`_. It is recommended to create a virtual environment and run everything within a virtualenv.

What is Differential Privacy?
------------------------------

`Differential Privacy (DP) <https://en.wikipedia.org/wiki/Differential_privacy>`_ is a mathematical framework that provides strong privacy guarantees when handling sensitive data. In Federated Learning, DP protects user information by adding carefully calibrated noise to the model training process.

**DP-SGD** adds noise during each optimization step:

1. **Gradient Clipping**: Gradients are clipped to bound sensitivity
2. **Noise Addition**: Gaussian noise is added to clipped gradients
3. **Privacy Accounting**: Privacy budget (ε, δ) is tracked

The privacy-utility trade-off is controlled by epsilon (ε):

- **Lower ε** = Stronger privacy, more noise, lower accuracy
- **Higher ε** = Weaker privacy, less noise, higher accuracy

Typical values:

- **ε ≤ 1.0**: Strong privacy (recommended for sensitive data)
- **ε = 1.0-3.0**: Moderate privacy (good balance) - default is 1.0
- **ε > 10**: Weak privacy (minimal protection)

NVIDIA FLARE Installation
--------------------------

For complete installation instructions, see `Installation <https://nvflare.readthedocs.io/en/main/installation.html>`_.

.. code-block:: bash

   pip install nvflare

First get the example code from github:

.. code-block:: bash

   git clone https://github.com/NVIDIA/NVFlare.git

Then navigate to the hello-dp directory:

.. code-block:: bash

   git switch <release branch>
   cd examples/hello-world/hello-dp

Install the dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Code Structure
--------------

.. code-block:: bash

   hello-dp
   |
   |-- client.py             # client training script with DP-SGD using Opacus
   |-- model.py              # MLP model definition for tabular data
   |-- job.py                # job recipe that defines client and server configurations
   |-- requirements.txt      # dependencies

Data
----

This example uses the `Credit Card Fraud Detection dataset <https://www.openml.org/d/1597>`_ from OpenML - a binary classification problem to detect fraudulent credit card transactions.

**Dataset characteristics:**

- ~284,000 samples (Normal: 284,315, Fraud: 492)
- 29 features (anonymized transaction features V1-V28, Amount)
- 2 classes: Normal (0) and Fraud (1)
- **Highly imbalanced**: ~99.8% normal, ~0.17% fraud

**Important Note**: This dataset is extremely imbalanced with only 492 fraud cases out of 284,807 transactions. This presents additional challenges for training:

- Standard accuracy can be misleading (99.8% accuracy by always predicting "normal")
- **F1 Score and Precision/Recall** are more meaningful metrics for fraud detection
- The model must learn to detect the rare fraud class despite the imbalance

This is a **privacy-sensitive** use case - credit card transaction data requires strong privacy protection, making it ideal for demonstrating differential privacy in federated learning.

**Data Distribution**: In a real FL experiment, each client would have their own dataset. For this example, the dataset is **automatically partitioned across clients** using a simple split, so each client has a **non-overlapping subset** of the data. This simulates a basic federated scenario where data is distributed across multiple institutions.

Model
-----

The model is a simple Multi-Layer Perceptron (MLP) for binary classification. The implementation can be found in `model.py <model.py>`_.

.. code-block:: python

   import torch.nn as nn

   class TabularMLP(nn.Module):
       """Simple Multi-Layer Perceptron for tabular data classification"""
       
       def __init__(self, input_dim=29, hidden_dims=[64, 32], output_dim=2):
           super(TabularMLP, self).__init__()
           
           layers = []
           prev_dim = input_dim
           
           # Build hidden layers
           for hidden_dim in hidden_dims:
               layers.append(nn.Linear(prev_dim, hidden_dim))
               layers.append(nn.ReLU())
               layers.append(nn.Dropout(0.2))
               prev_dim = hidden_dim
           
           # Output layer
           layers.append(nn.Linear(prev_dim, output_dim))
           
           self.model = nn.Sequential(*layers)

The architecture:

- **Input layer**: 29 features (transaction data)
- **Hidden layers**: 64 → 32 neurons with ReLU activation and dropout
- **Output layer**: 2 neurons (normal vs fraud)

Client Code with Differential Privacy
--------------------------------------

The client code `client.py <client.py>`_ implements DP-SGD using **Opacus**. The key difference from standard training is adding the ``PrivacyEngine``:

.. code-block:: python

   from opacus import PrivacyEngine
   import nvflare.client as flare

   # Initialize NVFlare client
   flare.init()

   # Initialize privacy engine once (in first round only)
   privacy_engine = None

   while flare.is_running():
       input_model = flare.receive()
       model.load_state_dict(input_model.params)
       
       # === Apply Differential Privacy (First Round Only) ===
       # Privacy budget accumulates across ALL federated rounds
       if input_model.current_round == 0:
           # Calculate total epochs across all rounds for privacy accounting
           total_epochs = args.epochs * input_model.total_rounds
           
           privacy_engine = PrivacyEngine()
           model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
               module=model,
               optimizer=optimizer,
               data_loader=train_loader,
               epochs=total_epochs,                    # Total across ALL rounds
               target_epsilon=args.target_epsilon,     # Target privacy budget
               target_delta=args.target_delta,         # Failure probability
               max_grad_norm=args.max_grad_norm,       # Gradient clipping
           )
           # Noise multiplier is computed automatically
           print(f"Noise multiplier: {optimizer.noise_multiplier:.4f}")
       # ==================================
       
       # Train as usual - PrivacyEngine handles gradient clipping & noise
       for epoch in range(args.epochs):
           for data, target in train_loader:
               optimizer.zero_grad()
               loss = criterion(model(data), target)
               loss.backward()
               optimizer.step()
       
       # Check cumulative privacy budget spent
       epsilon = privacy_engine.get_epsilon(args.target_delta)
       print(f"Cumulative privacy spent: (ε = {epsilon:.2f}, δ = {args.target_delta})")

The ``PrivacyEngine.make_private_with_epsilon()`` method:

1. Wraps the model to enable per-sample gradient computation
2. Automatically computes the noise multiplier for target epsilon
3. Modifies the optimizer to clip gradients and add noise
4. Wraps the data loader for privacy accounting
5. Tracks privacy budget cumulatively across all federated rounds

Server-Side Workflow
--------------------

This example uses the `FedAvgRecipe <https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.recipes.fedavg.html>`_, which implements the `FedAvg <https://proceedings.mlr.press/v54/mcmahan17a>`_ algorithm. The Recipe API handles all server-side logic automatically:

1. Initialize the global model
2. For each training round:

   - Sample available clients
   - Send the global model to selected clients
   - Wait for client updates
   - Aggregate client models into a new global model

With the Recipe API, **there is no need to write custom server code**. The federated averaging workflow is provided by NVFlare.

Job Recipe Code
---------------

The ``FedAvgRecipe`` combines the client training script with DP parameters:

.. code-block:: python

   recipe = FedAvgRecipe(
       name="hello-dp",
       min_clients=n_clients,
       num_rounds=num_rounds,
       # Model can be class instance or dict config
       # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.pt"
       model=TabularMLP(input_dim=29, hidden_dims=[64, 32], output_dim=2),
       train_script="client.py",
       train_args=f"--batch_size {batch_size} --target_epsilon {target_epsilon} --n_clients {n_clients}",
   )

   env = SimEnv(num_clients=n_clients)
   recipe.execute(env=env)

**Important**: Privacy budget (ε) accumulates across ALL federated rounds. The ``target_epsilon`` parameter specifies the total privacy budget for the entire training process, not per round.

Run Job
-------

From terminal simply run the job script to execute the job in a simulation environment.

.. code-block:: bash

   python job.py

To customize parameters:

.. code-block:: bash

   python job.py --n_clients 2 --num_rounds 10 --target_epsilon 1.0

Parameters:

- ``--n_clients``: Number of federated clients (default: 2)
- ``--num_rounds``: Number of federated rounds (default: 10)
- ``--batch_size``: Training batch size (default: 64)
- ``--target_epsilon``: **Total** privacy budget across all rounds - **lower = stronger privacy** (default: 1.0)

.. note::
   As part of the job script, use ``add_experiment_tracking(recipe, tracking_type="tensorboard")`` to stream training metrics to the server using NVIDIA FLARE's `SummaryWriter <https://nvflare.readthedocs.io/en/main/apidocs/nvflare.client.tracking.html#nvflare.client.tracking.SummaryWriter>`_ in `client.py <client.py>`_.

Visualize Results
-----------------

View training metrics and privacy budget in TensorBoard:

.. code-block:: bash

   tensorboard --logdir /tmp/nvflare/simulation/hello-dp

Open http\://localhost:6006 to see:

- Training loss over time
- **Accuracy** and **F1 Score** (fraud detection metrics)
- Privacy epsilon spent per client

Privacy-Utility Trade-off
--------------------------

Differential Privacy involves a trade-off between privacy and model utility. The privacy budget (ε) accumulates across all federated rounds:

.. list-table:: Privacy Levels
   :widths: 15 20 20 45
   :header-rows: 1

   * - Epsilon (ε)
     - Privacy Level
     - Model Accuracy
     - Use Case
   * - ε ≤ 0.5
     - Very Strong
     - Lower
     - Highly sensitive (medical)
   * - ε = 0.5-1.0
     - Strong
     - Moderate
     - Sensitive (financial)
   * - ε = 1.0-3.0
     - Moderate
     - Good (default)
     - General private data
   * - ε = 3.0-10
     - Weak
     - Better
     - Lightly sensitive
   * - ε > 10
     - Minimal
     - Best
     - Not recommended for privacy

**Important Notes:**

- The epsilon value is **cumulative** across all federated rounds
- Lower epsilon = stronger privacy but may require more rounds or lower accuracy
- The noise multiplier is automatically computed to meet your target epsilon

**Recommendations:**

- Start with ``--target_epsilon 1.0`` (default) for a good privacy-utility balance
- For highly sensitive data (medical, financial), use ε ≤ 1.0
- Adjust ``max_grad_norm`` (gradient clipping) if needed
- Consider pre-training on public data before fine-tuning on private data
- Monitor cumulative epsilon across rounds

Output Summary
--------------

Initialization
~~~~~~~~~~~~~~

* **TensorBoard**: Logs available at /tmp/nvflare/simulation/hello-dp/server/simulate_job/tb_events
* **Workflow**: FedAvg controller initialized with DP-enabled clients
* **Privacy**: Privacy engine initialized in round 0, tracks cumulative budget

Each Round
~~~~~~~~~~

* **Model Distribution**: Global model sent to clients
* **Local Training**: Each client trains with DP-SGD using Opacus
* **Privacy Tracking**: Cumulative epsilon (ε) logged for each client
* **Aggregation**: DP-trained models aggregated on server

Completion
~~~~~~~~~~

* **Final Model**: Trained model with privacy guarantees
* **Privacy Budget**: Final cumulative privacy budget reported (should be ≤ target_epsilon)
* **Expected Performance** (with default ε=1.0, 5 rounds, 2 clients):

  - Global Test Accuracy: **99.95%**
  - Global Test F1 Score: **81.58%**
  - These results demonstrate effective fraud detection while maintaining strong privacy protection

References
----------

1. Abadi, M., et al. (2016). `Deep Learning with Differential Privacy <https://arxiv.org/abs/1607.00133>`_. ACM CCS 2016.
2. McMahan, B., et al. (2017). `Communication-Efficient Learning of Deep Networks from Decentralized Data <https://proceedings.mlr.press/v54/mcmahan17a>`_. AISTATS 2017.
3. `Opacus: User-friendly library for training PyTorch models with differential privacy <https://opacus.ai/>`_
4. `NVIDIA FLARE Documentation <https://nvflare.readthedocs.io/>`_
