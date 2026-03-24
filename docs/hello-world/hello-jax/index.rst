Hello JAX
=========

This example demonstrates how to use NVIDIA FLARE with JAX, Flax, and Optax to train an MNIST classifier using federated averaging (FedAvg). It follows the same hello-world recipe structure as ``hello-pt``, but uses a JAX client training loop and a flattened parameter vector for transport.

Install NVFLARE and Dependencies
--------------------------------

For the complete installation instructions, see `Installation <https://nvflare.readthedocs.io/en/main/installation.html>`_.

.. code-block:: bash

   pip install nvflare

First get the example code from GitHub:

.. code-block:: bash

   git clone https://github.com/NVIDIA/NVFlare.git

Then navigate to the hello-jax directory:

.. code-block:: bash

   git switch <release branch>
   cd examples/hello-world/hello-jax

Install the dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Code Structure
--------------

.. code-block:: text

   hello-jax
   |
   |-- client.py         # client local training script
   |-- model.py          # JAX/Flax model helpers
   |-- prepare_data.py   # helper that downloads MNIST and writes .npy files
   |-- prepare_model.py  # helper that writes the initial flattened checkpoint
   |-- job.py            # job recipe that defines client and server configurations
   |-- requirements.txt  # dependencies

Data
----

This example uses the `MNIST <https://www.tensorflow.org/datasets/catalog/mnist>`_ dataset. The job script downloads the raw MNIST files once before the simulator starts and converts them into ``.npy`` files. Each client then loads from that prepared cache.

Model
-----

The model in :github_nvflare_link:`model.py <examples/hello-world/hello-jax/model.py>` is a small convolutional neural network implemented with Flax.

.. literalinclude:: ../../../examples/hello-world/hello-jax/model.py
    :language: python
    :linenos:
    :caption: model code (model.py)
    :lines: 14-

Client Code
-----------

The client code (:github_nvflare_link:`client.py <examples/hello-world/hello-jax/client.py>`) keeps the local training loop in JAX while using NVFlare's client API to receive the current global model and return the updated parameters.

.. literalinclude:: ../../../examples/hello-world/hello-jax/client.py
    :language: python
    :linenos:
    :caption: client code (client.py)
    :lines: 14-

Server Code
-----------

This example uses the base ``FedAvgRecipe`` configured for NumPy parameter exchange. The JAX parameter tree is flattened into a single NumPy vector before it is exchanged with the server, then reconstructed on the client before each training round.

The job script prepares two resources before the simulator starts:

- The initial flattened checkpoint is prepared by a small helper subprocess so the main job launcher does not import JAX before NVFlare forks/spawns its runtime processes.
- The shared MNIST ``.npy`` cache is prepared once up front so both simulated clients do not try to download the dataset at the same time or rely on TensorFlow-only data utilities.

Job Recipe Code
---------------

.. literalinclude:: ../../../examples/hello-world/hello-jax/job.py
    :language: python
    :linenos:
    :caption: job recipe (job.py)
    :lines: 14-

Run Job
-------

From terminal simply run the job script to execute the job in a simulation environment.

.. code-block:: bash

   python job.py

You can adjust the main hyperparameters from the command line as needed:

.. code-block:: bash

   python job.py --n_clients 2 --num_rounds 3 --epochs 1 --batch_size 128

Output Summary
--------------

- **Workflow**: The simulator sends the flattened global parameter vector to each site, each site trains locally in JAX, and FedAvg aggregates the returned updates.
