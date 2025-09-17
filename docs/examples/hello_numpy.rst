.. _hello_numpy:

Hello NumPy
===========

This example demonstrates how to use NVIDIA FLARE with NumPy to train a simple model using federated averaging (FedAvg).
The complete example code can be found in the :github_nvflare_link:`hello-numpy directory <examples/hello-world/hello-numpy/>`.

Before You Start
----------------

Feel free to refer to the :doc:`detailed documentation <../programming_guide>` at any point
to learn more about the specifics of `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_.

Make sure you have an environment with NVIDIA FLARE installed.

You can follow :ref:`getting_started` on the general concept of setting up a
Python virtual environment (the recommended environment) and how to install NVIDIA FLARE.

Introduction
-------------

This tutorial is meant solely to demonstrate how the NVIDIA FLARE system works, without introducing any actual deep
learning concepts.

Through this exercise, you will learn how to use NVIDIA FLARE with NumPy to perform basic
computations across two clients with the included :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>` workflow,
which sends the model to the clients then aggregates the results that come back.

Due to the simplified weights, you will be able to clearly see and understand
the results of the FL aggregation and the model persistor process.

The setup of this exercise consists of one **server** and two **clients**.
The model is set to the starting weights ``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]``.

The following steps compose one cycle of weight updates, called a **round**:

 #. Clients are responsible for adding a delta to the weights to calculate new weights for the model.
 #. These updates are then sent to the server which will aggregate them to produce a model with new weights.
 #. Finally, the server sends this updated version of the model back to each client, so the clients can continue to calculate the next model weights in future rounds.

Code Structure
--------------

The example consists of three main files:

- ``job.py`` - Defines the federated learning job using the Recipe API
- ``client.py`` - Client-side training script that runs on each client
- ``model.py`` - Simple NumPy model definition

Data
----

This example uses a simplified synthetic dataset. Each client performs basic operations on a 3x3 weight matrix, adding a small delta to each weight during training. This approach allows you to clearly observe the federated learning aggregation process without the complexity of real data loading and preprocessing.

Model
-----

The ``SimpleNumpyModel`` class implements a basic neural network model using NumPy arrays:

.. literalinclude:: ../../examples/hello-world/hello-numpy/model.py
   :language: python
   :linenos:
   :caption: model.py

This model represents a simple 3x3 weight matrix that can be trained through federated learning. The model starts with fixed weights ``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`` and provides methods for training and evaluation.

Client Side
-----------

The training script ``client.py`` is the main script that will be run on the clients. It contains the NumPy specific
logic for training:

.. literalinclude:: ../../examples/hello-world/hello-numpy/client.py
   :language: python
   :linenos:
   :caption: client.py

The code above uses the three essential methods of the NVFlare's Client API:

   - `init()`: Initializes NVFlare Client API environment.
   - `receive()`: Receives model from the FL server.
   - `send()`: Sends the model to the FL server.

Server Side
-----------

In this example, the server runs :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>` with the default settings.

The server automatically handles:
- Model persistence using NumPy-compatible persistor
- Model aggregation using federated averaging
- Client selection and task distribution

Job Recipe
----------

The ``job.py`` script uses the modern Job Recipe API to define the federated learning job:

.. literalinclude:: ../../examples/hello-world/hello-numpy/job.py
   :language: python
   :linenos:
   :caption: job.py

Run Job
-------

To run this example:

1. Clone the repository and navigate to the example directory:

.. code-block:: shell

   $ git clone https://github.com/NVIDIA/NVFlare.git
   $ cd NVFlare/examples/hello-world/hello-numpy

2. Install the required dependencies:

.. code-block:: shell

   $ pip install -r requirements.txt

3. Run the example:

.. code-block:: shell

   $ python job.py

The script will create an NVFlare job and run it using the FL Simulator.

Understanding the Results
-------------------------

The model starts with weights ``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`` and each client adds 1 to each weight during training.
After aggregation, you should see the weights increase by 1 each round, demonstrating the federated learning process.

The evaluation metric ``weight_mean`` tracks the mean of all weights, which will increase from 5.0 to 6.0 to 7.0, etc., as training progresses.

You can monitor the training progress through TensorBoard logs available at:
``/tmp/nvflare/simulation/hello-numpy/server/simulate_job/tb_events``

The full source code for this exercise can be found in
:github_nvflare_link:`examples/hello-world/hello-numpy <examples/hello-world/hello-numpy/>`.

Previous Versions of Hello NumPy
--------------------------------

This example consolidates the previous `hello-numpy-sag` and `hello-fedavg-numpy` examples:

   - `hello-numpy-sag for 2.0-2.4 <https://github.com/NVIDIA/NVFlare/tree/2.0/examples/hello-numpy-sag>`_ (legacy scatter-and-gather approach)
   - `hello-fedavg-numpy for 2.5+ <https://github.com/NVIDIA/NVFlare/tree/2.5/examples/hello-world/hello-fedavg-numpy>`_ (job API approach)

The current version uses the modern job recipe API for a cleaner, more maintainable structure that follows the same pattern as other hello-world examples.
