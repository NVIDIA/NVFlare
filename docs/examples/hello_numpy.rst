.. _hello_numpy:

Hello NumPy
===========

This example demonstrates how to use NVIDIA FLARE with NumPy to train a simple model using federated averaging (FedAvg). The complete example code can be found in the `hello-numpy directory <examples/hello-world/hello-numpy/>`_.
It is recommended to create a virtual environment and run everything within a virtualenv.

NVIDIA FLARE Installation
-------------------------
For the complete installation instructions, see `installation <../../installation.html>`_

.. code-block:: text

    pip install nvflare

Install the dependency

.. code-block:: text

    pip install -r requirements.txt


Code Structure
--------------

Get the example code from GitHub:

.. code-block:: text

    git clone https://github.com/NVIDIA/NVFlare.git

Navigate to the hello-numpy directory:

.. code-block:: text

    git switch <release branch>
    cd examples/hello-world/hello-numpy


.. code-block:: text

    hello-numpy
        |
        |-- client.py             # client local training script
        |-- model.py              # model definition
        |-- job.py                # job recipe that defines client and server configurations
        |-- requirements.txt      # dependencies


Data
-----------------
This example uses a simplified synthetic dataset. Each client performs basic operations on a 3x3 weight matrix, adding a small delta to each weight during training. This approach allows you to clearly observe the federated learning aggregation process without the complexity of real data loading and preprocessing.

In a real FL experiment, each client would have their own dataset used for their local training.
Here for simplicity's sake, we use synthetic data that allows you to clearly see and understand the federated learning aggregation process.

Model
------------------
The ``SimpleNumpyModel`` class implements a basic neural network model using NumPy arrays.
The model represents a simple 3x3 weight matrix that can be trained through federated learning. The model starts with fixed weights ``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`` and provides methods for training and evaluation.

The model implementation is located in ``model.py``.


.. literalinclude:: ../../../examples/hello-world/hello-numpy/model.py
    :language: python
    :linenos:
    :caption: model.py
    :lines: 14-



Client Code
------------------
Notice the training code follows the standard FL client pattern.
The only difference is that we added a few lines to receive and send data to the server.


.. literalinclude:: ../../../examples/hello-world/hello-numpy/client.py
    :language: python
    :linenos:
    :caption: Client Code (client.py)
    :lines: 14-


Server Code
------------------
In federated averaging, the server code is responsible for aggregating model updates from clients. The workflow pattern is similar to scatter-gather.
In this example, we will directly use the default federated averaging algorithm provided by NVFlare.
The FedAvg class is defined in `nvflare.app_common.workflows.fedavg.FedAvg`.
There is no need to define a customized server code for this example.


Job Recipe Code
------------------
Job Recipe contains the client.py and built-in fedavg algorithm.


.. literalinclude:: ../../../examples/hello-world/hello-numpy/job.py
    :language: python
    :linenos:
    :caption: job recipe (job.py)
    :lines: 14-


Run FL Job
-----------
This section provides the command to execute the federated learning job using the job recipe defined above. Run this command in your terminal.

.. note::

    The model starts with weights ``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`` and each client adds 1 to each weight during training.
    After aggregation, you should see the weights increase by 1 each round, demonstrating the federated learning process.



Command to execute the FL job
-----------------------------

Use the following command in your terminal to start the job with the specified number of rounds and number of clients.

.. code-block:: text

   python job.py --num_rounds 3 --n_clients 2

The full source code for this exercise can be found in
:github_nvflare_link:`examples/hello-world/hello-numpy <examples/hello-world/hello-numpy/>`.
