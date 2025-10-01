.. _hello_pt:

Hello Pytorch
=============

This example demonstrates how to use NVIDIA FLARE with PyTorch to train an image classifier using
federated averaging (FedAvg).The complete example code can be found in the`hello-pt directory <examples/hello-world/hello-pt/>_`.
It is recommended to create a virtual environment and run everything within a virtualenv.

NVIDIA FLARE Installation
-------------------------
for the complete installation instructions, see `installation <../../installation.html>`_

.. code-block:: text

    pip install nvflare

Install the dependency

.. code-block:: text

    pip install -r requirements.txt


Code Structure
--------------

first get the example code from github:

.. code-block:: text

    git clone https://github.com/NVIDIA/NVFlare.git

then navigate to the hello-pt directory:

.. code-block:: text

    git switch <release branch>
    cd examples/hello-world/hello-pt


.. code-block:: text

    hello-pt
        |
        |-- client.py             # client local training script
        |-- client_with_eval.py   # alternative client local training script with evaluation
        |-- model.py              # model definition
        |-- job.py                # job recipe that defines client and server configurations
        |-- requirements.txt      # dependencies


Data
-----------------
This example uses the `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset

In a real FL experiment, each client would have their own dataset used for their local training.
You can download the CIFAR10 dataset from the Internet via torchvision's datasets module,
You can split the datasets for different clients, so that each client has its own dataset.
Here for simplicity's sake, the same dataset we will be using on each client.

Model
------------------
In PyTorch, neural networks are implemented by defining a class that extends`nn.Module`. The network’s architecture is
set up in the `__init__` method,# while the `forward` method determines how input data flows through the layers.
For faster computations, the model is transferred to a hardware accelerator (such as CUDA GPUs) if available;
otherwise, it runs on the CPU. The implementation of this model can be found in model.py.

The model implementation is located in ``model.py``.


.. literalinclude:: ../../../examples/hello-world/hello-pt/model.py
    :language: python
    :linenos:
    :caption: model.py
    :lines: 14-



Client Code
------------------
Notice the training code is almost identical to the pytorch standard training code.
The only difference is that we added a few lines to receive and send data to the server.


.. literalinclude:: ../../../examples/hello-world/hello-pt/client.py
    :language: python
    :linenos:
    :caption: Client Code (client.py)
    :lines: 14-

or if you prefer both training and evaluation
.. literalinclude:: ../../../examples/hello-world/hello-pt/client_with_eval.py
    :language: python
    :linenos:
    :caption: Client Code (client_with_eval.py)
    :lines: 14-


Server Code
------------------
In federated averaging, the server code is responsible for
aggregating model updates from clients, the workflow pattern is similar to scatter-gather.
In this example, we will directly use the default federated averaging algorithm provided by NVFlare.
The FedAvg class is defined in `nvflare.app_common.workflows.fedavg.FedAvg`
There is no need to defined a customized server code for this example.


Job Recipe Code
------------------
Job Recipe contains the client.py and built-in fedavg algorithm.


.. literalinclude:: ../../../examples/hello-world/hello-pt/job.py
    :language: python
    :linenos:
    :caption: job recipe (job.py)
    :lines: 14-


Run FL Job
-----------
This section provides the command to execute the federated learning job
using the job recipe defined above. Run this command in your terminal.

.. note::

    depends on the number of clients, you might run into error due to several client try to download the data at the same time.
    suggest to pre-download the data to avoid such errors.



Command to execute the FL job
------------------
Use the following command in your terminal to start the job with the specified
number of rounds, batch size, and number of clients.

.. code-block:: text

   python job.py --num_rounds 2 --batch_size 16