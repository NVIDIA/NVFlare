.. _hello-world_hello-cyclic:


`hello pytorch <../hello-pt/doc.html>`_ ||
`hello lightning <../hello-lightning/doc.html>`_ ||
`hello tensorflow <../hello-tf/doc.html>`_ ||
`hello LR <../hello-lr/doc.html>`_ ||
`hello KMeans <../hello-kmeans/doc.html>`_ ||
`hello KM <../hello-km/doc.html>`_ ||
`hello stats <../hello-stats/doc.html>`_ ||
**hello cyclic** ||
`hello-xgboost <../hello-xgboost/doc.html>`_ ||
`hello-flower <../hello-flower/doc.html>`_ ||


Hello Cyclic
===================

This example demonstrates how to use NVIDIA FLARE with **Tensorflow** to train an image classifier using
cyclic weight transfer approach.The complete example code can be found in the`hello-cyclic directory <examples/hello-world/hello-cyclic/>`_.
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
    cd examples/hello-world/hello-cyclic


.. code-block:: text

    hello-cyclic
        |
        |-- client.py         # client local training script
        |-- model.py          # model definition
        |-- job.py            # job recipe that defines client and server configurations
        |-- requirements.txt  # dependencies

Data
-----------------
This example uses the `MNIST dataset`

.. literalinclude:: ../../../examples/hello-world/hello-cyclic/client.py
    :language: python
    :linenos:
    :caption: dataset
    :lines: 36-56

Model
------------------
In TensorFlow, neural networks can be implemented using the Keras Sequential API. The model's architecture is defined by stacking layers sequentially. In this example, the model consists of a Flatten layer to convert the input into a 1D array, followed by a Dense layer with 128 units and ReLU activation, a Dropout layer for regularization, and a final Dense layer with 10 units for classification. The implementation of this model can be found in model.py.

.. literalinclude:: ../../../examples/hello-world/hello-cyclic/model.py
    :language: python
    :linenos:
    :caption: model.py
    :lines: 14-

--------------


Client Code
------------------

Notice the training code is almost identical to the pytorch standard training code.
The only difference is that we added a few lines to receive and send data to the server.

.. literalinclude:: ../../../examples/hello-world/hello-cyclic/client.py
    :language: python
    :linenos:
    :caption: client.py
    :lines: 14-


Server Code
------------------
In federated averaging, the server code is responsible for
aggregating model updates from clients, the workflow pattern is similar to scatter-gather.
In this example, we will directly use the default federated averaging algorithm provided by NVFlare.
The Cyclic class is defined in `nvflare.app_common.workflows.cyclic import Cyclic`
There is no need to define a customized server code for this example.


Job Recipe Code
------------------
The job recipe code is used to define the client and server configurations.

 .. literalinclude:: ../../../examples/hello-world/hello-cyclic/job.py
    :language: python
    :linenos:
    :caption: Job Recipe (job.py)
    :lines: 14-

Run FL Job
------------------

This section provides the command to execute the federated learning job
using the job recipe defined above. Run this command in your terminal.

**Command to execute the FL job**

Use the following command in your terminal to start the job with the specified
number of rounds, batch size, and number of clients.


.. code-block:: text

  python job.py

