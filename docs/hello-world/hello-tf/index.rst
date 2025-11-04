Hello TensorFlow
================

This example demonstrates how to use `NVIDIA FLARE <https://nvflare.readthedocs.io/en/main/index.html>`_ with TensorFlow to train an image classifier using federated averaging (`FedAvg <https://arxiv.org/abs/1602.05629>`_). TensorFlow serves as the deep learning training framework in this example.

For detailed documentation, see the `Hello TensorFlow <https://www.tensorflow.org/datasets/catalog/mnist>`_ example page.

We recommend using the `NVIDIA TensorFlow docker <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow>`_ for GPU support. If GPU is not required, a Python virtual environment is sufficient.

To run this example with the FLARE API, refer to the `hello_world notebook <../hello_world.ipynb>`_.

Run NVIDIA TensorFlow Container
-------------------------------

Ensure the `NVIDIA container toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ is installed. Then execute the following command:

.. code-block:: bash

   docker run --gpus=all -it --rm -v [path_to_NVFlare]:/NVFlare nvcr.io/nvidia/tensorflow:xx.xx-tf2-py3

NVIDIA FLARE Installation
-------------------------

For complete installation instructions, visit `Installation <https://nvflare.readthedocs.io/en/main/installation.html>`_.

.. code-block:: bash

   pip install nvflare

clone the example code from GitHub:

.. code-block:: bash

   git clone https://github.com/NVIDIA/NVFlare.git

Navigate to the hello-tf directory:

.. code-block:: bash

   git switch <release branch>
   cd examples/hello-world/hello-tf

Install the dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Code Structure
--------------

.. code-block:: text

   hello-pt
   |
   |-- client.py         # client local training script
   |-- model.py          # model definition
   |-- job.py            # job recipe that defines client and server configurations
   |-- requirements.txt  # dependencies

Data
----

This example uses the `MNIST <https://www.tensorflow.org/datasets/catalog/mnist>`_ handwritten digits dataset, which is loaded within the trainer code.

Model
-----

The `model.py` file defines a simple neural network using TensorFlow's Keras API. The `Net` model is a sequential architecture designed for image classification, featuring:

- **Flatten Layer**: Prepares input data for dense layers.
- **Dense Layer**: 128 units with ReLU activation for non-linearity.
- **Dropout Layer**: 20% dropout rate to mitigate overfitting.
- **Output Layer**: 10 units for classifying MNIST digits.

This model is used in federated learning with NVIDIA FLARE, trained across clients using the FedAvg algorithm.


.. literalinclude:: ../../../examples/hello-world/hello-tf/model.py
    :language: python
    :linenos:
    :caption: model code (model.py)
    :lines: 14-
 

Client Code
-----------

The client code ``client.py`` is responsible for training. The training code closely resembles standard PyTorch training code, with additional lines to handle data exchange with the server.

.. literalinclude:: ../../../examples/hello-world/hello-tf/client.py
    :language: python
    :linenos:
    :caption: client code (client.py)
    :lines: 14-


Server Code
-----------

In federated averaging, the server code aggregates model updates from clients, following a scatter-gather workflow pattern. This example uses the default federated averaging algorithm provided by NVFlare, eliminating the need for custom server code.

Job Recipe Code
---------------

The job recipe includes `client.py` and the built-in FedAvg algorithm.


.. literalinclude:: ../../../examples/hello-world/hello-tf/job.py
    :language: python
    :linenos:
    :caption: job recipe (job.py)
    :lines: 14-

Run the Experiment
------------------

Execute the script using the job API to create the job and run it with the simulator:

.. code-block:: bash

   TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async python3 job.py

Access the Logs and Results
---------------------------

Find the running logs and results inside the simulator's workspace:

.. code-block:: bash

   $ ls /tmp/nvflare/jobs/workdir

Notes on Running with GPUs
--------------------------

When using GPUs, TensorFlow attempts to allocate all available GPU memory at startup. To prevent this in multi-client scenarios, set the following flags:

.. code-block:: bash

   TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async

If you have more GPUs than clients, consider running one client per GPU using the `--gpu` argument during simulation, e.g., `nvflare simulator -n 2 --gpu 0,1 [job]`.
