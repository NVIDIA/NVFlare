Hello Cyclic Weight Transfer
============================

`Cyclic Weight Transfer <https://pubmed.ncbi.nlm.nih.gov/29617797/>`_ (CWT) is an alternative to `FedAvg <https://arxiv.org/abs/1602.05629>`_. CWT uses the `Cyclic Controller <https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic.html>`_ to pass the model weights from one site to the next for repeated fine-tuning.

.. note::

   This example uses the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ handwritten digits dataset and will load its data within the trainer code.

Running Tensorflow with GPU
---------------------------

We recommend using `NVIDIA TensorFlow docker <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow>`_ if you want to use GPU.
If you don't need to run using GPU, you can just use python virtual environment.

Run NVIDIA TensorFlow container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please install the `NVIDIA container toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ first.
Then run the following command:

.. code-block:: bash

   docker run --gpus=all -it --rm -v [path_to_NVFlare]:/NVFlare nvcr.io/nvidia/tensorflow:xx.xx-tf2-py3

Notes on running with GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you choose to run the example using GPUs, it is important to note that,
by default, TensorFlow will attempt to allocate all available GPU memory at the start.
In scenarios where multiple clients are involved, you have to prevent TensorFlow from allocating all GPU memory
by setting the following flags.

.. code-block:: bash

   TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async

Install NVFlare
---------------

For the complete installation instructions, see `Installation <https://nvflare.readthedocs.io/en/main/installation.html>`_

.. code-block:: text

   pip install nvflare

Get the example code from GitHub:

.. code-block:: bash

   git clone https://github.com/NVIDIA/NVFlare.git
   git switch <release branch>
   cd examples/hello-world/hello-cyclic


Install the dependency

.. code-block:: text

   pip install -r requirements.txt

Code Structure
--------------

Code structure:

.. code-block:: text

   hello-cyclic
   |
   |-- client.py           # client local training script
   |-- model.py            # model definition
   |-- job.py              # job recipe that defines client and server configurations
   |-- prepare_data.sh     # scripts to download the data
   |-- requirements.txt    # dependencies

Data
----

In this example, we will use the MNIST datasets, which is provided by
TensorFlow Keras API.

Model
-----


The model.py file defines a simple neural network using TensorFlow’s Keras API. The Net model is a sequential architecture designed for image classification, featuring:

- Flatten Layer: Prepares input data for dense layers.
- Dense Layer: 128 units with ReLU activation for non-linearity.
- Dropout Layer: 20% dropout rate to mitigate overfitting.
- Output Layer: 10 units for classifying MNIST digits.


.. literalinclude:: ../../../examples/hello-world/hello-cyclic/model.py
    :language: python
    :linenos:
    :caption: Model (model.py)
    :lines: 14-


Client Code
-----------

The client code ``client.py`` is responsible for training. Notice the training code is almost identical to the PyTorch standard training code.
The only difference is that we added a few lines to receive and send data to the server.


.. literalinclude:: ../../../examples/hello-world/hello-cyclic/client.py
    :language: python
    :linenos:
    :caption: Client Code (client.py)
    :lines: 14-

Server Code
-----------

In cyclic transfer, the server code is responsible for replaying model updates from one client to another. We will directly use
the default federated cyclic algorithm provided by NVFlare.

Job Recipe
----------


.. literalinclude:: ../../../examples/hello-world/hello-cyclic/job.py
    :language: python
    :linenos:
    :caption: job recipe (job.py)
    :lines: 14-


Run the Experiment
------------------

Prepare the data first:

.. code-block:: bash

   bash ./prepare_data.sh
   python job.py

Access the Logs and Results
---------------------------

You can find the running logs and results inside the simulator's workspace:

.. code-block:: bash

   $ ls "/tmp/nvflare/simulation/cyclic"
