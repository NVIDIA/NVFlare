.. _hello_pt:

Hello PyTorch
=============

Before You Start
----------------

Feel free to refer to the :doc:`detailed documentation <../programming_guide>` at any point
to learn more about the specifics of `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_.

Make sure you have an environment with NVIDIA FLARE installed.

You can follow the :ref:`installation <installation>` guide on the general concept of setting up a
Python virtual environment (the recommended environment) and how to install NVIDIA FLARE.


Introduction
-------------

Through this exercise, you will integrate NVIDIA FLARE with the popular
deep learning framework `PyTorch <https://pytorch.org/>`_ and learn how to use NVIDIA FLARE to train a convolutional
network with the CIFAR10 dataset using the included Scatter and Gather workflow.

The setup of this exercise consists of one **server** and two **clients**.

The following steps compose one cycle of weight updates, called a **round**:

 #. Clients are responsible for generating individual weight-updates for the model using their own CIFAR10 dataset. 
 #. These updates are then sent to the server which will aggregate them to produce a model with new weights. 
 #. Finally, the server sends this updated version of the model back to each client.

For this exercise, we will be working with the ``hello-pt`` application in the examples folder. 
Custom FL applications can contain the folders:

 #. **custom**: contains the custom components (``simple_network.py``, ``cifar10trainer.py``)
 #. **config**: contains client and server configurations (``config_fed_client.json``, ``config_fed_server.json``)
 #. **resources**: contains the logger config (``log.config``)

Now that you have a rough idea of what is going on, let's get started. First clone the repo:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Now remember to activate your NVIDIA FLARE Python virtual environment from the installation guide.

Since you will use PyTorch and torchvision for this exercise, let's go ahead and install both libraries: 

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install torch torchvision


.. note::

  There is a pending fix related to Pillow, PyTorch==1.9 and Numpy.  If you see exception related to
  ``enumerate(self.train_loader)``, downgrade your Pillow to 8.2.0.

  .. code-block:: shell
  
    (nvflare-env) $ python3 -m pip install torch torchvision Pillow==8.2.0

If you would like to go ahead and run the exercise now, you can skip directly to :ref:`hands-on`.

NVIDIA FLARE Client
-------------------

Neural Network
^^^^^^^^^^^^^^^

With all the required dependencies installed, you are ready to run a Federated Learning
with two clients and one server. The training procedure and network 
architecture are modified from 
`Training a Classifier <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_.


Let's see what an extremely simplified CIFAR10 training looks like:

.. literalinclude:: ../../examples/hello-pt/custom/simple_network.py
   :language: python
   :lines: 15-
   :lineno-start: 15
   :linenos:
   :caption: simple_network.py

This ``SimpleNetwork`` class is your convolutional neural network to train with the CIFAR10 dataset.
This is not related to NVIDIA FLARE, so we implement it in a file called ``simple_network.py``.

Dataset & Setup
^^^^^^^^^^^^^^^^

Now implement the custom class ``Cifar10Trainer`` as an NVIDIA FLARE Executor in a file
called ``cifar10trainer.py``.

In a real FL experiment, each client would have their own dataset used for their local training.
For simplicity's sake, you can download the same CIFAR10 dataset from the Internet via torchvision's datasets module.
Additionally, you need to set up the optimizer, loss function and transform to process the data.
You can think of all of this code as part of your local training loop, as every deep learning training has a similar setup.

Since you will encapsulate every training-related step in the ``Cifar10Trainer`` class,
let's put this preparation stage into the ``__init__`` method:

.. literalinclude:: ../../examples/hello-pt/custom/cifar10trainer.py
   :language: python
   :lines: 37-82
   :lineno-start: 37
   :linenos:


Local Train
^^^^^^^^^^^

Now that you have your network and dataset setup, in the ``Cifar10Trainer`` class.
Let's also implement a local training loop in a method called ``local_train``:

.. literalinclude:: ../../examples/hello-pt/custom/cifar10trainer.py
   :language: python
   :pyobject: Cifar10Trainer.local_train


.. note::

  Everything up to this point is completely independent of NVIDIA FLARE. It is just purely a PyTorch
  deep learning exercise.  You will now build the NVIDIA FLARE application based on this PyTorch code.


Integrate NVIDIA FLARE with Local Train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NVIDIA FLARE makes it easy to integrate your local train code into the NVIDIA FLARE API.

The simplest way to do this is to subclass the ``Executor`` class and
implement one method ``execute``, which is called every time the client receives
an updated model from the server with the task "train" (the server will broadcast the "train" task in the Scatter and
Gather workflow we will configure below).
We can then call our local train inside the ``execute`` method.

.. note::

  The ``execute`` method inside the ``Executor`` class is where all of the client side computation occurs.
  In these exercises, we update the weights by training on a local dataset, however, it is important to remember that NVIDIA FLARE is not restricted to just deep learning.
  The type of data passed between the server and the clients, and the computations that the clients perform can be anything, as long as all of the FL Components agree on the same format.

Take a look at the following code:

.. literalinclude:: ../../examples/hello-pt/custom/cifar10trainer.py
   :language: python
   :pyobject: Cifar10Trainer.execute

The concept of ``Shareable`` is described in :ref:`shareable <shareable>`.
Essentially, every NVIDIA FLARE client receives the model weights from the server in ``shareable`` format.
It is then passed into the ``execute`` method, and returns a new ``shareable`` back to the server.
The data is managed by using DXO (see :ref:`data_exchange_object` for details).

Thus, the first thing is to retrieve the model weights delivered by server via ``shareable``, and this can be seen in
the first part of the code block above before ``local_train`` is called.

We then perform a local train so the client's model is trained with its own dataset.

After finishing the local train, the train method builds a new ``shareable`` with newly-trained weights
and metadata and returns it back to the NVIDIA FLARE server for aggregation.

There is additional logic to handle the "submit_model" task, but that is for the CrossSiteModelEval workflow,
so we will be addressing that in a later example.

FLContext
^^^^^^^^^

The ``FLContext`` is used to set and retrieve FL related information among the FL components via ``set_prop()`` and
``get_prop()`` as well as get services provided by the underlying infrastructure. You can find more details in the
:ref:`documentation <fl_context>`.

NVIDIA FLARE Server & Application
---------------------------------

In this exercise, you can use the default settings, which leverage NVIDIA FLARE built-in components for NVIDIA FLARE server.

These built-in components are commonly used in most deep learning scenarios.

However, you are encouraged to build your own components to fully customize NVIDIA FLARE to meet your environment,
 which we will demonstrate in the following exercises.


Application Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside the config folder there are two files, ``config_fed_client.json`` and ``config_fed_server.json``.

.. literalinclude:: ../../examples/hello-pt/config/config_fed_client.json
   :language: json
   :linenos:
   :caption: config_fed_client.json

Take a look at line 8.

This is the ``Cifar10Trainer`` you just implemented.

The NVIDIA FLARE client loads this application configuration and picks your implementation.

You can easily change it to another class so your NVIDIA FLARE client has different training logic.

The tasks "train" and "submit_model" have been configured to work with the ``Cifar10Trainer`` Executor.
The "validate" task for ``Cifar10Validator`` and the "submit_model" task are used for the ``CrossSiteModelEval`` workflow,
so we will be addressing that in a later example.


.. literalinclude:: ../../examples/hello-pt/config/config_fed_server.json
   :language: json
   :linenos:
   :caption: config_fed_server.json

The server application configuration, like said before, leverages NVIDIA FLARE built-in components.
Remember, you are encouraged to change them to your own classes whenever you have different application logic.

Note that on line 12, ``persistor`` points to ``PTFileModelPersistor``.
NVIDIA FLARE provides a built-in PyTorch implementation for a model persistor,
however for other frameworks/libraries, you will have to implement your own.

The Scatter and Gather workflow is implemented by :class:`ScatterAndGather<nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather>`
and is configured to make use of the components with id "aggregator", "persistor", and "shareable_generator".
The workflow code is all open source now, so feel free to study and use it as inspiration
to write your own workflows to support your needs.

.. _hands-on:

Train the Model, Federated!
---------------------------

.. |ExampleApp| replace:: hello-pt
.. include:: run_fl_system.rst

.. include:: access_result.rst

.. include:: shutdown_fl_system.rst

Congratulations!
You've successfully built and run your first federated learning system.

The full source code for this exercise can be found in
`examples/hello-pt <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-pt/>`_.
