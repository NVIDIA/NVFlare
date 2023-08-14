.. _hello_tf2:

Hello TensorFlow 2
==================

Before You Start
----------------

We recommend you first finish either the :doc:`hello_pt` or the :doc:`hello_scatter_and_gather` exercise.

Those guides go more in depth in explaining the federated learning aspect of `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_.

Here we assume you have already installed NVIDIA FLARE inside a python virtual environment
and have already cloned the repo.

Introduction
-------------

Through this exercise, you will integrate NVIDIA FLARE with the popular deep learning framework
`TensorFlow 2 <https://www.tensorflow.org/>`_ and learn how to use NVIDIA FLARE to train a convolutional
network with the MNIST dataset using the Scatter and Gather workflow.
You will also be introduced to some new components and concepts, including filters, aggregators, and event handlers.

The setup of this exercise consists of one **server** and two **clients**.

The following steps compose one cycle of weight updates, called a **round**:

 #. Clients are responsible for generating individual weight-updates for the model using their own MNIST dataset. 
 #. These updates are then sent to the server which will aggregate them to produce a model with new weights. 
 #. Finally, the server sends this updated version of the model back to each client.

For this exercise, we will be working with the ``hello-tf2`` application in the examples folder. 
Custom FL applications can contain the folders:

 #. **custom**: contains the custom components (``tf2_net.py``, ``trainer.py``, ``filter.py``, ``tf2_model_persistor.py``)
 #. **config**: contains client and server configurations (``config_fed_client.json``, ``config_fed_server.json``)
 #. **resources**: contains the logger config (``log.config``)

Let's get started.
Since this task is using TensorFlow, let's go ahead and install the library inside our virtual environment:

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install tensorflow


NVIDIA FLARE Client
-------------------

Neural Network
^^^^^^^^^^^^^^^

With all the required dependencies installed, you are ready to run a Federated Learning system
with two clients and one server.

Before you start, let's see what a simplified MNIST network looks like.

.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/custom/tf2_net.py
   :language: python
   :lines: 15-
   :lineno-start: 15
   :linenos:
   :caption: tf2_net.py

This ``Net`` class is the convolutional neural network to train with MNIST dataset.
This is not related to NVIDIA FLARE, so implement it in a file called ``tf2_net.py``.

Dataset & Setup
^^^^^^^^^^^^^^^^

Now you have to implement the class ``Trainer``, which is a subclass of ``Executor`` in NVIDIA FLARE,
in a file called ``trainer.py``.

Before you can really start a training, you need to set up your dataset.
In this exercise, you can download it from the Internet via ``tf.keras``'s datasets module,
and split it in half to create a separate dataset for each client.
Additionally, you must setup the optimizer, loss function and transform to process the data.

Since every step will be encapsulated in the ``SimpleTrainer`` class,
let's put this preparation stage into one method ``setup``:

.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/custom/trainer.py
   :language: python
   :lines: 41-71
   :lineno-start: 41
   :linenos:


How can you ensure this setup method is called before the client receives the model from the server?

The Trainer class is also a :ref:`FLComponent <fl_component>`, which always receives ``Event`` whenever
NVIDIA FLARE enters or leaves a certain stage.

In this case, there is an ``Event`` called ``EventType.START_RUN`` which perfectly matches these requirements. 
Because our trainer is a subclass of ``FLComponent``, you can implement the handler to handle the event and call the setup method:

.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/custom/trainer.py
   :language: python
   :lines: 37-39
   :lineno-start: 37
   :linenos:

.. note::

  This is a new concept you haven't learned in previous two exercises.

  The concepts of ``event`` and ``handler`` are very powerful because you are free to
  add your logic so it can run at different time and process various events.

  The entire list of events fired by NVIDIA FLARE is shown at :ref:`Event types <event_system>`.


You have everything you need, now let's implement the last method called ``execute``, which is
called every time the client receives an updated model from the server with the Task we will configure.


Link NVIDIA FLARE with Local Train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take a look at the following code:

.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/custom/trainer.py
   :language: python
   :pyobject: SimpleTrainer.execute

Every NVIDIA FLARE client receives the model weights from the server in the :ref:`shareable <shareable>`.
This application uses the ``exclude_var`` filter, so make sure to replace the missing layer with weights from the clients' previous training round:

.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/custom/trainer.py
   :language: python
   :lines: 111-115
   :lineno-start: 111
   :linenos:

Now update the local model with those received weights:

.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/custom/trainer.py
   :language: python
   :lines: 118
   :lineno-start: 118
   :linenos:

Then perform a simple :code:`self.model.fit` so the client's model is trained with its own dataset:

.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/custom/trainer.py
   :language: python
   :lines: 122-127
   :lineno-start: 122
   :linenos:
  
After finishing the local train, the train method uses the newly-trained weights to build a new ``DXO`` to update the
``Shareable`` with and then returns it back to the NVIDIA FLARE server.


NVIDIA FLARE Server & Application
---------------------------------

Filter
^^^^^^^ 

:ref:`filter <filters>` can be used for additional data processing in the ``Shareable``, for both
inbound and outbound data from the client and/or server.

For this exercise, we use a basic ``exclude_var`` filter to exclude the variable/layer ``flatten`` from the task result
as it goes outbound from the client to the server. The excluded layer is replaced with all zeros of the same shape,
which reduces compression size and ensures that the clients' weights for this variable are not shared with the server.

.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/custom/filter.py
   :language: python
   :lines: 15-
   :lineno-start: 15
   :linenos:
   :caption: filter.py

The filtering procedure occurs in the one required method, process, which receives and returns a shareable.
The parameters for what is excluded and the inbound/outbound option are all set in ``config_fed_client.json``
(shown later below) and passed in through the constructor.


Model Aggregator
^^^^^^^^^^^^^^^^

The :ref:`model aggregator <aggregator>` is used by the server to aggregate the clients' models into one model
within the Scatter and Gather workflow.

In this exercise, we perform a simple average over the two clients' weights with the
:class:`InTimeAccumulateWeightedAggregator<nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator>`
and configure for it to be used in ``config_fed_server.json`` (shown later below).

Model Persistor
^^^^^^^^^^^^^^^

The model persistor is used to load and save models on the server.

.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/custom/tf2_model_persistor.py
   :language: python
   :lines: 15-
   :lineno-start: 15
   :linenos:
   :caption: tf2_model_persistor.py

In this exercise, we simply serialize the model weights dictionary using pickle and
save it to a log directory calculated in initialize.
The file is saved on the FL server and the weights file name is defined in ``config_fed_server.json``.
Depending on the frameworks and tools, the methods of saving the model may vary.

FLContext is used throughout these functions to provide various useful FL-related information.
You can find more details in the :ref:`documentation <fl_context>`.

Application Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, inside the config folder there are two files, ``config_fed_client.json`` and ``config_fed_server.json``.

.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/config/config_fed_server.json
   :language: json
   :linenos:
   :caption: config_fed_server.json


Note how the :class:`ScatterAndGather<nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather>` workflow is
configured to use the included ``aggregator`` :class:`InTimeAccumulateWeightedAggregator<nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator>`
and ``shareable_generator`` :class:`FullModelShareableGenerator<nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator>`.
The ``persistor`` is configured to use ``TF2ModelPersistor`` in the custom directory of this hello_tf2 app with full
Python module paths.


.. literalinclude:: ../../examples/hello-world/hello-tf2/jobs/hello-tf2/app/config/config_fed_client.json
   :language: json
   :linenos:
   :caption: config_fed_client.json


Here, ``executors`` is configured with the Trainer implementation ``SimpleTrainer``.
Also, we set up ``filter.ExcludeVars`` as a ``task_result_filters`` and pass in ``["flatten"]`` as the argument.
Both of these are configured for the only Task that will be broadcast in the Scatter and Gather workflow, "train".

Train the Model, Federated!
---------------------------

.. |ExampleApp| replace:: hello-tf2
.. include:: run_fl_system.rst

.. include:: access_result.rst

.. include:: shutdown_fl_system.rst

Congratulations!

You've successfully built and run a federated learning system using TensorFlow 2.

The full source code for this exercise can be found in
`examples/hello-tf2 <https://github.com/NVIDIA/NVFlare/tree/2.3/examples/hello-world/hello-tf2>`_.

Previous Versions of Hello TensorFlow 2
---------------------------------------

   - `hello-tf2 for 2.0 <https://github.com/NVIDIA/NVFlare/tree/2.0/examples/hello-tf2>`_
   - `hello-tf2 for 2.1 <https://github.com/NVIDIA/NVFlare/tree/2.1/examples/hello-tf2>`_
   - `hello-tf2 for 2.2 <https://github.com/NVIDIA/NVFlare/tree/2.2/examples/hello-tf2>`_
