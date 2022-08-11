.. _hello_scatter_and_gather:

Hello Scatter and Gather
========================

Before You Start
----------------

Before jumping into this guide, make sure you have an environment with
`NVIDIA FLARE <https://pypi.org/project/nvflare/>`_ installed.

You can follow the :ref:`installation <installation>` guide on the general concept of setting up a
Python virtual environment (the recommended environment) and how to install NVIDIA FLARE.


Introduction
-------------

This tutorial is meant solely to demonstrate how the NVIDIA FLARE system works, without introducing any actual deep
learning concepts.

Through this exercise, you will learn how to use NVIDIA FLARE with numpy to perform basic
computations across two clients with the included Scatter and Gather workflow, which broadcasts the training tasks then
aggregates the results that come back.

Due to the simplified weights, you will be able to clearly see and understand
the results of the FL aggregation and the model persistor process.

The setup of this exercise consists of one **server** and two **clients**.
The server side model starting with weights ``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]``.

The following steps compose one cycle of weight updates, called a **round**:

 #. Clients are responsible for adding a delta to the weights to calculate new weights for the model.
 #. These updates are then sent to the server which will aggregate them to produce a model with new weights.
 #. Finally, the server sends this updated version of the model back to each client, so the clients can continue to calculate the next model weights in future rounds.

For this exercise, we will be working with the ``hello-numpy-sag`` application in the examples folder.
Custom FL applications can contain the folders:

 #. **custom**: contains any custom components (custom Python code)
 #. **config**: contains client and server configurations (``config_fed_client.json``, ``config_fed_server.json``)
 #. **resources**: contains the logger config (``log.config``)

Let's get started. First clone the repo, if you haven't already:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Remember to activate your NVIDIA FLARE Python virtual environment from the installation guide.
Ensure numpy is installed.

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install numpy

Now that you have all your dependencies installed, let's implement the federated learning system.


NVIDIA FLARE Client
-------------------
 
You will first notice that the ``hello-numpy-sag`` application does not contain a ``custom`` folder.

The code for the client and server components has been implemented in the
`nvflare/app-common/np <https://github.com/NVIDIA/NVFlare/tree/main/nvflare/app_common/np>`_ folder of the NVFlare code tree.

These files, for example the trainer in `np_trainer.py <https://github.com/NVIDIA/NVFlare/tree/main/nvflare/app_common/np/np_trainer.py>`_
can be copied into a ``custom`` folder in the ``hello-numpy-sag`` application as ``custom_trainer.py`` and modified to perform additional tasks.

The ``config_fed_client.json`` configuration discussed below would then be modified to point to this custom code by providing the custom path.

For example, replacing ``nvflare.app_common.np.np_trainer.NPTrainer`` with ``custom_trainer.NPTrainer``.

In the ``np_trainer.py`` trainer, we first import nvflare and numpy.
We then implement the ``execute`` function to enable the clients to perform
a simple addition of a diff to represent one calculation of training a round.

The server sends either the initial weights or any stored weights to each of the clients
through the ``Shareable`` object passed into ``execute()``.
Each client adds the diff to the model data after retrieving it from the DXO (see :ref:`data_exchange_object`)
obtained from the Shareable, and creates a new ``Shareable`` to include the new weights also contained
within a DXO.

In a real federated learning training scenario, each client does its training independently on its own dataset. 
As such, the weights returned to the server will likely be different for each of the clients. 

The FL server can ``aggregate`` (in this case average) the clients' results to produce the aggregated model.

You can learn more about ``Shareable`` and ``FLContext`` in the :ref:`programming guide <programming_guide>`.


NVIDIA FLARE Server & Application
---------------------------------

Model Persistor
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../nvflare/app_common/np/np_model_persistor.py
   :language: python
   :linenos:
   :caption: np_model_persistor.py

The model persistor is used to load and save models on the server.
Here, the model refer to weights packaged into a ``ModelLearnable`` object.

Internally, DXO is used to manage data after
:class:`FullModelShareableGenerator<nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator>`
converts Learnable to Shareable on the FL server.

The DXO helps all of the FL components agree on the format.

In this exercise, we can simply save the model as a binary ".npy" file.
Depending on the frameworks and tools, the methods of saving the model may vary.

Application Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside the config folder there are two files, ``config_fed_client.json`` and ``config_fed_server.json``.
For now, the default configurations are sufficient.


.. literalinclude:: ../../examples/hello-numpy-sag/config/config_fed_server.json
   :language: json
   :linenos:
   :caption: config_fed_server.json

.. literalinclude:: ../../examples/hello-numpy-sag/config/config_fed_client.json
   :language: json
   :linenos:
   :caption: config_fed_client.json


Here, in ``executors``, the Trainer implementation ``NPTrainer`` is configured for the task "train".

If you had implemented your own custom ``NPTrainer`` training routine,
for example in ``hello-numpy-sag/custom/custom_trainer.py``,
this config_fed_client.json configuration would be modified to point to this custom code by providing the custom path.

For example, replacing ``nvflare.app_common.np.np_trainer.NPTrainer`` with ``custom_trainer.NPTrainer``.


Federated Numpy with Scatter and Gather Workflow!
-------------------------------------------------

.. |ExampleApp| replace:: hello-numpy-sag
.. include:: run_fl_system.rst

After starting the server and clients, you should begin to see some outputs in each terminal
tracking the progress of the FL run.
If everything went as planned, you should see that through 10 rounds, the FL system has
aggregated new models on the server with the results produced by the clients.

.. include:: access_result.rst

.. include:: shutdown_fl_system.rst

Congratulations!

You've successfully built and run your first numpy federated learning system.

You now have a decent grasp of the main FL concepts, and are ready to start exploring how NVIDIA FLARE can be applied to many other tasks.

The full application for this exercise can be found in
`examples/hello-numpy-sag <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-numpy-sag>`_,
with the client and server components implemented in the `nvflare/app-common/np <https://github.com/NVIDIA/NVFlare/tree/main/nvflare/app_common/np>`_ folder of the NVFlare code tree.
