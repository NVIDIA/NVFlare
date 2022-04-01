Quickstart (Numpy)
===================

Before You Start
----------------

Before jumping into this QuickStart guide, make sure you have an environment with `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_
installed. You can follow :doc:`installation <../installation>` on the general concept of setting up a Python virtual
environment (the recommended environment) and how to install NVIDIA FLARE.


Introduction
-------------

This tutorial is meant to solely demonstrate how the NVIDIA FLARE system works, without introducing any actual deep
learning concepts. Through this exercise, you will learn how to use NVIDIA FLARE with numpy to perform basic
computations across two clients with the included Scatter and Gather workflow, which broadcasts the training tasks then
aggregates the results that come back. Due to the simplified weights, you will be able to clearly see and understand
the results of the FL aggregation and the model persistor process.

The design of this exercise consists of one **server** and two **clients** starting with weights ``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]``.
The following steps compose one cycle of weight updates, called a **round**:

 #. Clients are responsible for adding a delta to the weights to calculate new weights for the model.
 #. These updates are then sent to the server which will aggregate them to produce a model with new weights.
 #. Finally, the server sends this updated version of the model back to each client, so the clients can continue to calculate the next model weights in future rounds.

For this exercise, we will be working with the ``hello-numpy-sag`` application in the examples folder.
Custom FL applications can contain the folders:

 #. **custom**: contains the custom components (``np_trainer.py``, ``np_model_persistor.py``)
 #. **config**: contains client and server configurations (``config_fed_client.json``, ``config_fed_server.json``)
 #. **resources**: contains the logger config (``log.config``)

Let's get started. First clone the repo, if you haven't already:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Remember to activate your NVIDIA FLARE Python virtual environment from the installation guide. Ensure numpy is installed.

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install numpy

Now that you have all your dependencies installed, let's implement the federated learning system.


NVIDIA FLARE Client
-------------------
 
In a file called ``np_trainer.py``, import nvflare and numpy.
Now we will implement the ``execute`` function to enable the clients to perform
a simple addition of a diff to represent one calculation of training a round.

Find the full code of ``np_trainer.py`` at
``examples/hello-numpy-sag/custom/np_trainer.py`` to follow along.

The server sends either the initial weights or any stored weights to each of the clients
through the ``Shareable`` object passed into ``execute()``. Each client adds the
diff to the model data after retrieving it from the DXO (see :ref:`data_exchange_object`)
obtained from the Shareable, and creates a new ``Shareable`` to include the new weights also contained
within a DXO.

In a real federated learning training scenario, each client does its training independently on its own dataset. 
As such, the weights returned to the server will likely be different for each of the clients. 

The FL server can ``aggregrate`` (in this case average) the clients' results to produce the aggregated model.

You can learn more about ``Shareable`` and ``FLContext`` in the :ref:`documentation <programming_guide>`.


NVIDIA FLARE Server & Application
---------------------------------

Model Persistor
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../examples/hello-numpy-sag/custom/np_model_persistor.py
   :language: python
   :linenos:
   :caption: np_model_persistor.py

The model persistor is used to load and save models on the server. Here, the model is weights packaged into a ``ModelLearnable`` object.

Internally, DXO is used to manage data after :class:`FullModelShareableGenerator<nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator>`
converts Learnable to Shareable on the FL server. The DXO helps all of the FL components agree on the format.

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


Note that the component with id ``persistor`` points to the custom ``NPModelPersistor`` with full Python module path.


.. literalinclude:: ../../examples/hello-numpy-sag/config/config_fed_client.json
   :language: json
   :linenos:
   :caption: config_fed_client.json


Here, in ``executors``, the Trainer implementation ``NPTrainer`` is configured for the task "train".


Federated Numpy with Scatter and Gather Workflow!
-------------------------------------------------

Now you can use admin commands to upload, deploy, and start this example app. To do this on a proof of concept local
FL system, follow the sections :ref:`setting_up_poc` and :ref:`starting_poc` if you have not already.

Running the FL System
^^^^^^^^^^^^^^^^^^^^^

With the admin client command prompt successfully connected and logged in, enter the commands below in order.  Pay close
attention to what happens in each of four terminals.  You can see how the admin controls the server and clients with
each command.

.. code-block:: shell

    > upload_app hello-numpy-sag

Uploads the application from the admin client to the server's staging area.

.. code-block:: shell

    > set_run_number 1

Creates a run directory in the workspace for the run_number on the server and all clients. The run directory allows for
the isolation of different runs so the information in one particular run does not interfere with other runs.

.. code-block:: shell

    > deploy_app hello-numpy-sag all

This will make the hello-numpy-sag application the active one in the run_number workspace.  After the above two commands, the
server and all the clients know the hello-numpy-sag application will reside in the ``run_1`` workspace.


.. code-block:: shell

    > start_app all

This ``start_app`` command instructs the NVIDIA FLARE server and clients to start training with the hello-numpy-sag
application in the ``run_1`` workspace.

From time to time, you can issue ``check_status server`` in the admin client to check the entire training progress.

You should now see how the training does in the very first terminal (the one that started the server).

After starting the server and clients, you should begin to see
some outputs in each terminal tracking the progress of the FL run. If everything went as
planned, you should see that through 10 rounds, the FL system has aggregated new models on the server
with the results produced by the clients.

Once the fl run is complete and the server has successfully aggregated the client's results after all the rounds, run
the following commands in the fl_admin to shutdown the system (while inputting ``admin`` when prompted with password):

.. code-block:: shell

    > shutdown client
    > shutdown server
    > bye

In order to stop all processes, run ``./stop_fl.sh``.

Congratulations!
You've successfully built and run your first numpy federated learning system. 
You now have a decent grasp of the main FL concepts, and are ready to start exploring how NVIDIA FLARE can be applied to many other tasks.

The full `source code <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-numpy-sag>`_ for this exercise can be found in ``examples/hello-numpy-sag``.
