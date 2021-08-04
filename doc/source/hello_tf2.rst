Quickstart (TensorFlow 2)
===========================

Before You Start
----------------

We recommend you first finish either the :doc:`hello_pt` or the :doc:`hello_numpy` exercise.
Those guides go more in depth in explaining the federated learning aspect of `NVFlare <https://pypi.org/project/nvflare/>`_. 
 
Here we assume you have already installed NVFlare inside a python virtual environment and have already cloned the repo.

Introduction
-------------

Through this exercise, you will integrate NVFlare with the popular
deep learning framework `TensorFlow 2 <https://www.tensorflow.org/>`_ and learn how to use NVFlare to train a convolutional network with the MNIST dataset.
You will also be introduced to some new components and concepts, including the filter, aggregrator, and event handler.

The design of this exercise consists of one **server** and two **clients** all having the same TensorFlow 2 model. 
The following steps compose one cycle of weight updates, called a **round**:

 #. Clients are responsible for generating individual weight-updates for the model using their own MNIST dataset. 
 #. These updates are then sent to the server which will aggregate them to produce a model with new weights. 
 #. Finally, the server sends this updated version of the model back to each client.

For this exercise, we will be working with the ``hello-tf2`` application in the examples folder. 
Every custom FL application must contain three folders:

 #. **custom**: contains the custom components (``net.py``, ``trainer.py``, ``filter.py``, ``model_aggregator.py``, ``model_persistor.py``)
 #. **config**: contains client and server configurations (``config_fed_client.json``, ``config_fed_server.json``)
 #. **resources**: contains the logger config (``log.config``)

Let's get started. Since this task is using TensorFlow, let's go ahead and install the library inside our virtual environment: 

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install tensorflow


NVFlare Client
--------------

Network
^^^^^^^^

With all the required dependencies installed, you are ready to run a Federated Learning
with two clients and one server. Before you start, let's see what a simplified MNIST network looks like.

.. literalinclude:: ../../hello_nvflare/examples/hello-tf2/custom/net.py
   :language: python
   :linenos:
   :caption: net.py

This Net class is the convolutional neural network to train with MNIST dataset. This is not related to NVFlare, so implement it in a file called ``net.py``.

Dataset & Setup
^^^^^^^^^^^^^^^^

Now you have to implement the only class related to NVFlare, ``Trainer``, in a file called ``trainer.py``.

Before you can really start a training, you need to setup your dataset.
In this exercise, you can download it from the Internet via ``tf.keras``'s datasets module, and split it in half to create a separate dataset for each client.
Additionally, you must setup the optimizer, loss function and transform to process the data.

Since every step will be encapsulated in the ``Trainer`` class, let's put this preparation stage into one method ``setup``:

.. literalinclude:: ../../hello_nvflare/examples/hello-tf2/custom/trainer.py
   :language: python
   :lines: 24-53
   :lineno-start: 24
   :linenos:


How can you ensure this setup method is called before the client receives the model from the server?  The Trainer
class is also a :ref:`FLComponent <programming_guide:FLComponent>`, which always receives ``Event`` whenever NVFlare enters or leaves a certain stage.
In this case, there is an ``Event`` called ``EventType.START_RUN`` which perfectly matches these requirements. 
Because our trainer is a subclass of ``FLComponent``, you can implement the handler to handle the event and call the setup method:

.. literalinclude:: ../../hello_nvflare/examples/hello-tf2/custom/trainer.py
   :language: python
   :lines: 20-22
   :lineno-start: 20
   :linenos:

.. note::

  This is a new concept you haven't learned in previous two exercises.  The concepts about ``event`` and ``handler`` are very powerful because
  you are free to add your logic so it can run at different time and process various events.  The entire list of events fired by
  NVFlare is shown at :ref:`Event types <programming_guide:Event types>`.


You have everything you need, now let's implement the last method called ``train``, which is 
called every time the client receives an updated model from the server.


Link NVFlare with Local Train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take a look at the following code:

.. literalinclude:: ../../hello_nvflare/examples/hello-tf2/custom/trainer.py
   :language: python
   :lines: 55-
   :lineno-start: 55
   :linenos:

Every NVFlare client receives the model weights from the server in the :ref:`shareable <programming_guide:Shareable>`.
This exercise uses a simple ``exclude_var`` filter, so make sure to replace the missing layer with weights from the clients' previous training round:

.. code-block:: python

  for key, value in model_weights.items():
        if np.all(value == 0):
            model_weights[key] = prev_weights[key]

Now update the local model with those received weights:

.. code-block:: python

  self.model.set_weights(list(model_weights.values()))

Then peform a simple :code:`self.model.fit` so the client's model is trained with its own dataset:

.. code-block:: python

    self.model.fit(
        self.train_images,
        self.train_labels,
        epochs=self.epochs_per_round,
        validation_data=(self.test_images, self.test_labels),
    )
  
After finishing the local train, the train method builds a new ``shareable`` with newly-trained weights and returns it
back to the NVFlare server.


NVFlare Server & Application
-----------------------------

Filter
^^^^^^^ 

The :ref:`filter <programming_guide:Filters>` is used for additional data processing in the ``shareable``, either inbound or outbound from the client and/or server. 

For this exercise, we use a basic ``exclude_var`` filter to exclude the variable/layer ``flatten`` outbound from the client to the server.
The excluded layer is replaced with all zeros of the same shape, which reduces compression size and ensures that the clients' weights
for this variable are not shared with the server.

.. literalinclude:: ../../hello_nvflare/examples/hello-tf2/custom/filter.py
   :language: python
   :linenos:
   :caption: filter.py

The filtering procedure occurs in the one required method, process, which receives and returns a shareable.
The parameters for what is excluded and the inbound/outbound option are all set in ``config_fed_client.json`` (shown later below) and passed in through the constructor.


Model Aggregator
^^^^^^^^^^^^^^^^

The :ref:`model aggregator <programming_guide:Aggregator>` is used by the server to aggregrate the clients' models into one model.

.. literalinclude:: ../../hello_nvflare/examples/hello-tf2/custom/model_aggregator.py
   :language: python
   :linenos:
   :caption: model_aggregator.py

In this exercise, we perform a simple average over the two clients' weights. 

The ``accept()`` function is responsible for managing the shareables from the clients. 
Note that the second return boolean can be set to ``True`` based on some logic to call ``aggregate()`` early, 
otherwise by default ``aggregate()`` will always be called once the number of submissions is >= ``min_clients``.

The ``aggregrate()`` function then performs the aggregration procedure and returns a new shareable.

FLContext is used throughout these functions to provide various useful FL-related information. 
You can find more details in the :ref:`documentation <programming_guide:FLContext>`.

Model Persistor
^^^^^^^^^^^^^^^

The model persistor is used to load and save models on the server.

.. literalinclude:: ../../hello_nvflare/examples/hello-tf2/custom/model_persistor.py
   :language: python
   :linenos:
   :caption: model_persistor.py

In this exercise, we simply serialize the model weights dictionary using pickle and save it to ``server/run_1/mmar_server/models/tf2weights.pickle``
inside the server directory (weights file name defined in ``config_fed_server.json``).
Depending on the frameworks and tools, the methods of saving the model may vary.

Application Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, inside the config folder there are two files, ``config_fed_client.json`` and ``config_fed_server.json``.

.. literalinclude:: ../../hello_nvflare/examples/hello-tf2/config/config_fed_server.json
   :language: json
   :linenos:
   :caption: config_fed_server.json


Note that the ``aggregator`` points to the custom ``AccumulateAggregator``, 
and the ``persistor`` points to the custom ``TF2ModelPersistor`` with full Python module paths. 


.. literalinclude:: ../../hello_nvflare/examples/hello-tf2/config/config_fed_client.json
   :language: json
   :linenos:
   :caption: config_fed_client.json


Here, the ``client_trainer`` points to the Trainer implementation ``SimpleTrainer``. 
Also we set the outbound filter path to ``filter.ExcludeVars`` and pass in ``["flatten"]`` as the argument.

Train the Model, Federated!
---------------------------

Now you must set up a local environment and generate packages to simulate the server, clients, and admin.

Setting Up the Application Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This command generates a poc folder with server, one client and one admin:

.. code-block:: shell

    $ poc

Here we duplicate the client folder into two folders to create two clients, site-1 and site-2:

.. code-block:: shell

    $ cp -r poc/client poc/site-1
    $ cp -r poc/client poc/site-2

Finally, we copy necessary files (the exercise codes) to a working folder:

.. code-block:: shell

  $ mkdir -p poc/admin/transfer
  $ cp -rf examples/* poc/admin/transfer

With both the client and server ready, you can now run everything and see federated
learning in action. FL systems usually have a server and multiple clients. We
therefore have to start the server first:

.. code-block:: shell

    $ ./poc/server/startup/start.sh

Once the server is running you can start the clients in different terminals.
Open a new terminal and start the first client:

.. code-block:: shell

    $ ./poc/site-1/startup/start.sh site-1 localhost

Open another terminal and start the second client:

.. code-block:: shell

    $ ./poc/site-2/startup/start.sh site-2 localhost

In one last terminal, start the admin:

.. code-block:: shell

  $ ./poc/admin/startup/fl_admin.sh localhost


This will launch a command prompt, where you can input commands to control and monitor many aspects of
the FL process. Log in by entering ``admin`` for both the username and password.

Running the FL
^^^^^^^^^^^^^^

Enter the commands below in order.  Pay close attention to what happens in each of four terminals.  You
can see the admin controls server and clients with each command.

.. code-block:: shell

    > upload_app hello-tf2

Uploads an application in the server's registry.  This creates the application entry, populates the configuration and links the name
``hello-tf2`` with such application configuration.  Later, you can control this application via this name.

.. code-block:: shell

    > set_run_number 1

Creates a workspace with the run_number on the server and all clients.  The purpose of the workspace is to isolate different runs so
the information in one particular run does not interfere with other runs.

.. code-block:: shell

    > deploy_app hello-tf2 all

This will make the hello-tf2 application the active one in the run_number workspace.  In this exercise, after the above two commands, the
server and all the clients know the hello-tf2 application will reside in ``run_1`` workspace.


.. code-block:: shell

    > start_app all

This ``start_app`` command instructs the NVFlare server and clients to start training with the hello-tf2 application in that ``run_1`` workspace.

From time to time, you can issue ``check_status server`` in the admin client to check the entire training progress.

You should now see how the training does in the very first terminal (the one that started the server):


.. code-block:: shell

    2021-07-02 14:25:41,361 - FederatedServer - INFO - starting secure server at localhost:8002
    deployed FL server trainer.
    2021-07-02 14:25:41,366 - FedAdminServer - INFO - Starting Admin Server localhost on Port 8003
    2021-07-02 14:25:41,366 - root - INFO - Server started
    2021-07-02 14:25:50,530 - ClientManager - INFO - Client: New client site-1@127.0.0.1 joined. Sent token: fd9e29f-cb4a-4162-b684-b16be3958830.  Total clients: 1
    2021-07-02 14:25:58,435 - ClientManager - INFO - Client: New client site-2@127.0.0.1 joined. Sent token: 42fa2ef-5330-4c2f-9639-77ceac1f6db1.  Total clients: 2
    Create initial model message...
    created initial model_data...
    2021-07-02 14:27:01,586 - FederatedServer - INFO - Server training has been started.
    2021-07-02 14:27:13,344 - FederatedServer - INFO - GetModel requested from: fd96e29f-cb4a-4162-b684-b16be395830
    2021-07-02 14:27:13,347 - FederatedServer - INFO - Return model to : fd96e29f-cb4a-4162-b684-b16be3958830 fo round: 0

On your client terminal:

.. code-block:: shell

    starting the client .....
    token is: 42f6a2ef-5330-4c2f-9639-77ceac1f6db1 run_number is: 1 uid: site-2 listen_port: 46465
    Created the listener on port: 46465
    2021-07-02 14:27:14,772 - FederatedClient - INFO - Starting to fetch global model.
    2021-07-02 14:27:14,825 - Communicator - INFO - Received example_project model at round 0 (407809 Bytes). GetModel time: 0.04601311683654785 seconds
    2021-07-02 14:27:15.154289: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
    2021-07-02 14:27:15.174696: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2900000000 Hz
    Get global model for round: 0
    pull_models completed. Status:True rank:0
    Epoch 1/2
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.0128 - accuracy: 0.9959 - val_loss: 0.1210 - val_accuracy: 0.9822
    Epoch 2/2
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.0122 - accuracy: 0.9959 - val_loss: 0.1257 - val_accuracy: 0.9821
    Send model to server.
    2021-07-02 14:27:23,849 - FederatedClient - INFO - Starting to push model.
    2021-07-02 14:27:23,853 - Communicator - INFO - Send example_project at round 0

Once the fl run is complete and the server has successfully aggregrated the clients' results after all the rounds, 
run the following commands in the fl_admin to shutdown the system (while inputting ``admin`` when prompted with user name):

.. code-block:: shell

    > shutdown client
    > shutdown server
    > bye

In order to stop all processes, run ``./stop_fl.sh``.

All artifacts from the FL run can be found in the server run folder you created with ``set_run_number``.  In this exercise,
the folder is ``run_1``.

Congratulations!
You've successfully built and run a federated learning system using TensorFlow 2.
The full `source code <https://gitlab-master.nvidia.com/dlmed/hello_nvflare/-/blob/main/examples/hello-tf2/>`_ for this exercise can be found in ``examples/hello-tf2``.
