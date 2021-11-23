Quickstart (PyTorch)
====================

Before You Start
----------------

Feel free to refer to the official :doc:`documentation <../programming_guide>` at any point to learn more about the specifics of `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_.

Make sure you have an environment with NVIDIA FLARE installed.  You can follow the
:doc:`installation <../installation>` guide on the general concept of Python virtual environment (the recommended environment) and how to
install NVIDIA FLARE.


Introduction
-------------

Through this exercise, you will integrate NVIDIA FLARE with the popular
deep learning framework `PyTorch <https://pytorch.org/>`_ and learn how to use NVIDIA FLARE to train a convolutional network with the CIFAR10 dataset.

The design of this exercise consists of one **server** and two **clients** all having the same PyTorch model. 
The following steps compose one cycle of weight updates, called a **round**:

 #. Clients are responsible for generating individual weight-updates for the model using their own CIFAR10 dataset. 
 #. These updates are then sent to the server which will aggregate them to produce a model with new weights. 
 #. Finally, the server sends this updated version of the model back to each client.

For this exercise, we will be working with the ``hello-pt`` application in the examples folder. 
Every custom FL application must contain three folders:

 #. **custom**: contains the custom components (``net.py``, ``trainer.py``)
 #. **config**: contains client and server configurations (``config_fed_client.json``, ``config_fed_server.json``)
 #. **resources**: contains the logger config (``log.config``)

Now that you have a rough idea of what is going on, let's get started. First clone the repo:

.. code-block:: shell

  $ git clone https://github.com/nvidia/nvflare.git

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

.. literalinclude:: ../../examples/hello-pt/custom/pt_net.py
   :language: python
   :caption: pt_net.py

This ``Net`` class is your convolutional neural network to train with the CIFAR10 dataset.
This is not related to NVIDIA FLARE, so we implement it in a file called ``net.py``.

Dataset & Setup
^^^^^^^^^^^^^^^^

Now all we have left is to implement the one required custom class related to NVIDIA FLARE, ``SimpleTrainer``, in a file
called ``trainer.py``.

In a real FL experiment, each client would have their own dataset used for their local training.
For simplicity's sake, you can download the same CIFAR10 dataset from the Internet via torchvision's datasets module.
Additionally, you need to setup the optimizer, loss function and transform to process the data.
You can think of all of this code as part of your local training loop, as every deep learning training has a similar setup.

Since you will encapsulate every training-related step in the ``SimpleTrainer`` class, let's put this preparation stage into the ``__init__`` method:

.. literalinclude:: ../../examples/hello-pt/custom/trainer.py
   :language: python
   :lines: 1-31


Local Train
^^^^^^^^^^^

Now that you have your network and dataset setup, in the ``Trainer`` class let's also implement a local training loop in a method called ``local_train``:

.. literalinclude:: ../../hello_nvflare/examples/hello-pt/custom/trainer.py
   :language: python
   :pyobject: SimpleTrainer.local_train


This exercise does not include the validation portion and any print for simplicity. 
The number of epochs is hardcoded to 2, but this can be easily set from a parameter.

.. note::

  Everything up to this point is completely independent of NVIDIA FLARE. It is just purely a PyTorch
  deep learning exercise.  You will now build the NVIDIA FLARE application based on this PyTorch code.


Integrate NVIDIA FLARE with Local Train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NVIDIA FLARE makes it very easy to integrate your local train code into the NVIDIA FLARE API.

The simplest way to do this is to subclass the ``Trainer`` class and
implement one method ``train``, which is called every time the client receives 
an updated model from the server. We can then call our local train inside the ``train`` method.

.. note::

  The ``train`` method inside the ``Trainer`` class is where all of the client side computation occurs.
  In these exercises, we update the weights by training on a local dataset, however, it is important to remember that NVIDIA FLARE is not restricted to just deep learning.
  The type of data passed between the server and the clients, and the computations that the clients perform can be anything, as long as all of the FL Components agree on the same format. 

Take a look at the following code:

.. literalinclude:: ../../hello_nvflare/examples/hello-pt/custom/trainer.py
   :language: python
   :pyobject: SimpleTrainer.train

The concept of ``Shareable`` is described briefly at :ref:`shareable` section.  Essentially, every NVIDIA FLARE client receives the model weights
from the server in ``shareable`` passed into the train method, and returns a new ``shareable`` back to the server.

Thus, the first thing is to retrieve the model weights delivered by server via ``shareable``:

.. code-block:: python

  # retrieve model weights download from server's shareable
  model_weights = shareable[ShareableKey.MODEL_WEIGHTS]

Now we can update the local model with those received weights:

.. code-block:: python

  self.model.load_state_dict({k: torch.as_tensor(v) for k,v in model_weights.items()})

We then perform a local train so the client's model is trained with its own dataset:

.. code-block:: python

  self.local_train()

After finishing the local train, the train method builds a new ``shareable`` with newly-trained weights and metadata and returns it
back to the NVIDIA FLARE server for aggregation:

.. code-block:: python

  # build the shareable
  shareable = Shareable()
  shareable[ShareableKey.META] = {FLConstants.NUM_STEPS_CURRENT_ROUND: 1}
  shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHTS
  shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
  shareable[ShareableKey.MODEL_WEIGHTS] = {
      k: v.cpu().numpy() for k, v in self.model.state_dict().items()
  }
  return shareable

.. _shareable:

Shareable
^^^^^^^^^

 * The ``Shareable`` is a dictionary object used to share data and metadata between server and clients.  
 * The keys used in the exercise are:

    - ``ShareableKey.META``: used to store metadata.
    - ``ShareableKey.MODEL_WEIGHTS``: used to receive and store weights. (The format is a dictionary with values of type numpy arrays by default. However, this format can be changed to anything as long as you ensure the client trainer and the aggregator agree on the structure.)
    - ``ShareableKey.TYPE``: value is TYPE_WEIGHTS in this exercise.
    - ``Shareablekey.DATA_TYPE``: value is DATA_TYPE_UNENCRYPTED in this exercise.
 
 * There are many other predefined keys and values for the ``Shareable`` which can
   be found in the ``ShareableKey`` and ``ShareableValue`` classes respectively.

You can find more details in the :ref:`documentation <programming_guide:Shareable>`.

FLContext
^^^^^^^^^

 * The ``FLContext`` is used to set and retrieve FL related information among the FL components via ``set_prop()`` and ``get_prop()``. 
 * For example:

    - fl_ctx.get_prop(FLConstants.CLIENT_NAME) retrieves the client name
    - fl_ctx.get_prop(FLConstants.CURRENT_ROUND) retrives the current round of FL
 * The other defined keys can be found in the FLConstants class.

You can find more details in the :ref:`documentation <programming_guide:FLContext>`.

NVIDIA FLARE Server & Application
---------------------------------

In this exercise, you can use the default settings, which leverage NVIDIA FLARE built-in components for NVIDIA FLARE server.  These
built-in components are commonly used in most deep learning scenarios.  However, you are encouraged to build your own components
to fully customize NVIDIA FLARE to meet your environment, which we will demonstrate in the following exercises.


Application Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside the config folder there are two files, ``config_fed_client.json`` and ``config_fed_server.json``.

.. literalinclude:: ../../hello_nvflare/examples/hello-pt/config/config_fed_client.json
   :language: json
   :linenos:
   :caption: config_fed_client.json

Take a look at line 9.  This is the ``SimpleTrainer`` you just implemented.  The NVIDIA FLARE client loads this
application configuration and picks your implementation.  You can easily change it to another class so
your NVIDIA FLARE client has different training logic.


.. literalinclude:: ../../hello_nvflare/examples/hello-pt/config/config_fed_server.json
   :language: json
   :linenos:
   :caption: config_fed_server.json

The server application configuration, like said before, leverages NVIDIA FLARE built-in components. Remember, you
are encouraged to change them to your own classes whenever you have different application logic.

Note that on line 26, ``persistor`` points to ``PTFileModelPersistor``. 
NVIDIA FLARE provides a built-in PyTorch implementation for a model persistor, however for other frameworks/libraries, you will have to implement your own.


.. _hands-on:

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

    > upload_app hello-pt

Uploads an application in the server's registry.  This creates the application entry, populates the configuration and links the name
``hello-pt`` with such application configuration.  Later, you can control this application via this name.

.. code-block:: shell

    > set_run_number 1

Creates a workspace with the run_number on the server and all clients.  The purpose of this workspace is to isolate different runs so
the information in one particular run does not interfere with other runs.

.. code-block:: shell

    > deploy_app hello-pt all

This will make the hello-pt application the active one in the run_number workspace.  After the above two commands,
the server and all the clients know the hello-pt application will reside in the ``run_1`` workspace.


.. code-block:: shell

    > start_app all

This ``start_app`` command instructs the NVIDIA FLARE server and clients to start training with the hello-pt application in that ``run_1`` workspace.

From time to time, you can issue ``check_status server`` in the admin client to check the entire training progress.

You should now see how the training does in the very first terminal (the one that started the server):



.. code-block:: shell

    2021-06-25 08:30:28,755 - FederatedServer - INFO - starting secure server at localhost:8002
    deployed FL server trainer.
    2021-06-25 08:30:28,763 - FedAdminServer - INFO - Starting Admin Server localhost on Port 8003
    2021-06-25 08:30:28,763 - root - INFO - Server started
    2021-06-25 08:30:41,862 - ClientManager - INFO - Client: New client site-1@127.0.0.1 joined. Sent token: 51b4bbc2-385f-4193-9891-31392cace676.  Total clients: 1
    Create initial model message...
    created initial model_data...
    2021-06-25 08:31:27,362 - FederatedServer - INFO - Server training has been started.
    2021-06-25 08:31:38,389 - FederatedServer - INFO - GetModel requested from: 51b4bbc2-385f-4193-9891-31392cace676
    2021-06-25 08:31:38,402 - FederatedServer - INFO - Return model to : 51b4bbc2-385f-4193-9891-31392cace676 for round: 0

On your client terminal:

.. code-block:: shell

    PYTHONPATH is /local/custom:
    start fl because of no pid.fl
    new pid 10719
    2021-06-25 08:30:41,863 - FederatedClient - INFO - Successfully registered client:site-1 for example_project. Got token:51b4bbc2-385f-4193-9891-31392cace676
    created /tmp/fl/site-1/comm/training/x
    created /tmp/fl/site-1/comm/training/y
    created /tmp/fl/site-1/comm/training/t
    2021-06-25 08:31:37,523 - ClientAdminInterface - INFO - Starting client training. rank: 0
    training child process ID: 10938
    starting the client .....
    token is: 51b4bbc2-385f-4193-9891-31392cace676 run_number is: 1 uid: site-1 listen_port: 58949
    2021-06-25 08:31:38,199 - SimpleTrainer - INFO - epochs_per_round: 1, validation_interval: 2000
    2021-06-25 08:31:38,332 - FederatedClient - INFO - Starting to fetch global model.
    2021-06-25 08:31:38,540 - ProcessExecutor - INFO - waiting for process to finish
    Created the listener on port: 58949
    2021-06-25 08:31:38,690 - Communicator - INFO - Received example_project model at round 0 (4800501 Bytes). GetModel time: 0.35499143600463867 seconds
    Get global model for round: 0
    pull_models completed. Status:True rank:0

Once the fl run is complete and the server has successfully aggregrated the clients' results after all the rounds, run the following commands in the fl_admin to shutdown the system (while inputting ``admin`` when prompted with password):

.. code-block:: shell

    > shutdown client
    > shutdown server
    > bye

In order to stop all processes, run ``./stop_fl.sh``.

All artifacts from the FL run can be found in the server run folder you created with ``set_run_number``.  In this exercise, the folder is ``run_1``.

Congratulations!
You've successfully built and run your first federated learning system.
The full `source code <https://gitlab-master.nvidia.com/dlmed/hello_nvflare/-/blob/main/examples/hello-pt/>`_ for this exercise can be found in ``examples/hello-pt``.
