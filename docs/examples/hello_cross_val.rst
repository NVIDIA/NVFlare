Quickstart (Numpy - Cross Site Validation)
==========================================

Before You Start
----------------

Before jumping into this QuickStart guide, make sure you have an environment with `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_ installed.  You can follow the
:doc:`installation <../installation>` guide on the general concept of Python virtual environment (the recommended environment) and how to
install NVIDIA FLARE.

Prerequisite
-------------

This example builds on the :doc:`Hello Numpy <hello_numpy>` example. Please make sure you go through it completely as the concepts are heavily tied.

Introduction
-------------

This tutorial is meant to solely demonstrate how the NVIDIA FLARE system works, without introducing any actual deep learning concepts.
Through this exercise, you will learn how to use NVIDIA FLARE with numpy to perform cross site validation after training. The training process is explained in the :doc:`Hello Numpy <hello_numpy>` example.
Using simplified weights and metrics, you will be able to clearly see how NVIDIA FLARE performs validation across different sites with little extra work.

The design of this exercise follows on the :doc:`Hello Numpy <hello_numpy>` example which consists of one **server** and two **clients**, starting with weights ``[0, 1]``. For the purpose of this basic example, we'll consider the Fibonacci array as our local model. Cross site validation consists of the following steps:

During training: 

#. Trainer must save a local model to disk.
#. Trainer must register this local model for validation using MLModelRegistry (explained later).
#. Server must register any models during training (explained later).

Cross site validation:

#. Client retrieves the local model from MLModelRegistry and submits it to the server.
#. Client then asks the server for models to validate.
#. Server finds the list of models this client should validate.
#. Server sends one of these models to the client along with a flag *more_models_available*.
#. Client validates the received model from server and submits the results.
#. Client checks the *more_models_available* flag and decides if another request should be sent or cross validation should be finished.

During this exercise, we will see how NVIDIA FLARE takes care of most of the above steps with little work from the user. We will be working with the ``hello-cross-val`` application in the examples folder. Every custom FL application must contain three folders:

 #. **custom**: contains the custom components (``trainer.py``, ``model_persistor.py``, ``validator.py``)
 #. **config**: contains client and server configurations (``config_fed_client.json``, ``config_fed_server.json``)
 #. **resources**: contains the logger config (``log.config``)

Let's get started. First clone the repo, if you haven't already:

.. code-block:: shell

  $ git clone https://github.com/nvidia/nvflare.git

Remember to activate your NVIDIA FLARE Python virtual environment from the installation guide. Ensure numpy is installed.

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install numpy

Now that you have all your dependencies installed, let's implement the Federated Learning system.


Register MLModel during training
--------------------------------
 
In the :doc:`Hello Numpy <hello_numpy>` example, we implemented the ``Trainer`` object that calculates a 
new sequence to submit to the server. For this example, we will consider this new sequence as this 
client's local model. In order to get our model validated on other client's data, first, we must register
it during training.

.. literalinclude:: ../../hello_nvflare/examples/hello-cross-val/custom/trainer.py
   :language: python
   :lines: 47-62 
   :caption: trainer.py

Let's take a look at the above code in *trainer.py*. First, we must save a model to disk so that it 
can be used later. In this example, we simply save the numpy array as a *.npz* file. In a real application, 
this could be a ML model.

Then, we need to register this model for cross site validation. The *fl_ctx* contains the ``MLModelRegistry``.
This is a shared register for keeping track of the model between training and validation. Training produces the 
model and adds to the register while validation retrieves the model. An entry in the register is denoted by ``MLModelEntry``
which consists of name, meta dict and files dict. Validation will look for a model entry with key ``MLModelKeys.LOCAL_BEST_MODEL``. 
As you can see in the example, trainer registers the numpy array it saved using the **LOCAL_BEST_MODEL** key.

You can learn more about ``Shareable`` and ``FLContext`` in the :doc:`hello_pt` exercise or in the :ref:`documentation <programming_guide:Key Objects>`.

Note that the server also produces a global model. We can register it for cross site validation similar 
to client. This is done in *model_persistor.py* as this is the component responsible for saving the model.

.. literalinclude:: ../../hello_nvflare/examples/hello-cross-val/custom/model_persistor.py
   :language: python
   :lines: 65-71
   :caption: model_persistor.py


Implementing the Validator
--------------------------

The validator is responsible for validating the models received from the server. These models could 
be from other clients or models generated on server. 

.. literalinclude:: ../../hello_nvflare/examples/hello-cross-val/custom/validator.py
   :language: python
   :linenos:
   :caption: validator.py

The validator implements the **validate** function which receives a shareable. The shareable 
contains two important keys:

#. **ShareableKey.DATA**: This contains the dict of files for a model. These are the same files that
#.  were added to MLModelEntry with ``add_files()``.
#. **ShareableKey.META**: This contains the meta data for a model. This is the same metadata added 
#. to the MLModelEntry with ``add_meta_data()``.

To validate the model, the common steps are :

#. Retreive the model files from shareable (using **ShareableKey.DATA**)
#. Load the model from the files (user implementation)
#. Validate the model (user implementation)
#. Return the results in a shareable to server. 

The shareable that needs to be returned must be of **TYPE_METRICS**. Metrics are added using 
**ShareableKey.METRICS** key.

.. note::

  Note that in our hello-examples, we are demonstrating Federated Learning using the familiar deep learning model concept.

  However, Federated Learning is not restricted to just deep learning and may not always involve models. Thus, 
  we define a :ref:`Learnable <programming_guide:Learnable>` object (subclasses ``dict``) as the most general form of data object generated through Federated Learning.
  The ``Model`` class we are using here is simply a subclass of ``Learnable``.


Application Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside the config folder there are two files, ``config_fed_client.json`` and ``config_fed_server.json``.
By default, cross site validation is disabled. In order to enable it, the following configuration needs to be added

.. literalinclude:: ../../hello_nvflare/examples/hello-cross-val/config/config_fed_client.json
   :language: python
   :lines: 13-22
   :caption: config_fed_client.json

The key settings are:

#. **cross_validate**: This flag turns cross validation workflow ON or OFF. The workflow includes sharing own local model & validating other models.
#. **cross_site_validation**: This configuration defines the workflow parameters.
#. **is_participating**: If True, client shares their local model with server. If false, client does not share their model.
#. **validator**: Defines the validator to run validation.

.. note::

    Note that if cross site validation workflow is enabled, the client will always validate models from other sources.

Cross site validation!
----------------------

Now you must set up a local environment and generate packages to simulate the server, clients, and admin. The steps to set up and run this application are identical to the :doc:`Hello Numpy <hello_numpy>` example except the app is now ``hello-cross-val`` instead of ``hello-numpy``.

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

    > upload_app hello-cross-val

Uploads an application in the server's registry.  This creates the application entry, populates the configuration and links the name
``hello-cross-val`` with such application configuration.  Later, you can control this application via this name.

.. code-block:: shell

    > set_run_number 1

Creates a workspace with the run_number on the server and all clients.  The purpose of this workspace is to isolate different runs so
the information in one particular run does not interfere with other runs.

.. code-block:: shell

    > deploy_app hello-cross-val all

This will make the hello-cross-val application the active one in the run_number workspace.  After the above two commands, the
server and all the clients know the hello-cross-val application will reside in the ``run_1`` workspace.


.. code-block:: shell

    > start_app all

This ``start_app`` command instructs the NVIDIA FLARE server and clients to start training with the hello-cross-val application in that ``run_1`` workspace.

From time to time, you can issue ``check_status server`` in the admin client to check the entire training progress. During the first phase,
the model will be trained. During the second phase, cross site validation will happen. The status of each
client will change to cross site validation as it enters this second phase.

Accessing the results
---------------------

During cross site validation, every client validates other clients' models and server models (if present).
This can produce a lot of results. All the results are kept on the server in 
*<run_dir>/cross_val_results.json*. All the models sent to the server are also present in the
*<run_dir>/<client_uid>/* directory. To access results using fl_admin, please run:

.. code-block:: shell

    > validate <data_client> <model_client>

OR

.. code-block:: shell

    > validate all

The results will be in the json format.


Understanding the Output
^^^^^^^^^^^^^^^^^^^^^^^^^

After starting the server and clients, you should begin to see 
some outputs in each terminal tracking the progress of the FL run. 
As each client finishes training, it will start the cross site validation process.
Druing this you'll see several important outputs the track the progress 
of cross site validation.

Let's see the progress of cross site validation using output snippets. When 
cross site validation begins, you should see the following prompt on client side.
This mentions the timeout (max time to wait without receiving models from server) 
and the response of server for submitting client's own local model.

.. code-block:: shell
    :caption: site-1

    2021-08-29 15:05:32,312 - CrossSiteValManager - INFO - Cross validation timeout is set to 10 minutes.
    2021-08-29 15:05:32,351 - Communicator - INFO - Server reply to SubmitBestLocalModel:  Received best model from site-1.. SubmitBestLocalModel time: 0.014548540115356445 seconds

Now the client will start asking server for models. One scenario could be that 
server has no models currently available. In that case, server will ask client to wait
until more models are available. The output will look like 

.. code-block:: shell
    :caption: site-2

    2021-08-29 15:05:32,364 - FederatedClient - INFO - Getting other models from server for cross validation.
    2021-08-29 15:05:32,377 - Communicator - INFO - Received 0 models for validation. GetValidationModels time: 0.012315750122070312 seconds
    2021-08-29 15:05:32,382 - FederatedClient - INFO - Server has no models available currently. Waiting 60 secs before asking again.

Another scenario is that the server has models available. In this case, client will receive the model and
send it to validator. After validation, the results will be sent back to the server. Client output 
will look something like

.. code-block:: shell
    :caption: server

    2021-08-29 15:06:32,451 - FederatedClient - INFO - Getting other models from server for cross validation.
    2021-08-29 15:06:32,468 - Communicator - INFO - Received 1 models for validation. GetValidationModels time: 0.014800786972045898 seconds
    2021-08-29 15:06:32,495 - SimpleValidator - INFO - Numpy model loaded from shareable: [54. 88.]
    2021-08-29 15:06:32,495 - SimpleValidator - INFO - Metadata received with model: {'cookie': {'cross_val_model_owner': 'site-1'}}
    2021-08-29 15:06:32,496 - SimpleValidator - INFO - Validating the model.
    2021-08-29 15:06:32,497 - FederatedClient - INFO - Submitting cross validation results to server.
    2021-08-29 15:06:32,513 - Communicator - INFO - Received comments:  Received Cross Validation results from site-1.. SubmitCrossSiteValidationResults time: 0.014247655868530273 seconds

As you can see, the client receives a shareable with Numpy array of ``[54, 88]`` and meta data showing the owner of the model.
After validation the client sends the results to the server.

On the server side, you can see a full log of each client who requested the model and the results received.

.. code-block:: shell
    :caption: server

    2021-08-29 15:52:47,670 - ServerCrossSiteValManager - INFO - Received best model from site-2
    2021-08-29 15:52:47,695 - ServerCrossSiteValManager - INFO - Client site-2 requested models for cross site validation.
    2021-08-29 15:52:47,696 - ServerCrossSiteValManager - INFO - Sent 1 out of 2 models to site-2. Will be asked to wait and retry.
    2021-08-29 15:52:47,697 - ServerCrossSiteValManager - INFO - Sending site-1's model to site-2 for validation.
    2021-08-29 15:52:47,737 - ServerCrossSiteValManager - INFO - site-2 submitted results of validating site-1's model.
    2021-08-29 15:52:47,738 - ServerCrossSiteValManager - INFO - New results added are: {'site-2': {'site-1': {'result': [10, 11]}}}

As you can see the server shows the log of each client requesting models, the models it sends and the 
results received. Since the server could be responding to many clients at the same time, it may 
require careful examination to make proper sense of events from the jumbled logs.

Once the FL run is complete and the server has successfully aggregrated the client's results after all the rounds, and cross site validation is finished, run the following commands in the fl_admin to shutdown the system (while inputting ``admin`` when prompted with password):

.. code-block:: shell

    > shutdown client
    > shutdown server
    > bye

In order to stop all processes, run ``./stop_fl.sh``. 

Congratulations!
You've successfully run your numpy federated learning system with cross site validation.
