Quickstart (Numpy - Cross Site Validation)
==========================================

Before You Start
----------------

Before jumping into this QuickStart guide, make sure you have an environment with `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_
installed. You can follow :doc:`installation <../installation>` on the general concept of setting up a Python virtual
environment (the recommended environment) and how to install NVIDIA FLARE.

Prerequisite
-------------

This example builds on the :doc:`Hello Numpy <hello_numpy>` example based on the :class:`ScatterAndGather<nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather>`
workflow. Please make sure you go through it completely as the concepts are heavily tied.

Introduction
-------------

This tutorial is meant to solely demonstrate how the NVIDIA FLARE system works, without introducing any actual deep learning concepts.
Through this exercise, you will learn how to use NVIDIA FLARE with numpy to perform cross site validation after training.
The training process is explained in the :doc:`Hello Numpy <hello_numpy>` example.
Using simplified weights and metrics, you will be able to clearly see how NVIDIA FLARE performs validation across different
sites with little extra work.

The design of this exercise follows on the :doc:`Hello Numpy <hello_numpy>` example which consists of one **server** and
two **clients** starting with weights ``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]``.

Cross site validation consists of the following steps:

    - During the initial phase of training with the :class:`ScatterAndGather<nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather>`
      workflow, NPTrainer saves the local model to disk for the clients.
    - The :class:`CrossSiteModelEval<nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval>` workflow gets
      the client models with the ``submit_model`` task.
    - The ``validate`` task is broadcast to the all participating clients with the model shareable containing the model data,
      and results from the ``validate`` task are saved.

During this exercise, we will see how NVIDIA FLARE takes care of most of the above steps with little work from the user.
We will be working with the ``hello-numpy-cross-val`` application in the examples folder. Custom FL applications
can contain the folders:

 #. **custom**: contains the custom components (``np_trainer.py``, ``np_model_persistor.py``, ``np_validator.py``, ``np_model_locator``, ``np_formatter``)
 #. **config**: contains client and server configurations (``config_fed_client.json``, ``config_fed_server.json``)
 #. **resources**: contains the logger config (``log.config``)

Let's get started. First clone the repo, if you haven't already:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Remember to activate your NVIDIA FLARE Python virtual environment from the installation guide. Ensure numpy is installed.

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install numpy

Now that you have all your dependencies installed, let's implement the Federated Learning system.


Training
--------------------------------
 
In the :doc:`Hello Numpy <hello_numpy>` example, we implemented the ``NPTrainer`` object. In this example, we use the
same ``NPTrainer`` but extend it to process the ``submit_model`` task to work with the :class:`CrossSiteModelEval<nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval>`
workflow to get the client models.

The code in *np_trainer.py* saves the model to disk after each step of training in the model.

Note that the server also produces a global model. The :class:`CrossSiteModelEval<nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval>`
workflow submits the server model for evaluation after the client models.

Implementing the Validator
--------------------------

The validator is an Executor that is called for validating the models received from the server during the :class:`CrossSiteModelEval<nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval>`
workflow. These models could be from other clients or models generated on server.

.. literalinclude:: ../../examples/hello-numpy-cross-val/custom/np_validator.py
   :language: python
   :lines: 15-
   :lineno-start: 15
   :linenos:
   :caption: np_validator.py

The validator is an Executor and implements the **execute** function which receives a Shareable. It handles the ``validate``
task by performing a calculation to find the sum divided by the max of the data and adding a random random_epsilon before
returning the results packaged with a DXO into a Shareable.

.. note::

  Note that in our hello-examples, we are demonstrating Federated Learning using data that does not have to do with deep learning.
  NVIDIA FLARE can be used with any data packaged inside a :ref:`Shareable <shareable>` object (subclasses ``dict``), and
  :ref:`DXO <data_exchange_object>` is recommended as a way to manage that data in a standard way.

Application Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside the config folder there are two files, ``config_fed_client.json`` and ``config_fed_server.json``.

.. literalinclude:: ../../examples/hello-numpy-cross-val/config/config_fed_server.json
   :language: python
   :linenos:
   :caption: config_fed_server.json

The server now has a second workflow configured after Scatter and Gather, :class:`CrossSiteModelEval<nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval>`.
The components "model_locator" and "formatter" have been added to work with the cross site model evaluation workflow,
and the rest is the same as in :doc:`Hello Numpy <hello_numpy>`.


.. literalinclude:: ../../examples/hello-numpy-cross-val/config/config_fed_client.json
   :language: python
   :linenos:
   :caption: config_fed_client.json

The client configuration now has more tasks and an additional Executor ``NPValidator`` configured to handle the "validate" task.
The "submit_model" task has been added to the ``NPTrainer`` Executor to work with the :class:`CrossSiteModelEval<nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval>`
workflow to get the client models.

Cross site validation!
----------------------

Now you can use admin commands to upload, deploy, and start this example app. To do this on a proof of concept local
FL system, follow the sections :ref:`setting_up_poc` and :ref:`starting_poc` if you have not already.

Running the FL System
^^^^^^^^^^^^^^^^^^^^^

With the admin client command prompt successfully connected and logged in, enter the commands below in order.  Pay close
attention to what happens in each of four terminals.  You can see how the admin controls the server and clients with
each command.

.. code-block:: shell

    > upload_app hello-numpy-cross-val

Uploads the application from the admin client to the server's staging area.

.. code-block:: shell

    > set_run_number 1

Creates a run directory in the workspace for the run_number on the server and all clients. The run directory allows for
the isolation of different runs so the information in one particular run does not interfere with other runs.

.. code-block:: shell

    > deploy_app hello-numpy-cross-val all

This will make the hello-numpy-cross-val application the active one in the run_number workspace.  After the above two commands, the
server and all the clients know the hello-numpy-cross-val application will reside in the ``run_1`` workspace.


.. code-block:: shell

    > start_app all

This ``start_app`` command instructs the NVIDIA FLARE server and clients to start training with the hello-numpy-cross-val
application in that ``run_1`` workspace.

From time to time, you can issue ``check_status server`` in the admin client to check the entire training progress. During the first phase,
the model will be trained. During the second phase, cross site validation will happen. The workflow on the
client will change to :class:`CrossSiteModelEval<nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval>`
as it enters this second phase.

Accessing the results
---------------------

During cross site model evaluation, every client validates other clients' models and server models (if present).
This can produce a lot of results. All the results are kept on the server in 
*<run_dir>/cross_val_results.json*. All the models sent to the server are also present in the
*<run_dir>/<client_uid>/* directory.

The results will be in the json format.


Understanding the Output
^^^^^^^^^^^^^^^^^^^^^^^^^

After starting the server and clients, you should begin to see 
some outputs in each terminal tracking the progress of the FL run. 
As each client finishes training, it will start the cross site validation process.
Druing this you'll see several important outputs the track the progress 
of cross site validation.

The server shows the log of each client requesting models, the models it sends and the
results received. Since the server could be responding to many clients at the same time, it may 
require careful examination to make proper sense of events from the jumbled logs.

Once the FL run is complete and the server has successfully aggregated the client's results after all the rounds, and
cross site model evaluation is finished, run the following commands in the fl_admin to shutdown the system (while
inputting ``admin`` when prompted with password):

.. code-block:: shell

    > shutdown client
    > shutdown server
    > bye

In order to stop all processes, run ``./stop_fl.sh``. 

Congratulations!
You've successfully run your numpy federated learning system with cross site validation.
The full `source code <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-numpy-cross-val/>`_ for this exercise can be found in ``examples/hello-numpy-cross-val``.
