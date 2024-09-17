.. _hello_tf_job_api:

Hello TensorFlow with Job API
==============================

Before You Start
----------------
Feel free to refer to the :doc:`detailed documentation <../programming_guide>` at any point
to learn more about the specifics of `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_.

We recommend you first finish the :doc:`Hello FedAvg with NumPy <hello_fedavg_numpy>` exercise since it introduces the
federated learning concepts of `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_.

Make sure you have an environment with NVIDIA FLARE installed.

You can follow :ref:`getting_started` on the general concept of setting up a
Python virtual environment (the recommended environment) and how to install NVIDIA FLARE.

Here we assume you have already installed NVIDIA FLARE inside a python virtual environment
and have already cloned the repo.

Introduction
-------------
Through this exercise, you will integrate NVIDIA FLARE with the popular deep learning framework
`TensorFlow <https://www.tensorflow.org/>`_ and learn how to use NVIDIA FLARE to train a convolutional
network with the MNIST dataset using the :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>` workflow.

You will also be introduced to some new components and concepts, including filters, aggregators, and event handlers.

The setup of this exercise consists of one **server** and two **clients**.

The following steps compose one cycle of weight updates, called a **round**:

 #. Clients are responsible for generating individual weight-updates for the model using their own MNIST dataset. 
 #. These updates are then sent to the server which will aggregate them to produce a model with new weights. 
 #. Finally, the server sends this updated version of the model back to each client.

For this exercise, we will be working with the ``hello-tf`` application in the examples folder. 

Let's get started. Since this task is using TensorFlow, let's go ahead and install the library inside our virtual environment:

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install tensorflow

With all the required dependencies installed, you are ready to run a Federated Learning system
with two clients and one server. If you would like to go ahead and run the exercise now, you can run
the ``fedavg_script_executor_hello-tf.py`` script which builds the job with the Job API and runs the
job with the FLARE Simulator.

NVIDIA FLARE Job API
--------------------
The ``fedavg_script_executor_hello-tf.py`` script for this hello-tf example is very similar to the ``fedavg_script_executor_hello-numpy.py`` script
for the :doc:`Hello FedAvg with NumPy <hello_fedavg_numpy>` example and also the script for the :doc:`Hello PyTorch <hello_pt_job_api>`
example. Other than changes to the names of the job and client script, the only difference is the line to define the initial global model
for the server:

.. code-block:: python

   # Define the initial global model and send to server
   job.to(TFNet(), "server")


NVIDIA FLARE Client Training Script
------------------------------------
The training script for this example, ``hello-tf_fl.py``, is the main script that will be run on the clients. It contains the TensorFlow specific
logic for training.

Neural Network
^^^^^^^^^^^^^^^
Let's see what a simplified MNIST network looks like.

.. literalinclude:: ../../examples/hello-world/hello-tf/src/tf_net.py
   :language: python
   :lines: 15-
   :lineno-start: 15
   :linenos:
   :caption: tf_net.py

This ``TFNet`` class is the convolutional neural network to train with MNIST dataset.
This is not related to NVIDIA FLARE, and it is implemented in a file called ``tf_net.py``.

Dataset & Setup
^^^^^^^^^^^^^^^^
Before starting training, you need to set up your dataset.
In this exercise, it is downloaded from the Internet via ``tf.keras``'s datasets module
and split in half to create a separate dataset for each client. Note that this is just for an example since in a real-world scenario,
you will likely have different datasets for each client.

Additionally, the optimizer and loss function need to be configured.

All of this happens before the ``while flare.is_running():`` line in ``hello-tf_fl.py``.

.. literalinclude:: ../../examples/hello-world/hello-tf/src/hello-tf_fl.py
   :language: python
   :lines: 29-57
   :lineno-start: 29
   :linenos:
   :caption: hello-tf_fl.py

Client Local Train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The client code gets the weights from the input_model received from the server then performs a simple :code:`self.model.fit`
so the client's model is trained with its own dataset:

.. literalinclude:: ../../examples/hello-world/hello-tf/src/hello-tf_fl.py
   :language: python
   :lines: 58-91
   :lineno-start: 58
   :linenos:
  
After finishing the local training, the newly-trained weights are sent back to the NVIDIA FLARE server in the params of
:mod:`FLModel<nvflare.app_common.abstract.fl_model>`.


NVIDIA FLARE Server & Application
---------------------------------
In this example, the server runs :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>` with the default settings.

If you export the job with the :func:`export<nvflare.job_config.api.FedJob.export>` function, you will see the
configurations for the server and each client. The server configuration is ``config_fed_server.json`` in the config folder
in app_server:

.. code-block:: json

   {
      "format_version": 2,
      "workflows": [
         {
               "id": "controller",
               "path": "nvflare.app_common.workflows.fedavg.FedAvg",
               "args": {
                  "num_clients": 2,
                  "num_rounds": 3
               }
         }
      ],
      "components": [
         {
               "id": "json_generator",
               "path": "nvflare.app_common.widgets.validation_json_generator.ValidationJsonGenerator",
               "args": {}
         },
         {
               "id": "model_selector",
               "path": "nvflare.app_common.widgets.intime_model_selector.IntimeModelSelector",
               "args": {
                  "aggregation_weights": {},
                  "key_metric": "accuracy"
               }
         },
         {
               "id": "persistor",
               "path": "nvflare.app_opt.tf.model_persistor.TFModelPersistor",
               "args": {
                  "model": {
                     "path": "src.tf_net.TFNet",
                     "args": {}
                  }
               }
         }
      ],
      "task_data_filters": [],
      "task_result_filters": []
   }

This is automatically created by the Job API. The server application configuration leverages NVIDIA FLARE built-in components.

Note that ``persistor`` points to ``TFModelPersistor``. This is automatically configured when the model is added
to the server with the :func:`to<nvflare.job_config.api.FedJob.to>` function. The Job API detects that the model is a TensorFlow model
and automatically configures :class:`TFModelPersistor<nvflare.app_opt.tf.model_persistor.TFModelPersistor>`.


Client Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

The client configuration is ``config_fed_client.json`` in the config folder of each client app folder:

.. code-block:: json

   {
      "format_version": 2,
      "executors": [
         {
               "tasks": [
                  "*"
               ],
               "executor": {
                  "path": "nvflare.app_common.executors.script_executor.ScriptExecutor",
                  "args": {
                     "task_script_path": "src/hello-tf_fl.py"
                  }
               }
         }
      ],
      "components": [
         {
               "id": "event_to_fed",
               "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
               "args": {
                  "events_to_convert": [
                     "analytix_log_stats"
                  ]
               }
         }
      ],
      "task_data_filters": [],
      "task_result_filters": []
   }

The ``task_script_path`` is set to the path of the client training script.

The full source code for this exercise can be found in
:github_nvflare_link:`examples/hello-tf <examples/hello-world/hello-tf>`.

Previous Versions of Hello TensorFlow (previously Hello TensorFlow 2)
---------------------------------------------------------------------

   - `hello-tf2 for 2.0 <https://github.com/NVIDIA/NVFlare/tree/2.0/examples/hello-tf2>`_
   - `hello-tf2 for 2.1 <https://github.com/NVIDIA/NVFlare/tree/2.1/examples/hello-tf2>`_
   - `hello-tf2 for 2.2 <https://github.com/NVIDIA/NVFlare/tree/2.2/examples/hello-tf2>`_
   - `hello-tf2 for 2.3 <https://github.com/NVIDIA/NVFlare/tree/2.3/examples/hello-world/hello-tf2>`_
   - `hello-tf2 for 2.4 <https://github.com/NVIDIA/NVFlare/tree/2.4/examples/hello-world/hello-tf2>`_
