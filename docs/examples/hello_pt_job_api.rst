.. _hello_pt_job_api:

Hello PyTorch with Job API
==========================

Before You Start
----------------

Feel free to refer to the :doc:`detailed documentation <../programming_guide>` at any point
to learn more about the specifics of `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_.

We recommend you first finish the :doc:`Hello FedAvg with NumPy <hello_fedavg_numpy>` exercise since it introduces the
federated learning concepts of `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_.

Make sure you have an environment with NVIDIA FLARE installed.

You can follow :ref:`getting_started` on the general concept of setting up a
Python virtual environment (the recommended environment) and how to install NVIDIA FLARE.

Introduction
-------------

Through this exercise, you will integrate NVIDIA FLARE with the popular
deep learning framework `PyTorch <https://pytorch.org/>`_ and learn how to use NVIDIA FLARE to train a convolutional
network with the CIFAR10 dataset using the included :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>` workflow.

The setup of this exercise consists of one **server** and two **clients**.

The following steps compose one cycle of weight updates, called a **round**:

 #. Clients are responsible for generating individual weight-updates for the model using their own CIFAR10 dataset. 
 #. These updates are then sent to the server which will aggregate them to produce a model with new weights. 
 #. Finally, the server sends this updated version of the model back to each client.

For this exercise, we will be working with the ``hello-pt`` application in the examples folder.

Let's get started. First clone the repo:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Remember to activate your NVIDIA FLARE Python virtual environment from the installation guide.

Since you will use PyTorch and torchvision for this exercise, let's go ahead and install both libraries: 

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install torch torchvision

If you would like to go ahead and run the exercise now, you can run the ``fedavg_script_executor_hello-pt.py`` script which
builds the job with the Job API and runs the job with the FLARE Simulator.

NVIDIA FLARE Job API
--------------------

The ``fedavg_script_executor_hello-pt.py`` script for this hello-pt example is very similar to the ``fedavg_script_executor_hello-numpy.py`` script
for the :doc:`Hello FedAvg with NumPy <hello_fedavg_numpy>` exercise. Other than changes to the names of the job and client script, the only difference
is a line to define the initial global model for the server:

.. code-block:: python

   # Define the initial global model and send to server
   job.to(SimpleNetwork(), "server")


NVIDIA FLARE Client Training Script
------------------------------------
The training script for this example, ``hello-pt_cifar10_fl.py``, is the main script that will be run on the clients. It contains the PyTorch specific
logic for training.

Neural Network
^^^^^^^^^^^^^^^

The training procedure and network architecture are modified from 
`Training a Classifier <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_.

Let's see what an extremely simplified CIFAR10 training looks like:

.. literalinclude:: ../../examples/hello-world/hello-pt/src/simple_network.py
   :language: python
   :caption: simple_network.py

This ``SimpleNetwork`` class is your convolutional neural network to train with the CIFAR10 dataset.
This is not related to NVIDIA FLARE, so we implement it in a file called ``simple_network.py``.

Dataset & Setup
^^^^^^^^^^^^^^^^

In a real FL experiment, each client would have their own dataset used for their local training.
You can download the CIFAR10 dataset from the Internet via torchvision's datasets module, so for simplicity's sake, this is
the dataset we will be using on each client.
Additionally, you need to set up the optimizer, loss function and transform to process the data.
You can think of all of this code as part of your local training loop, as every deep learning training has a similar setup.

In the ``hello-pt_cifar10_fl.py`` script, we take care of all of this setup before the ``flare.init()``.

Local Train
^^^^^^^^^^^

Now with the network and dataset setup, let's also implement the local training loop with the NVFlare's Client API:

.. code-block:: python

   flare.init()

   summary_writer = SummaryWriter()
   while flare.is_running():
      input_model = flare.receive()

      model.load_state_dict(input_model.params)

      steps = epochs * len(train_loader)
      for epoch in range(epochs):
         running_loss = 0.0
         for i, batch in enumerate(train_loader):
               images, labels = batch[0].to(device), batch[1].to(device)
               optimizer.zero_grad()

               predictions = model(images)
               cost = loss(predictions, labels)
               cost.backward()
               optimizer.step()

               running_loss += cost.cpu().detach().numpy() / images.size()[0]

      output_model = flare.FLModel(params=model.cpu().state_dict(), meta={"NUM_STEPS_CURRENT_ROUND": steps})
      
      flare.send(output_model)


The code above is simplified from the ``hello-pt_cifar10_fl.py`` script to focus on the three essential methods of the NVFlare's Client API to
achieve the training workflow:

   - `init()`: Initializes NVFlare Client API environment.
   - `receive()`: Receives model from the FL server.
   - `send()`: Sends the model to the FL server.

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
                  "num_rounds": 2
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
               "id": "receiver",
               "path": "nvflare.app_opt.tracking.tb.tb_receiver.TBAnalyticsReceiver",
               "args": {
                  "events": [
                     "fed.analytix_log_stats"
                  ]
               }
         },
         {
               "id": "persistor",
               "path": "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor",
               "args": {
                  "model": {
                     "path": "src.simple_network.SimpleNetwork",
                     "args": {}
                  }
               }
         },
         {
               "id": "model_locator",
               "path": "nvflare.app_opt.pt.file_model_locator.PTFileModelLocator",
               "args": {
                  "pt_persistor_id": "persistor"
               }
         }
      ],
      "task_data_filters": [],
      "task_result_filters": []
   }

This is automatically created by the Job API. The server application configuration leverages NVIDIA FLARE built-in components.

Note that ``persistor`` points to ``PTFileModelPersistor``. This is automatically configured when the model SimpleNetwork is added
to the server with the :func:`to<nvflare.job_config.api.FedJob.to>` function. The Job API detects that the model is a PyTorch model
and automatically configures :class:`PTFileModelPersistor<nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor>`
and :class:`PTFileModelLocator<nvflare.app_opt.pt.file_model_locator.PTFileModelLocator>`.


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
                     "task_script_path": "src/hello-pt_cifar10_fl.py"
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
:github_nvflare_link:`examples/hello-world/hello-pt <examples/hello-world/hello-pt/>`.

Previous Versions of Hello PyTorch
----------------------------------

   - `hello-pt for 2.0 <https://github.com/NVIDIA/NVFlare/tree/2.0/examples/hello-pt>`_
   - `hello-pt for 2.1 <https://github.com/NVIDIA/NVFlare/tree/2.1/examples/hello-pt>`_
   - `hello-pt for 2.2 <https://github.com/NVIDIA/NVFlare/tree/2.2/examples/hello-pt>`_
   - `hello-pt for 2.3 <https://github.com/NVIDIA/NVFlare/tree/2.3/examples/hello-world/hello-pt>`_
   - `hello-pt for 2.4 <https://github.com/NVIDIA/NVFlare/tree/2.4/examples/hello-world/hello-pt>`_
