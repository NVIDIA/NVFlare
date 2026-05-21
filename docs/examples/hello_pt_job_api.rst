.. _hello_pt_job_api:

Hello PyTorch with Job API
==========================

This example demonstrates how to use NVIDIA FLARE with PyTorch to train an image classifier using federated averaging (FedAvg).
The complete example code can be found in the :github_nvflare_link:`hello-pt directory <examples/hello-world/hello-pt/>`.

Before You Start
----------------

Feel free to refer to the :doc:`detailed documentation <../developer_guide>` at any point
to learn more about the specifics of `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_.

We recommend you first finish the :doc:`Hello NumPy <hello_numpy>` exercise since it introduces the
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

Running the Example
-------------------
To run this example:

1. Clone the repository and navigate to the example directory:

.. code-block:: shell

   $ git clone https://github.com/NVIDIA/NVFlare.git
   $ cd NVFlare/examples/hello-world/hello-pt

2. Install the required dependencies:

.. code-block:: shell

   $ pip install -r requirements.txt

3. Run the example:

.. code-block:: shell

   $ python job.py

The script creates an NVFlare job recipe and runs it using the FL Simulator.

To export the job folder for submission to a running FL system, use the standard Recipe API export flags:

.. code-block:: shell

   $ python job.py --export --export-dir /tmp/nvflare/jobs/job_config

The exported job is written to ``/tmp/nvflare/jobs/job_config/hello-pt``.
You can combine the export flags with example-specific options, for example:

.. code-block:: shell

   $ python job.py --export --export-dir /tmp/nvflare/jobs/job_config \
       --enable_log_streaming --synthetic_data --train_size 2048 --test_size 256 \
       --num_rounds 2 --epochs 1 --batch_size 64 --num_workers 0

NVIDIA FLARE Job API
--------------------

The ``job.py`` script for this hello-pt example defines a :class:`FedAvgRecipe<nvflare.app_opt.pt.recipes.fedavg.FedAvgRecipe>`.
The recipe combines the PyTorch model, client training script, and simulator/export behavior:

.. code-block:: python

   recipe = FedAvgRecipe(
       name="hello-pt",
       min_clients=n_clients,
       num_rounds=num_rounds,
       model=SimpleNetwork(),
       train_script="client.py",
       train_args=train_args,
   )


NVIDIA FLARE Client Training Script
------------------------------------
The training script for this example, ``client.py``, is the main script that will be run on the clients. It contains the PyTorch specific
logic for training.

Neural Network
^^^^^^^^^^^^^^^

The training procedure and network architecture are modified from 
`Training a Classifier <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_.

Let's see the simplified CIFAR10 model used in this example:

- :github_nvflare_link:`model.py <examples/hello-world/hello-pt/model.py>`

This ``SimpleNetwork`` class is your convolutional neural network to train with the CIFAR10 dataset.
This is not related to NVIDIA FLARE, so we implement it in a file called ``model.py``.

Dataset & Setup
^^^^^^^^^^^^^^^^

In a real FL experiment, each client would have their own dataset used for their local training.
You can download the CIFAR10 dataset from the Internet via torchvision's datasets module, so for simplicity's sake, this is
the dataset we will be using on each client.
Additionally, you need to set up the optimizer, loss function and transform to process the data.
You can think of all of this code as part of your local training loop, as every deep learning training has a similar setup.

In the ``client.py`` script, we take care of all of this setup before the ``flare.init()``.

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


The code above is simplified from the ``client.py`` script to focus on the three essential methods of the NVFlare's Client API to
achieve the training workflow:

   - `init()`: Initializes NVFlare Client API environment.
   - `receive()`: Receives model from the FL server.
   - `send()`: Sends the model to the FL server.

NVIDIA FLARE Server & Application
---------------------------------
In this example, the server runs :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>` with the default settings.

If you export the job with ``python job.py --export --export-dir <job_folder>``, you will see the
configurations for the server and each client. The server configuration is ``config_fed_server.json`` in the config folder
in the exported app folder:

.. code-block:: json

   {
      "format_version": 2,
      "workflows": [
         {
               "id": "controller",
               "path": "nvflare.app_common.workflows.fedavg.FedAvg",
               "args": {
                  "aggregation_weights": {},
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
                     "analytix_log_stats",
                     "fed.analytix_log_stats"
                  ]
               }
         },
         {
               "id": "persistor",
               "path": "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor",
               "args": {
                  "model": {
                     "path": "model.SimpleNetwork",
                     "args": {}
                  }
               }
         },
         {
               "id": "locator",
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

Note that ``persistor`` points to ``PTFileModelPersistor``. This is automatically configured from the
``SimpleNetwork`` model supplied to the recipe. The Job API detects that the model is a PyTorch model
and automatically configures :class:`PTFileModelPersistor<nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor>`
and :class:`PTFileModelLocator<nvflare.app_opt.pt.file_model_locator.PTFileModelLocator>`.


Client Configuration
^^^^^^^^^^^^^^^^^^^^

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
               "path": "nvflare.app_opt.pt.in_process_client_api_executor.PTInProcessClientAPIExecutor",
               "args": {
                  "task_script_path": "client.py",
                  "task_script_args": "--batch_size 16 --epochs 2 --num_workers 2"
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
   - `hello-pt for 2.5 <https://github.com/NVIDIA/NVFlare/tree/2.5/examples/hello-world/hello-pt>`_
