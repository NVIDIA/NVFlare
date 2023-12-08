.. _client_api:

##########
Client API
##########

The FLARE Client API provides an easy way for users to convert their centralized, local
training code into federated learning code with the following benefits:

* Only requires a few lines of code changes, without the need to restructure the code or implement a new class
* Reduces the number of new FLARE specific concepts exposed to users
* Easy adaptation from existing local training code using different frameworks (PyTorch, PyTorch Lightning, HuggingFace)

Core concept
============

Federated learning's concept is for each participating site to get a good model (better than
locally trained model) without sharing the data.

It is done by sharing model parameters or parameter differences (certain filters can be used to
ensure privacy-preserving and protects against gradient inversion attacks) to each other.

The aggregators will take in all these model parameters submitted by each site and produce a
new global model.

We hope that this new global model will be better than locally trained model since it
conceptually trained on more data.

One of the popular federated learning workflow, "FedAvg" is like this:

The general structure of Federated Learning algorithms involve the following steps:

#. controller site initializes an initial model
#. For each round (global iteration):

   #. controller sends the global model to clients
   #. each client starts with this global model and trains on their own data
   #. each client sends back their trained model
   #. controller aggregates all the models and produces a new global model

On the client side, the training workflow is:

#. receive model from controller
#. perform local training on received model, evaluate global model for model selection
#. send new model back to controller

To be able to support different training frameworks, we define a standard data structure called "FLModel"
for the local training code to exchange information with the FLARE system.

We explain its attributes below:

.. literalinclude:: ../../../nvflare/app_common/abstract/fl_model.py
   :language: python
   :lines: 41-67
   :linenos:
   :caption: fl_model.py

Users only need to obtain the required information from this received FLModel,
run local training, and put the results in a new FLModel to send back to the controller.

For a general use case, there are three essential methods for the Client API:

* `init()`: Initializes NVFlare Client API environment.
* `receive()`: Receives model from NVFlare side.
* `send()`: Sends the model to NVFlare side.

Users can use these APIs to change their centralized training code to federated learning, for example:

.. code-block:: python

    import nvflare.client as flare

    flare.init()
    input_model = flare.receive()
    new_params = local_train(input_model.params)
    output_model = flare.FLModel(params=new_params)
    flare.send(output_model)

See below for more in-depth information about all of the Client API functionalities.

Client API Module
=================

nvflare.client.init
-------------------

- Description: initialize required environment variables for NVFlare ML2FL client API
- Arguments:

  - config (str or dict): the path to the config file or the config dictionary
  - rank (str): local rank of the process. It is only useful when the training script has multiple worker processes. (for example multi GPU)

- Returns: None

Usage:

``nvflare.client.init(config="./config.json")``

Config example:

.. code-block:: json

   {
      "exchange_path": "./",
      "exchange_format": "pytorch"
      "transfer_type" : "FULL"
   }

Exchange_path is the file path where the model will be exchanged.
Exchange_format is the format we expect of the model, pre-defined ones are "raw", "numpy", "pytorch"
Transfer_type is how to transfer the model, FULL means send it as it is, DIFF means calculate the difference between new model VS initial received model

nvflare.client.receive
----------------------
- Description: receive FLModel from NVFlare side
- Arguments:

  - Timeout (Optional[float]): timeout to receive an FLModel

- Returns: FLModel

Usage:

``model = nvflare.client.receive()``

nvflare.client.send
-------------------

- Description: send back the FLModel to NVFlare side
- Arguments:

  - fl_model (FLModel): FLModel to be sent
  - clear_registry (bool): whether to clear the model registry after send

- Returns: None

Usage:

``nvflare.client.send(model=FLModel(xxx))``


nvflare.client.system_info
--------------------------

- Description: gets system's metadata
- Arguments: None
- Returns: A dictionary of system's metadata

Usage:

``sys_info = nvflare.client.system_info()``

System's metadata includes:

- identity
- Job_id

nvflare.client.get_job_id
-------------------------

- Description: gets the NVFlare job id
- Arguments: None
- Returns: JOB_ID (str)

Usage:

``job_id = nvflare.client.get_job_id()``

nvflare.client.get_identity
---------------------------
- Description: gets the NVFlare site name that this process is running on
- Arguments: None
- Returns: identity (str)

Usage:

``identity = nvflare.client.get_identity()``

nvflare.client.clear
--------------------

- Description: clears the model registry
- Arguments: None
- Returns: None

Usage:

``nvflare.client.clear()``

nvflare.client.get_config
-------------------------

- Description: gets the model registry config
- Arguments: None
- Returns: identity (dict)

Usage:

``config = nvflare.client.get_config()``

nvflare.client.is_running
-------------------------

- Description: check if FLARE job is still running in the case of launching once
- Arguments: None
- Returns: bool

Usage:

.. code-block:: python

   while nvflare.client.is_running():
      # receive model, perform task, send model, etc.

nvflare.client.is_train
-----------------------

- Description: check if current task is train
- Arguments: None
- Returns: bool

Usage:

.. code-block:: python

   if nvflare.client.is_train():
      # perform train task on received model

nvflare.client.is_evaluate()
----------------------------

- Description: check if current task is evaluate
- Arguments: None
- Returns: bool

Usage:

.. code-block:: python

   if nvflare.client.is_evaluate():
      # perform evaluate task on received model

nvflare.client.is_submit_model()
--------------------------------

- Description: check if current task is submit_model
- Arguments: None
- Returns: bool

Usage:

.. code-block:: python

   if nvflare.client.is_submit_model():
      # perform submit_model task to obtain best local model

Client Decorator Module
=======================
nvflare.client.train
--------------------

Use cases:

.. code-block:: python

   @nvflare.client.train
   def my_train(input_model=None, device="cuda:0"):
      ...
      return new_model

NVFlare will pass the FLModel received from the NVFlare server side to the first argument of the "decorated" method.
The return value needs to be an FLModel object, we will send it directly to the NVFlare server side.


nvflare.client.evaluate
-----------------------

Use cases:

.. code-block:: python

   @nvflare.client.evaluate
   def my_eval(input_model, device="cuda:0"):
      ...
      return metrics

NVFlare will pass the model received from the NVFlare server side to the first argument of the "decorated" method.
The return value needs to be a "float" metric.
The decorated "my_eval" method needs to be run BEFORE the training method, so the metrics will be sent along with the trained output model.

Lightning Integration
=====================
nvflare.client.lightning.patch
------------------------------

- Description: patch the PyTorch Lightning Trainer object
- Arguments: trainer
- Returns: None

Usage:

.. code-block:: python

   trainer = Trainer(max_epochs=1)
   flare.patch(trainer)

Advanced Usage:

Note that if users want to pass additional information to NVFlare server side VIA the lightning API, they will need to set the information inside the attributes called ``__fl_meta__`` in their LightningModule. For example:

.. code-block:: python

   class LitNet(LightningModule):
      def __init__(self):
         super().__init__()
         self.save_hyperparameters()
         self.model = Net()
         self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
         self.valid_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
         self.__fl_meta__ = {"CUSTOM_VAR": "VALUE_OF_THE_VAR"}

Configuration and Installation
==============================

In the client_config.json, in order to launch the training script we use the :class:`SubprocessLauncher<nvflare.app_common.launchers.subprocess_launcher.SubprocessLauncher>` component.
The defined ``script`` is invoked, and ``launch_once`` can be set to either launch once for the whole job, or launch a process for each task received from the server.

A corresponding :class:`LauncherExecutor<nvflare.app_common.executors.launcher_executor.LauncherExecutor>` is used as the executor to handle the tasks and peform the data exchange using the pipe.
For the Pipe component we provide implementations of :class:`FilePipe<nvflare.fuel.utils.pipe.file_pipe>` and :class:`CellPipe<nvflare.fuel.utils.pipe.cell_pipe>`.

.. literalinclude:: ../../../job_templates/sag_pt/config_fed_client.conf
    :language: json

For example configurations, take a look at the :github_nvflare_link:`job_templates <job_templates>` directory for templates using the launcher and Client API.

.. note::
   In that case that the user does not need to launch the process via the SubprocessLauncher and instead has their own external training system, this would involve using
   the :ref:`3rd_party_integration`, which is based on the same underlying mechanisms.
   Rather than a LauncherExecutor, the parent class :class:`TaskExchanger<nvflare.app_common.executors.task_exchanger>` would be used to handle the tasks and enable pipe data exchange.
   Additionally, the :class:`FlareAgent<nvflare.client.flare_agent>` would be used to communicate with the Flare Client Job Cell to get the tasks and submit the result.

Examples
========

For examples of using Client API with different frameworks,
please refer to :github_nvflare_link:`examples/hello-world/ml-to-fl <examples/hello-world/ml-to-fl>`.

For additional examples, also take a look at the :github_nvflare_link:`step-by-step series <examples/hello-world/step-by-step>`
that use a :github_nvflare_link:`Client API trainer <examples/hello-world/step-by-step/cifar10/code/fl/train.py>`.
