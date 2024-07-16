.. _fed_job_api:

##########
FedJob API
##########

The FLARE :class:`FedJob<nvflare.job_config.fed_job.FedJob>` API allows users to Pythonically define and create job configurations.

Core Concepts
=============

* Use the :func:`to<nvflare.job_config.fed_job.FedJob.to>` routine to assign objects (e.g. controllers, executor, models, filters, components etc.) to the server or clients.
* Export the job to a configuration with :func:`export_job<nvflare.job_config.fed_job.FedJob.export_job>`.
* Run the job in the simulator with :func:`simulator_run<nvflare.job_config.fed_job.FedJob.simulator_run>`.

Table overview of the :class:`FedJob<nvflare.job_config.fed_job.FedJob>` API:

.. list-table:: FedJob
   :widths: 25 35 50
   :header-rows: 1

   * - API
     - Description
     - API Doc Link
   * - to
     - Assign object to target.
     - :func:`to<nvflare.job_config.fed_job.FedJob.to>`
   * - to_server
     - Assign object to server.
     - :func:`to_server<nvflare.job_config.fed_job.FedJob.to_server>`
   * - to_clients
     - Assign object to all clients.
     - :func:`to_clients<nvflare.job_config.fed_job.FedJob.to_clients>`
   * - as_id
     - Return generated uuid of object. Object will be added as component if referenced.
     - :func:`as_id<nvflare.job_config.fed_job.FedJob.as_id>`
   * - simulator_run
     - Run the job with the simulator.
     - :func:`simulator_run<nvflare.job_config.fed_job.FedJob.simulator_run>`
   * - export_job
     - Export the job configuration.
     - :func:`export_job<nvflare.job_config.fed_job.FedJob.export_job>`


Here is an example of how to create a simple cifar10_fedavg job using the :class:`FedJob<nvflare.job_config.fed_job.FedJob>` API.
We assign a FedAvg controller and the initial PyTorch model to the server, and assign a ScriptExecutor for our training script to the clients.
Then we use the simulator to run the job:

.. code-block:: python

  from src.net import Net

  from nvflare import FedAvg, FedJob, ScriptExecutor

  if __name__ == "__main__":
      n_clients = 2
      num_rounds = 2
      train_script = "src/cifar10_fl.py"

      job = FedJob(name="cifar10_fedavg")

      # Define the controller workflow and send to server
      controller = FedAvg(
          num_clients=n_clients,
          num_rounds=num_rounds,
      )
      job.to_server(controller)

      # Define the initial global model and send to server
      job.to_server(Net())

      # Send executor to all clients
      executor = ScriptExecutor(
          task_script_path=train_script, task_script_args=""  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
      )
      job.to_clients(executor)

      # job.export_job("/tmp/nvflare/jobs/job_config")
      job.simulator_run("/tmp/nvflare/jobs/workdir", n_clients=n_clients)


Initializing the FedJob
=======================

Initialize the :class:`FedJob<nvflare.job_config.fed_job.FedJob>` object with the following arguments:

* ``name`` (str): for job name.
* ``min_clients`` (int): required for the job, will be set in the ``meta.json``.
* ``mandatory_clients`` (List[str]): to run the job, will be set in the ``meta.json``.
* ``key_metric`` (str): the metric used for global model selection, will be used by the preconfigured :class:`IntimeModelSelector<nvflare.app_common.widgets.intime_model_selector.IntimeModelSelector>`.

Example:

.. code-block:: python

  job = FedJob(name="cifar10_fedavg", min_clients=2, mandatory_clients=["site-1", "site-2"], key_metric="accuracy")

Assigning objects with :func:`to<nvflare.job_config.fed_job.FedJob.to>`
=======================================================================

Assign objects with :func:`to<nvflare.job_config.fed_job.FedJob.to>` for a specific ``target``,
:func:`to_server<nvflare.job_config.fed_job.FedJob.to_server>` for the server, and
:func:`to_clients<nvflare.job_config.fed_job.FedJob.to_clients>` for all the clients.

These functions have the following parameters which are used depending on the type of object:

* ``obj`` (any): The object to be assigned. The obj will be given a default id if non is provided based on its type.
* ``target`` (str): (For :func:`to<nvflare.job_config.fed_job.FedJob.to>`) The target location of the object. Can be “server” or a client name, e.g. “site-1”.
* ``tasks`` (List[str]): If object is an Executor or Filter, optional list of tasks that should be handled. Defaults to None. If None, all tasks will be handled using [*].
* ``gpu`` (int | List[int]): GPU index or list of GPU indices used for simulating the run on that target.
* ``filter_type`` (FilterType): The type of filter used. Either FilterType.TASK_RESULT or FilterType.TASK_DATA.
* ``id`` (int): Optional user-defined id for the object. Defaults to None and ID will automatically be assigned.

Below we cover in-depth how different types of objects are handled when using :func:`to<nvflare.job_config.fed_job.FedJob.to>`:

Controller
----------

If the object is a :class:`Controller<nvflare.apis.impl.controller.Controller>`, the controller is added to the server app workflows.

* If the ``key_metric`` is defined in the FedJob (see initialization), an :class:`IntimeModelSelector<nvflare.app_common.widgets.intime_model_selector.IntimeModelSelector>` widget will be added for best model selection.
* A :class:`ValidationJsonGenerator<nvflare.app_common.widgets.validation_json_generator.ValidationJsonGenerator>` is automatically added for creating json validation results.
* If PyTorch and TensorBoard are supported, then :class:`TBAnalyticsReceiver<nvflare.app_common.pt.tb_receiver.TBAnalyticsReceiver>` is automatically added to receives analytics data to save to TensorBoard. Other types of receivers can be added as components with :func:`to<nvflare.job_config.fed_job.FedJob.to>`.

Example:

.. code-block:: python

  controller = FedAvg(
      num_clients=n_clients,
      num_rounds=num_rounds,
  )
  job.to(controller, "server")


Executor
--------

If the object is an :class:`Executor<nvflare.apis.executor.Executor>`, the executor is added to the client app executors.

* The ``tasks`` parameter specifies the tasks that the executor is defined the handle.
* The ``gpu`` parameter specifies which gpus to use for simulating the run on the target.
* If the object is a :class:`ScriptExecutor<nvflare.app_common.executors.script_executor.ScriptExecutor>`, the task_script_path will be added to the external scripts to be included in the custom directory.
* The :class:`ConvertToFedEvent<nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent>` widget is automatically added to convert local events to federated events.

Example:

.. code-block:: python

  executor = ScriptExecutor(task_script_path="src/cifar10_fl.py", task_script_args="")
  job.to(executor, "site-1", tasks=["train"], gpu=0)


Script (str)
------------

If the object is a str, it is treated as an external script and will be included in the custom directory.

Example:

.. code-block:: python

  job.to("src/cifar10_fl.py", "site-1")


Filter
------

If the object is a :class:`Filter<nvflare.apis.filter.Filter>`, users must specify the ``filter_type``
as either FilterType.TASK_RESULT (flow from executor to controller) or FilterType.TASK_DATA (flow from controller to executor).

The filter will be added task_data_filters and task_result_filters accordingly and be applied to the specified ``tasks``.

Example:

.. code-block:: python

  pp_filter = PercentilePrivacy(percentile=10, gamma=0.01)
  job.to(pp_filter, "site-1", tasks=["train"], filter_type=FilterType.TASK_RESULT)


Model
-----
If the object is a common model type, a corresponding persistor will automatically be configured with the model.

For PyTorch models (``torch.nn.Module``) we add a :class:`PTFileModelPersistor<nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor>` and
:class:`PTFileModelLocator<nvflare.app_opt.pt.file_model_locator.PTFileModelLocator>`, and for TensorFlow models (``tf.keras.Model``) we add a :class:`TFModelPersistor<nvflare.app_opt.tf.model_persistor.TFModelPersistor>`.

Example:

.. code-block:: python

  job.to(Net(), "server")

For unsupported models, the model and persistor can be added as components.


Components
----------
For any object that does not fall under any of the previous types, it is added as a component with ``id``.
The ``id`` can be either specified as a parameter, or it will be automatically assigned.Components may reference other components by id

If an id generated by :func:`as_id<nvflare.job_config.fed_job.FedJob.as_id>`, is referenced by another added object, this the referenced object will also be added as a component.
In the example below, comp2 is assigned to the server. Since comp1 was referenced in comp2 with :func:`as_id<nvflare.job_config.fed_job.FedJob.as_id>`, comp1 will also be added as a component to the server.

Example:

.. code-block:: python

  comp1 = Component1()
  comp2 = Component2(sub_component_id=job.as_id(comp1))
  job.to(comp2, "server")


Running the Job
===============

Simulator
---------

Run the FedJob with the simulator with :func:`simulator_run<nvflare.job_config.fed_job.FedJob.simulator_run>` in the ``workspace`` with ``n_clients`` and ``threads``.
(Note: only set ``n_clients`` if you have not specified clients using :func:`to<nvflare.job_config.fed_job.FedJob.to>`)

Example:

.. code-block:: python

  job.simulator_run(workspace="/tmp/nvflare/jobs/workdir", n_clients=2, threads=2)


Export Configuration
--------------------
We can export the job configuration with :func:`export_job<nvflare.job_config.fed_job.FedJob.export_job>` to the ``job_root`` directory.

Example:

.. code-block:: python

  job.export_job(job_root="/tmp/nvflare/jobs/job_config")

Examples
========

To see examples of how the FedJob API can be used for different applications, refer the :github_nvflare_link:`Getting Started <examples/getting_started>` examples.
