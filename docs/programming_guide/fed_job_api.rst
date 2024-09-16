.. _fed_job_api:

##########
FedJob API
##########

The FLARE :class:`FedJob<nvflare.job_config.api.FedJob>` API allows users to Pythonically define and create job configurations.

Core Concepts
=============

* Use the :func:`to<nvflare.job_config.api.FedJob.to>` routine to assign objects (e.g. Controller, ScriptRunner, Executor, PTModel, Filters, Components etc.) to the server or clients.
* Objects can define how they are added to the job by implementing ``add_to_fed_job``, otherwise they are added as components.
* Export the job to a configuration with :func:`export_job<nvflare.job_config.api.FedJob.export_job>`.
* Run the job in the simulator with :func:`simulator_run<nvflare.job_config.api.FedJob.simulator_run>`.

Table overview of the :class:`FedJob<nvflare.job_config.api.FedJob>` API:

.. list-table:: FedJob API
   :widths: 25 35 50
   :header-rows: 1

   * - API
     - Description
     - API Doc Link
   * - to
     - Assign object to target.
     - :func:`to<nvflare.job_config.api.FedJob.to>`
   * - to_server
     - Assign object to server.
     - :func:`to_server<nvflare.job_config.api.FedJob.to_server>`
   * - to_clients
     - Assign object to all clients.
     - :func:`to_clients<nvflare.job_config.api.FedJob.to_clients>`
   * - set_up_client
     - To be used in FedJob subclasses. Setup routine called by FedJob when first sending object to a client target.
     - :func:`set_up_client<nvflare.job_config.api.FedJob.set_up_client>`
   * - as_id
     - Return generated uuid of object. Object will be added as component if referenced.
     - :func:`as_id<nvflare.job_config.api.FedJob.as_id>`
   * - simulator_run
     - Run the job with the simulator.
     - :func:`simulator_run<nvflare.job_config.api.FedJob.simulator_run>`
   * - export_job
     - Export the job configuration.
     - :func:`export_job<nvflare.job_config.api.FedJob.export_job>`


Here is an example of how to create a simple cifar10_fedavg job using the :class:`FedJob<nvflare.job_config.api.FedJob>` API.
We assign a FedAvg controller and the initial PyTorch model to the server, and assign a ScriptExecutor for our training script to the clients.
Then we use the simulator to run the job:

.. code-block:: python

  from src.net import Net

  from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
  from nvflare.app_common.workflows.fedavg import FedAvg
  from nvflare.app_opt.pt.job_config.model import PTModel

  from nvflare.job_config.api import FedJob
  from nvflare.job_config.script_runner import ScriptRunner

  if __name__ == "__main__":
      n_clients = 2
      num_rounds = 2
      train_script = "src/cifar10_fl.py"

      # Create the FedJob
      job = FedJob(name="cifar10_fedavg")

      # Define the FedAvg controller workflow and send to server
      controller = FedAvg(
          num_clients=n_clients,
          num_rounds=num_rounds,
      )
      job.to_server(controller)

      # Define the initial global model with PTModel wrapper and send to server
      job.to_server(PTModel(Net()))

      # Add model selection widget and send to server
      job.to_server(IntimeModelSelector(key_metric="accuracy"))

      # Send ScriptRunner to all clients
      runner = ScriptRunner(
          script=train_script, script_args="f--batch_size 32 --data_path /tmp/data/site-{i}"
      )
      job.to_clients(runner)

      # job.export_job("/tmp/nvflare/jobs/job_config")
      job.simulator_run("/tmp/nvflare/jobs/workdir", n_clients=n_clients)


Initializing the FedJob
=======================

Initialize the :class:`FedJob<nvflare.job_config.api.FedJob>` object with the following arguments:

* ``name`` (str): for job name.
* ``min_clients`` (int): required for the job, will be set in the meta.json.
* ``mandatory_clients`` (List[str]): to run the job, will be set in the meta.json.

Example:

.. code-block:: python

  job = FedJob(name="cifar10_fedavg", min_clients=2, mandatory_clients=["site-1", "site-2"])

Assigning objects with :func:`to<nvflare.job_config.api.FedJob.to>`
=====================================================================

Assign objects with :func:`to<nvflare.job_config.api.FedJob.to>` for a specific ``target``,
:func:`to_server<nvflare.job_config.api.FedJob.to_server>` for the server, and
:func:`to_clients<nvflare.job_config.api.FedJob.to_clients>` for all the clients.

These functions have the following parameters which are used depending on the type of object:

* ``obj`` (any): The object to be assigned. The obj will be given a default id if none is provided based on its type.
* ``target`` (str): (For :func:`to<nvflare.job_config.api.FedJob.to>`) The target location of the object. Can be “server” or a client name, e.g. “site-1”.
* ``**kwargs``: if the object implements the ``add_to_fed_job`` method, ``kwargs`` are additional args to be passed to this function. See the specific object's section for more details.

.. note::

    In order for the FedJob to use the values of arguments passed into the ``obj``, the arguments must be set as instance variables of the same name (or prefixed with "_") in the constructor.

Below we cover in-depth how different types of objects are handled when using :func:`to<nvflare.job_config.api.FedJob.to>`:


Controller
----------

If the object is a :class:`Controller<nvflare.apis.impl.controller.Controller>` sent to the server, the controller is added to the server app workflows.

Example:

.. code-block:: python

  controller = FedAvg(
      num_clients=n_clients,
      num_rounds=num_rounds,
  )
  job.to(controller, "server")

If the object is a :class:`Controller<nvflare.apis.impl.controller.Controller>` sent to a client, the controller is added to the client app components as a client-side controller.
The controller can then be used by the :class:`ClientControllerExecutor<nvflare.app_common.ccwf.client_controller_executor.ClientControllerExecutor>`.

ScriptRunner
------------

The :class:`ScriptRunner<nvflare.job_config.script_runner.ScriptRunner>` can be added to clients and is used to run or launch a script.
The ``tasks`` parameter specifies the tasks the script is defined the handle (defaults to "[*]" for all tasks).

ScriptRunner args:

* ``script``: the script to run, will automatically be added to the custom folder.
* ``script_args``: arguments appended to the end of script.
* ``launch_external_process``: two modes, default in-process (launch_external_process=False) and ex-process (launch_external_process=True).
* ``command``: in the ex-process mode, command is prepended to the script (defaults to "python3").
* ``framework``: determines what :class:`FrameworkType<nvflare.job_config.script_runner.FrameworkType>` to use for the script.


Example:

.. code-block:: python

  # in-process: runs `__main__` of "src/cifar10_fl.py" with argv "--batch_size 32"
  in_process_runner = ScriptRunner(
      script="src/cifar10_fl.py",
      script_args="--batch_size 32"
  )
  job.to(in_process_runner, "site-1", tasks=["train"])

  # subprocess: runs `python3 -u custom/src/cifar10_fl.py --batch_size 32`
  external_process_runner = ScriptRunner(
      script="src/cifar10_fl.py",
      script_args="--batch_size 32",
      launch_external_process=True,
      command="python3 -u"
  )
  job.to(external_process_runner, "site-2", tasks=["train"])


For more details on how the ScriptRunner internally configures the ``InProcessClientAPIExecutor`` or ``ClientAPILauncherExecutor``, refer to its
:func:`add_to_fed_job<nvflare.job_config.script_runner.ScriptRunner.add_to_fed_job>` implementation.
A dictionary of component ids added is also returned to be used if needed.


Executor
--------

If the object is an :class:`Executor<nvflare.apis.executor.Executor>`, it must be sent to a client. The executor is added to the client app executors.
The ``tasks`` parameter specifies the tasks that the executor is defined the handle (defaults to "[*]" for all tasks).

Example:

.. code-block:: python

  executor = MyExecutor()
  job.to(executor, "site-1", tasks=["train"])


Resource (str)
--------------

If the object is a str, it is treated as an external resource and will be included in the custom directory.

* If the object is a script, it will be copied to the custom directory.
* If the object is a directory, the directory will be copied flat to the custom directory.

Example:

.. code-block:: python

  job.to("src/cifar10_fl.py", "site-1") # script
  job.to("content_dir", "site-1") # directory


Filter
------

If the object is a :class:`Filter<nvflare.apis.filter.Filter>`,

* Users must specify the ``filter_type`` as either FilterType.TASK_RESULT (flow from executor to controller) or FilterType.TASK_DATA (flow from controller to executor).
* The filter will be added task_data_filters and task_result_filters accordingly and be applied to the specified ``tasks`` (defaults to "[*]" for all tasks).

Example:

.. code-block:: python

  pp_filter = PercentilePrivacy(percentile=10, gamma=0.01)
  job.to(pp_filter, "site-1", tasks=["train"], filter_type=FilterType.TASK_RESULT)


Model Wrappers
--------------

Model Wrappers :class:`PTModel<nvflare.app_opt.pt.job_config.model.PTModel>` and :class:`TFModel<nvflare.app_opt.tf.job_config.model.TFModel>` are used for adding a model with persistor.

* :class:`PTModel<nvflare.app_opt.pt.job_config.model.PTModel>`: for PyTorch models (torch.nn.Module) we add a :class:`PTFileModelPersistor<nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor>` and :class:`PTFileModelLocator<nvflare.app_opt.pt.file_model_locator.PTFileModelLocator>`, and return a dictionary for these added component ids.
* :class:`TFModel<nvflare.app_opt.tf.job_config.model.TFModel>`: for TensorFlow models (tf.keras.Model) we add a :class:`TFModelPersistor<nvflare.app_opt.tf.model_persistor.TFModelPersistor>` and return the added persistor id.

Example:

.. code-block:: python

  component_ids = job.to(PTModel(Net()), "server")

For other types of models, the model and persistor can be added explicitly as components.


Components
----------
For any object that does not fall under any of the previous types and does not implement ``add_to_fed_job``, then it is added as a component with ``id``.

* The ``id`` can be either specified as a parameter, or it will be automatically assigned.
* If adding a component with a previously used id, then the id will be incremented (e.g. "component_id1", "component_id2") and returned.
* Components may reference other components by id.

Example:

.. code-block:: python

  job.to_server(IntimeModelSelector(key_metric="accuracy"))


In the case that an id generated by :func:`as_id<nvflare.job_config.api.FedJob.as_id>`, is referenced by another added object, this the referenced object will also be added as a component.
In the example below, comp2 is assigned to the server. Since comp1 was referenced in comp2 with :func:`as_id<nvflare.job_config.api.FedJob.as_id>`, comp1 will also be added as a component to the server.

Example:

.. code-block:: python

  comp1 = Component1()
  comp2 = Component2(sub_component_id=job.as_id(comp1))
  job.to(comp2, "server")


add_to_fed_job
===============

If the obj implements the ``add_to_fed_job`` method, it will be called with the kwargs. The implementation of add_to_fed_job is specific to the obj being added.
This method must follow this signature:

.. code-block:: python

    add_to_fed_job(job, ctx, ...)

Many of the object types covered in the above sections have implemented add_to_fed_job as they either have special cases or server as wrappers to add additional related components.

As shown in the table below, the Object Developer FedJob API provides functions to add components, Controllers, Executors, Filters, and resources.
The Job Context ``ctx`` should simply be passed to these "add_xxx" methods, and does need to be accessed.
Additionally, the check_kwargs function can check and enforce required arguments in the kwargs.

.. note::

    When adding other components, a good practice is to return the ids of the extra components added in case they might be needed elsewhere.


Example of :class:`TFModel<nvflare.app_opt.tf.job_config.model.TFModel>` :func:`add_to_fed_job<nvflare.app_opt.tf.job_config.model.TFModel.add_to_fed_job>`:

.. code-block:: python

    def add_to_fed_job(self, job, ctx):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:
            dictionary of ids of component added
        """
        if isinstance(self.model, tf.keras.Model):  # if model, create a TF persistor
            persistor = TFModelPersistor(model=self.model)
            persistor_id = job.add_component(comp_id="persistor", obj=persistor, ctx=ctx)
            return persistor_id
        else:
            raise ValueError(
                f"Unable to add {self.model} to job with TFModelPersistor. Expected tf.keras.Model but got {type(self.model)}."
            )


.. list-table:: FedJob Object Developer API
   :widths: 25 35 50
   :header-rows: 1

   * - API
     - Description
     - API Doc Link
   * - add_component
     - Add a component to the job.
     - :func:`add_component<nvflare.job_config.api.FedJob.add_component>`
   * - add_controller
     - Add a Controller object to the job.
     - :func:`add_controller<nvflare.job_config.api.FedJob.add_controller>`
   * - add_executor
     - Add an executor to the job.
     - :func:`add_executor<nvflare.job_config.api.FedJob.add_executor>`
   * - add_filter
     - Add a filter to the job.
     - :func:`add_filter<nvflare.job_config.api.FedJob.add_filter>`
   * - add_resources
     - Add resources to the job.
     - :func:`add_resources<nvflare.job_config.api.FedJob.add_resources>`
   * - check_kwargs
     - Check kwargs for arguments. Raise Error if required arg is missing, or unexpected arg is given.
     - :func:`check_kwargs<nvflare.job_config.api.FedJob.check_kwargs>`


Job Pattern Inheritance
========================

Job inheritance can be useful when there are common patterns that can be reused in many jobs.

When subclassing FedJob, any number of objects can be sent to the server in the __init__,
and :func:`set_up_client<nvflare.job_config.api.FedJob.set_up_client>` can be implemented to send objects to clients.
``set_up_client`` is called by FedJob when first sending object to a client target, as the specific client targets can vary.

For example of a Job pattern, we can use :class:`FedAvgJob<nvflare.app_opt.pt.job_config.fed_avg.FedAvgJob>` to simplify the creation of a FedAvg job.
The FedAvgJob automatically adds the FedAvg controller, PTFileModelPersistor and IntimeModelSelector, resulting in the following experience:

.. code-block:: python

    job = FedAvgJob(name="cifar10_fedavg", num_rounds=num_rounds, n_clients=n_clients, initial_model=Net())

For more examples of job patterns, see:

* :class:`BaseFedJob<nvflare.app_opt.pt.job_config.base_fed_job.BaseFedJob>`
* :class:`FedAvgJob<nvflare.app_opt.pt.job_config.fed_avg.FedAvgJob>` (pytorch)
* :class:`FedAvgJob<nvflare.app_opt.tf.job_config.fed_avg.FedAvgJob>` (tensorflow)
* :class:`CCWFJob<nvflare.app_common.ccwf.ccwf_job.CCWFJob>`
* :class:`FlowerJob<nvflare.app_opt.flower.flower_job.FlowerJob>`

.. note::

  Some of the default components included in these patterns are different, always refer to the
  exported job configs for a full list of components used at every site.


Running the Job
===============

Simulator
---------

Run the FedJob with the simulator with :func:`simulator_run<nvflare.job_config.api.FedJob.simulator_run>` in the ``workspace``, with ``n_clients``, ``threads``, and ``gpu`` assignments.

.. note::

    Only set ``n_clients`` if you have not specified clients using :func:`to<nvflare.job_config.api.FedJob.to>`.

Example:

.. code-block:: python

  job.simulator_run(workspace="/tmp/nvflare/jobs/workdir", n_clients=2, threads=2, gpu="0,1")


Export Configuration
--------------------
We can export the job configuration with :func:`export_job<nvflare.job_config.api.FedJob.export_job>` to the ``job_root`` directory to be used in other modes.

Example:

.. code-block:: python

  job.export_job(job_root="/tmp/nvflare/jobs/job_config")

Examples
========

To see examples of how the FedJob API can be used for different applications, refer the :github_nvflare_link:`Getting Started <examples/getting_started>` and :github_nvflare_link:`Job API <examples/advanced/job_api>` examples.
