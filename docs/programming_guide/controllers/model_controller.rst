.. _model_controller:

###################
ModelController API
###################

The FLARE :mod:`ModelController<nvflare.app_common.workflows.model_controller>` API provides an easy way for users to write and customize FLModel-based controller workflows.

* Highly flexible with a simple API (run routine and basic communication and utility functions)
* :ref:`fl_model` for the communication data structure, everything else is pure Python
* Option to support pre-existing components and FLARE-specific functionalities

.. note::

    The ModelController API is a high-level API meant to simplify writing workflows.
    If users prefer or need the full flexibility of the Controller with all the capabilities of FLARE functions, refer to the :ref:`controllers`.


Core Concepts
=============

As an example, we can take a look at the popular federated learning workflow, "FedAvg" which has the following steps:

#. FL server initializes an initial model
#. For each round (global iteration):

   #. FL server sends the global model to clients
   #. Each FL client starts with this global model and trains on their own data
   #. Each FL client sends back their trained model
   #. FL server aggregates all the models and produces a new global model


To implement this workflow using the ModelController there are a few essential parts:

* Import and subclass the :class:`nvflare.app_common.workflows.model_controller.ModelController`.
* Implement the ``run()`` routine for the workflow logic.
* Utilize ``send_model()`` / ``send_model_and_wait()`` for communication to send tasks with FLModel to target clients, and receive FLModel results.
* Customize workflow using predefined utility functions and components, or implement your own logics.


Here is an example of the FedAvg workflow using the :class:`BaseFedAvg<nvflare.app_common.workflows.base_fedavg.BaseFedAvg>` base class:

.. code-block:: python

    # BaseFedAvg subclasses ModelController and defines common functions and variables such as aggregate(), update_model(), self.start_round, self.num_rounds
    class FedAvg(BaseFedAvg):

      # run routine that user must implement
      def run(self) -> None:
          self.info("Start FedAvg.")

          # load model (by default uses persistor, can provide custom method)
          model = self.load_model()
          model.start_round = self.start_round
          model.total_rounds = self.num_rounds

          # for each round (global iteration)
          for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
              self.info(f"Round {self.current_round} started.")
              model.current_round = self.current_round

              # obtain self.num_clients clients
              clients = self.sample_clients(self.num_clients)

              # send model to target clients with default train task, wait to receive results
              results = self.send_model_and_wait(targets=clients, data=model)

              # use BaseFedAvg aggregate function
              aggregate_results = self.aggregate(
                  results, aggregate_fn=self.aggregate_fn
              )  # using default aggregate_fn with `WeightedAggregationHelper`. Can overwrite self.aggregate_fn with signature Callable[List[FLModel], FLModel]

              # update global model with aggregation results
              model = self.update_model(model, aggregate_results)

              # save model (by default uses persistor, can provide custom method)
              self.save_model(model)

          self.info("Finished FedAvg.")


Below is a comprehensive table overview of the :class:`ModelController<nvflare.app_common.workflows.model_controller.ModelController>` API:


.. list-table:: ModelController API
   :widths: 25 35 50
   :header-rows: 1

   * - API
     - Description
     - API Doc Link
   * - run
     - Run routine for workflow.
     - :func:`run<nvflare.app_common.workflows.model_controller.ModelController.run>`
   * - send_model_and_wait
     - Send a task with data to targets (blocking) and wait for results..
     - :func:`send_model_and_wait<nvflare.app_common.workflows.model_controller.ModelController.send_model_and_wait>`
   * - send_model
     - Send a task with data to targets (non-blocking) with callback.
     - :func:`send_model<nvflare.app_common.workflows.model_controller.ModelController.send_model>`
   * - sample_clients
     - Returns a list of num_clients clients.
     - :func:`sample_clients<nvflare.app_common.workflows.model_controller.ModelController.sample_clients>`
   * - save_model
     - Save model with persistor.
     - :func:`save_model<nvflare.app_common.workflows.model_controller.ModelController.save_model>`
   * - load_model
     - Load model from persistor.
     - :func:`load_model<nvflare.app_common.workflows.model_controller.ModelController.load_model>`


Communication
=============

The ModelController uses a task based communication where tasks are sent to targets, and targets execute the tasks and return results.
The :ref:`fl_model` is standardized data structure object that is sent along with each task, and :ref:`fl_model` responses are received for the results.

.. note::

    The :ref:`fl_model` object can be any type of data depending on the specific task.
    For example, in the "train" and "validate" tasks we send the model parameters along with the task so the target clients can train and validate the model.
    However in many other tasks that do not involve sending the model (e.g. "submit_model"), the :ref:`fl_model` can contain any type of data (e.g. metadata, metrics etc.) or may not be needed at all.


send_model_and_wait
-------------------
:func:`send_model_and_wait<nvflare.app_common.workflows.model_controller.ModelController.send_model_and_wait>` is the core communication function which enables users to send tasks to targets, and wait for responses.

The ``data`` is an :ref:`fl_model` object, and the ``task_name`` is the task for the target executors to execute (Client API executors by default support "train", "validate", and "submit_model", however executors can be written for any arbitrary task name).

``targets`` can be chosen from client names obtained with ``sample_clients()``.

Returns the :ref:`fl_model` responses from the target clients once the task is completed (``min_responses`` have been received, or ``timeout`` time has passed).

send_model
----------
:func:`send_model<nvflare.app_common.workflows.model_controller.ModelController.send_model>` is the non-blocking version of 
:func:`send_model_and_wait<nvflare.app_common.workflows.model_controller.ModelController.send_model_and_wait>` with a user-defined callback when receiving responses.

A callback with the signature ``Callable[[FLModel], None]`` can be passed in, which will be called when a response is received from each target.

The task is standing until either ``min_responses`` have been received, or ``timeout`` time has passed.
Since this call is asynchronous, the Controller :func:`get_num_standing_tasks<nvflare.apis.impl.controller.Controller.get_num_standing_tasks>` method can be used to get the number of standing tasks for synchronization purposes.

For example, in the :github_nvflare_link:`CrossSiteEval <app_common/workflows/cross_site_eval.py>` workflow, the tasks are asynchronously sent with :func:`send_model<nvflare.app_common.workflows.model_controller.ModelController.send_model>` to get each client's model.
Then through a callback, the clients' models are sent to the other clients for validation.
Finally, the workflow waits for all standing tasks to complete with :func:`get_num_standing_tasks<nvflare.apis.impl.controller.Controller.get_num_standing_tasks>`.
Below is an example of how these functions can be used. For more details view the implementation of :github_nvflare_link:`CrossSiteEval <app_common/workflows/cross_site_eval.py>`.


.. code-block:: python

    class CrossSiteEval(ModelController):
        ...
        def run(self) -> None:
            ...
            # Create submit_model task and broadcast to all participating clients
            self.send_model(
                task_name=AppConstants.TASK_SUBMIT_MODEL,
                data=data,
                targets=self._participating_clients,
                timeout=self._submit_model_timeout,
                callback=self._receive_local_model_cb,
            )
            ...
            # Wait for all standing tasks to complete, since we used non-blocking `send_model()`
            while self.get_num_standing_tasks():
                if self.abort_signal.triggered:
                    self.info("Abort signal triggered. Finishing cross site validation.")
                    return
                self.debug("Checking standing tasks to see if cross site validation finished.")
                time.sleep(self._task_check_period)

            self.save_results()
            self.info("Stop Cross-Site Evaluation.")

        def _receive_local_model_cb(self, model: FLModel):
            # Send this model to all clients to validate
            model.meta[AppConstants.MODEL_OWNER] = model_name
            self.send_model(
                task_name=AppConstants.TASK_VALIDATION,
                data=model,
                targets=self._participating_clients,
                timeout=self._validation_timeout,
                callback=self._receive_val_result_cb,
            )
        ...


Saving & Loading
================

persistor
---------
The :func:`save_model<nvflare.app_common.workflows.model_controller.ModelController.save_model>` and :func:`load_model<nvflare.app_common.workflows.model_controller.ModelController.load_model>`
functions utilize the configured :class:`ModelPersistor<nvflare.app_common.abstract.model_persistor.ModelPersistor>` set in the ModelController ``persistor_id: str = "persistor"`` init argument.

custom save & load
------------------
Users can also choose to instead create their own custom save and load functions rather than use a persistor.

For example we can use PyTorch's save and load functions for the model parameters, and save the FLModel metadata with :mod:`FOBS<nvflare.fuel.utils.fobs>` separately to different filepaths.

.. code-block:: python

    import torch
    from nvflare.fuel.utils import fobs

    class MyController(ModelController):
        ...
        def save_model(self, model, filepath=""):
            params = model.params
            # PyTorch save
            torch.save(params, filepath)

            # save FLModel metadata
            model.params = {}
            fobs.dumpf(model, filepath + ".metadata")
            model.params = params

        def load_model(self, filepath=""):
            # PyTorch load
            params = torch.load(filepath)

            # load FLModel metadata
            model = fobs.loadf(filepath + ".metadata")
            model.params = params
            return model


Note: for non-primitive data types such as ``torch.nn.Module`` (used for the initial PyTorch model),
we must configure a corresponding FOBS decomposer for serialization and deserialization.
Read more at :ref:`serialization`.

.. code-block:: python

  from nvflare.app_opt.pt.decomposers import TensorDecomposer

  fobs.register(TensorDecomposer)


Additional Functionalities
==========================

In some cases, more advanced FLARE-specific functionalities may be of use.

The :mod:`BaseModelController<nvflare.app_common.workflows.base_model_controller>` class provides access to the engine ``self.engine`` and FLContext ``self.fl_ctx`` if needed.
Functions such as ``get_component()`` and ``build_component()`` can be used to load or dynamically build components.

Furthermore, the underlying :mod:`Controller<nvflare.apis.impl.controller>` class offers additional communication functions and task related utilities.
Many of our pre-existing workflows are based on this lower-level Controller API.
For more details refer to the :ref:`controllers` section.

Examples
========

Examples of basic workflows using the ModelController API:

* :github_nvflare_link:`Cyclic <nvflare/app_common/workflows/cyclic.py>`
* :github_nvflare_link:`BaseFedAvg <nvflare/app_common/workflows/base_fedavg.py>`
* :github_nvflare_link:`FedAvg <nvflare/app_common/workflows/fedavg.py>`

Advanced examples:

* :github_nvflare_link:`Scaffold <nvflare/app_common/workflows/scaffold.py>`
* :github_nvflare_link:`FedOpt <nvflare/app_opt/pt/fedopt_ctl.py>`
* :github_nvflare_link:`PTFedAvgEarlyStopping <nvflare/app_opt/pt/fedavg_early_stopping.py>`
* :github_nvflare_link:`Kaplan-Meier <examples/advanced/kaplan-meier-he/src/kaplan_meier_wf_he.py>`
* :github_nvflare_link:`Logistic Regression Newton Raphson <examples/advanced/lr-newton-raphson/job/newton_raphson/app/custom/newton_raphson_workflow.py>`
* :github_nvflare_link:`FedBPT <research/fed-bpt/src/global_es.py>`
