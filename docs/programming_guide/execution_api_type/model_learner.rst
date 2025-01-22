.. _model_learner:

#############
Model Learner
#############

Introduction
============

The goal of :github_nvflare_link:`ModelLearner <nvflare/app_common/abstract/model_learner.py>` is to make it easier to write learning logic by minimizing FLARE specific concepts exposed to the user.

The central concept of the ModelLearner is :github_nvflare_link:`FLModel <nvflare/app_common/abstract/fl_model.py>`, which defines a structure to support federated learning functions with familiar learning terms.
To create a concrete model learner, the researcher will implement the training and validation methods only with the FLModel object. 
The researcher no longer needs to deal with FLARE specific concepts such as Shareable and FLContext, though they are still available for advanced cases where FLModel is not enough.

How to Create Model Learner
===========================

To create a concrete model learner, you extend from the ModelLearner class. The following shows the example of NPLearner:

.. code-block:: python

   from nvflare.app_common.abstract.model_learner import ModelLearner
   from nvflare.app_common.abstract.fl_model import FLModel, ParamsType


   class NPLearner(ModelLearner):

The following methods must be implemented:

.. code-block:: python

      def initialize(self)
      def train(self, model: FLModel) -> Union[str, FLModel]:
      def get_model(self, model_name: str) -> Union[str, FLModel]:
      def validate(self, model: FLModel) -> Union[str, FLModel]:
      def configure(self, model: FLModel)
      def abort(self)
      def finalize(self)

Please see the docstrings of these methods for explanation at :class:`ModelLearner<nvflare.app_common.abstract.model_learner.ModelLearner>`.

Initialization and Finalization
-------------------------------

In the case that the ModelLearner requires initialization, put your initialization logic in the ``initialize`` method, which is called only once before the learning job starts.
The ModelLearner base class provides many convenience methods that you may use in the initialization logic. 

Similarly your ModelLearner may need to be properly ended.
If so, put such logic in the ``finalize`` method, which is called only once when the learning job is finished.

Learning Logic
--------------

Your learning logic is implemented in the ``train`` and ``validate`` methods. All learning information is contained in the FLModel object.
Similarly the result of the learning methods is either a FLModel object (when processing succeeds) or a str for the ReturnCode when processing fails for some reason.

You should check the FLModel object's params_type to ensure that it has the params you expected.

If possible, you should periodically check whether the ModelLearner has been asked to abort in your learning logic, especially before or after a long-running step.
You can do so by calling the ``self.is_aborted()`` method. The typical usage pattern is:

.. code-block:: python

	if self.is_aborted():
   		return ReturnCode.TASK_ABORTED


If you run into a case that prevents the learning logic from proceeding, you can simply return a proper ReturnCode from the learning method.

Return Requested Model
----------------------

The ModelLearner may be asked to return a specified type of model (e.g. best model).
For example, when training is done, the server may ask you to return the best local model so then it can send it to other sites to validate. 
To support this, you need to implement the ``get_model`` method and return the requested model.

Dynamic Configuration
---------------------

If you want to configure the ModelLearner dynamically based on information sent from the server (instead of statically based on locally configured information), you can do so by implementing the ``configure`` method.
The FLModel object should specify the config parameters for the model learning functions.

Abort Gracefully
----------------

The ModelLearner may be asked to abort during the execution of its learning methods (e.g. the user may issue the ``abort_job`` command, or the server's controller may decide to abort the task).
Depending on the framework your learning method uses (e.g. MONAI, Ignite, TensorFlow, etc.), you may need to do something to make the training framework abort gracefully. 
In this case, you will put such logic in the ``abort`` method.

The ``abort`` method is optional. You don't need to implement this method if your training framework cannot be interrupted or does not need to be interrupted.

Logging Methods
---------------

The ModelLearner base class provides convenience methods for logging: 

.. code-block:: python

   def debug(self, msg: str)
   def info(self, msg: str)
   def error(self, msg: str)
   def warning(self, msg: str)
   def exception(self, msg: str)
   def critical(self, msg: str)

You can use these methods to create log messages at different log levels in your learning logic.

Get Additional Component
------------------------

FLARE runtime provides many service components (e.g. stats logging, security, config service) that you can use in your learner implementation. 
You can get these objects via this method provided by the ModelLearner class:

.. code-block:: python

   def get_component(self, component_id: str) -> Any

You usually should call this when initializing the learner.

Here is an example of using an AnalyticsSender client component in CIFAR10ModelLearner:

.. code-block:: python

   self.writer = self.get_component(
      self.analytic_sender_id
   ) 

Get Contextual Information
--------------------------

The FLModel object contains essential information about the learning task. There is still other contextual information that you may need:

- site_name: the name of the training site
- engine: the FLARE engine that provides additional information and services
- workspace: the workspace that you can use to retrieve and/or write data to
- job_id: the ID of the job
- app_root: the root directory of the current job in the workspace.
- shareable: the Shareable object that comes with the task
- fl_ctx: the FLContext object that comes with the task

These are directly available in your learner object (self).

The ModelLearner base class also provides additional convenience methods for you to get properties in the Shareable and FLContext objects:

.. code-block:: python

   def get_shareable_header(self, key: str, default=None)
   def get_context_prop(self, key: str, default=None)

How to Install Model Learner
============================

Once your model learner is developed, you need to install it to the training client. 
The model learner must work with the ModelLearnerExecutor that FLARE provides. 
The following example shows how the model learner is configured in the job's ``config_fed_client.json``:

.. code-block:: json

   {
      "format_version": 2,
      "executors": [
         {
            "tasks": [
               "train"
            ],
            "executor": {
               "name": "LearnerExecutor",
               "path": "nvflare.app_common.executors.model_learner_executor.ModelLearnerExecutor",
               "args": {
                  "learner_id": "np_learner"
               }
            }
         }
      ],
      "task_result_filters": [
      ],
      "task_data_filters": [
      ],
      "components": [
         {
            "id": "np_learner",
            "path": "np_learner.NPLearner",
            "args": {
            }
         }
      ]
   }

Pay attention to the following:

- The ``path`` of the ``executor`` must be ``nvflare.app_common.executors.model_learner_executor.ModelLearnerExecutor``.
- The ``learner_id`` in the ``executor`` and the ``id`` in the ``components`` must match (In this example it is ``np_learner``).
- The path of the ``np_learner`` component must point to your model learner implementation.

More Resources
==============

In addition to the :github_nvflare_link:`ModelLearner <nvflare/app_common/abstract/model_learner.py>` and :github_nvflare_link:`FLModel <nvflare/app_common/abstract/fl_model.py>` APIs, also take a look at some examples using the ModelLearner:

- :github_nvflare_link:`Step-by-step ModelLearner <examples/hello-world/step-by-step/cifar10/sag_model_learner/sag_model_learner.ipynb>`
- :github_nvflare_link:`CIFAR10 ModelLearner <examples/advanced/cifar10/pt/learners/cifar10_model_learner.py>`
