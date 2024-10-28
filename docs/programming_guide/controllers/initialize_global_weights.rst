.. _initialize_global_weights_workflow:

Initialize Global Weights Workflow for Client-Side Global Model Initialization
------------------------------------------------------------------------------
The SAG controller requires the global model weights to be initialized before the training is started. Currently it is the job of
the Persistor component to provide the initial weights, which either loads it from some predefined model file, or dynamically generates
it by running a piece of custom Python code. The 1st approach requires the hassle of defining a model file; the second approach requires
custom Python code, which could be a security risk.

We introduce the third approach to generate initial model weights based on initial weights from FL clients: creating a controller to
collect weights from clients, and putting this controller in front of the SAG! This controller is 
:class:`nvflare.app_common.workflows.initialize_global_weights.InitializeGlobalWeights`.

How to Use InitializeGlobalWeights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following steps are based on the ``hello-pt`` example in the NVFLARE repo.

Step 1: Modify config_fed_server.json
"""""""""""""""""""""""""""""""""""""

Two changes are needed:

    - Remove the ``model`` arg from the PTFileModelPersistor component configuration.
    - Add the ``InitializeGlobalWeights`` controller as the first controller in the workflow.

The updated file should look like the following:

.. literalinclude:: ../../resources/init_weights_1_config_fed_server.json
   :language: json


Note that ``PTFileModelPersistor`` no longer requires the custom ``SimpleNetwork`` as the model object.

Pay attention to the value of the ``task_name``, "get_weights", in the InitializeGlobalWeights configuration.

Step 2: Modify config_fed_client.json
"""""""""""""""""""""""""""""""""""""

Add the task "get_weights" for the trainer, as highlighted in below. Note that this task name must match the
task_name of the InitializeGlobalWeights config in config_fed_server.json.

.. code-block:: json
    :emphasize-lines: 8

    {
      "format_version": 2,
      "executors": [
          {
              "tasks": [
                  "train",
                  "submit_model",
                  "get_weights"
              ],
              "executor": {
                  "path": "cifar10trainer.Cifar10Trainer",
                  "args": {
                      "lr": 0.01,
                      "epochs": 1
                  }
              }
          },
          {
              "tasks": [
                  "validate"
              ],
              "executor": {
                  "path": "cifar10validator.Cifar10Validator",
                  "args": {}
              }
          }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": []
    }


Step 3: Update the Trainer code
"""""""""""""""""""""""""""""""
A new ``pre_train_task_name`` (defaults to "get_weights") is added to Cifar10Trainer, which is an Executor. 

The Trainer is augmented to process this task and return the current model weights (which should be randomly initialized).

The following are the relevant code snippets:


.. code-block:: python

    class Cifar10Trainer(Executor):
    
        def __init__(self, lr=0.01, epochs=5,
                train_task_name=AppConstants.TASK_TRAIN,
                submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
                exclude_vars=None,
                pre_train_task_name=AppConstants.TASK_GET_WEIGHTS):

.. code-block:: python

    def _get_model_weights(self) -> Shareable:
        # Get state dict and send as weights
        new_weights = self.model.state_dict()
        new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=new_weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )
        return outgoing_dxo.to_shareable()


.. code-block:: python

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._pre_train_task_name:
                # return model weights
                return self._get_model_weights()
            elif task_name == self._train_task_name:
                ...

The full implementation is in ``cifar10trainer.py`` of the custom folder of ``hello-pt``.

InitializeGlobalWeights Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When processing client responses (which are model weights of the clients), ``GlobalWeightsInitializer`` selects one as the global weight.
It supports two weight selection methods (specified with the "weight_method" argument):

  - **first** - use the weight reported by the first client responded. This is the default method.
  - **client** - use the weight reported by a designated client. This could be useful for deterministic training.

To be complete, weights reported from all clients should be validated and compared to make sure they are valid and compatible. Currently,
``GlobalWeightsInitializer`` does not do this, since it is not clear how to do this exactly.

InitializeGlobalWeights Implementation Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``InitializeGlobalWeights`` controller is implemented by extending the general-purpose ``BroadcastAndProcess`` controller. 

The ``BroadcastAndProcess`` controller requires a ``ResponseProcessor`` component to process client responses. The ``BroadcastAndProcess`` controller works as follows:

  - It broadcasts a task of a configured name to all or a configured list of clients to ask for their data.
  - Each time a response is received from a client, ``BroadcastAndProcess`` invokes the ``ResponseProcessor`` component to process the response.
  - Once the task is completed (responses received from all clients or timed out), ``BroadcastAndProcess`` invokes the ``ResponseProcessor`` component
    to do the final check.

The ``InitializeGlobalWeights`` controller simply extends the ``BroadcastAndProcess`` with the ``GlobalWeightsInitializer`` as the ``ResponseProcessor``
component.
