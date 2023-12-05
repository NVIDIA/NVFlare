.. _fl_clients:

##########
FL Clients
##########

FLARE Clients are workers in the FL system that perform tasks.
We provide different levels of abstraction for writing FL Client code to support use cases ranging from complete customizability to easy user adaption.
Here is a general overview of the key ideas and use cases of each FL Client type ordered from most FLARE-specific to least FLARE-specific:

**Executor**

An :ref:`executor` is an FLComponent for clients used for executing tasks, wherein the execute method receives and returns a Shareable object given a task name.
Executors are the most flexible for defining custom logic and tasks, as with a custom executor and controller, any form of computation can be performed.
However, Executors must deal directly with FLARE-specific communication concepts such as :class:`Shareable<nvflare.apis.shareable.Shareable>`, :class:`DXO<nvflare.apis.dxo.DXO>`, and :class:`FLContext<nvflare.apis.fl_context.FLContext>`.
As a result, many higher level APIs are built on top of Executors in order to abstract these concepts away for easier user adaption.

Overall, writing an Executor is most useful when implementing tasks and logic that do not fit within the structure of higher level APIs or other predefined Executors.

**Model Learner**

The :ref:`model_learner` is designed to simplify writing learning logic by minimizing FLARE specific concepts.
The :class:`ModelLearner<nvflare.app_common.abstract.model_learner.ModelLearner>` defines familiar learning functions for training and validation, and uses the :class:`FLModel<nvflare.app_common.abstract.fl_model.FLModel>` object for transferring learning information.
The ModelLearner also contains serveral convenience capabilities, such as lifecycle and logging information.

The Model Learner is best used when working with standard machine learning code that can fit well into the train and validate methods and can be easily adapated to the ModelLearner subclass and method structure.

**Client API**

The :ref:`client_api` provides the most straightforward way to write FL code, and can easily be used to convert centralized code with minimal code changes.
The client API uses the :class:`FLModel<nvflare.app_common.abstract.fl_model.FLModel>` object for data transfer, and supports common tasks such as train, validate, and submit_model.
Additionally, options for using decorators or PyTorch Lightning are also available.

As of version 2.4.0, we recommend users start with the Client API, and to consider the other Client types for more specific cases as required.

**3rd-Party System Integration**

The :ref:`3rd_party_integration` pattern allows for a seamless integration between the FLARE system and a third-party external training system.
This is especially useful with pre-existing ML/DL training system infrastructure that cannot be easily adapted to the FLARE client.

With the use of the :mod:`FlareAgent <nvflare.client.flare_agent>` and :mod:`TaskExchanger <nvflare.app_common.executors.task_exchanger>`, users can easily enable any 3rd-party system to receive tasks and submit results back to the server.


For more details about each client type, refer to each page below.

.. toctree::
   :maxdepth: 3

   fl_clients/executor
   fl_clients/model_learner
   fl_clients/client_api
   fl_clients/3rd_party_integration