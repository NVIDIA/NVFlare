.. _fl_clients:

##########
FL Clients
##########

In the NVFlare system, a federated learning algorithm is defined in a Job format (for details, please refer to :ref:`job`).
A Job consists of multiple "workflows" and "executors."

The simplified job execution flow is as follows:

- The workflow schedules a task for the FL clients.
- Each FL client performs the received task and sends the result back.
- The workflow receives the results and determines if it is done.
- If it is not done, it schedules a new task
- If it is done, it proceeds to the next workflow in the Job.

So, we can view the FL Client as a simple task performer that takes input and produces an output.

We offer various levels of abstraction for writing task execution code,
catering to use cases that span from complete customizability to easy user adaptation.

Below is a general overview of the key ideas and use cases for each type:

**Executor**

An :ref:`executor` is an FLComponent for clients used for executing tasks, wherein the execute method receives and returns a Shareable object given a task name.
Executors are the most flexible for defining custom logic and tasks, as with a custom executor and controller, any form of computation can be performed.
However, Executors must deal directly with FLARE-specific communication concepts such as :class:`Shareable<nvflare.apis.shareable.Shareable>`, :class:`DXO<nvflare.apis.dxo.DXO>`, and :class:`FLContext<nvflare.apis.fl_context.FLContext>`.
As a result, many higher-level APIs are built on top of Executors in order to abstract these concepts away for easier user adaptation.

Overall, writing an Executor is most useful when implementing tasks and logic that do not fit within the structure of higher-level APIs or other predefined Executors.

**Model Learner**

The :ref:`model_learner` is designed to simplify writing learning logic by minimizing FLARE specific concepts.
The :class:`ModelLearner<nvflare.app_common.abstract.model_learner.ModelLearner>` defines familiar learning functions for training and validation, and uses the :class:`FLModel<nvflare.app_common.abstract.fl_model.FLModel>` object for transferring learning information.
The ModelLearner also contains several convenient capabilities, such as lifecycle and logging information.

The ModelLearner is best used when working with standard machine learning code that can fit well into the train and validate methods and can be easily adapted to the ModelLearner subclass and method structure.

**Client API**

The :ref:`client_api` provides the most straightforward way to write FL code, and can easily be used to convert centralized code with minimal code changes.
The client API uses the :class:`FLModel<nvflare.app_common.abstract.fl_model.FLModel>` object for data transfer and supports common tasks such as train, validate, and submit_model.
Additionally, options for using decorators or PyTorch Lightning are also available.

We recommend users start with the Client API, and to consider the other types for more specific cases as required.

**3rd-Party System Integration**

The :ref:`3rd_party_integration` pattern allows for a seamless integration between the FLARE system and a third-party external training system.
This is especially useful with pre-existing ML/DL training system infrastructure that cannot be easily adapted to the FLARE client.

With the use of the :mod:`FlareAgent <nvflare.client.flare_agent>` and :mod:`TaskExchanger <nvflare.app_common.executors.task_exchanger>`, users can easily enable any 3rd-party system to receive tasks and submit results back to the server.


For more details about each type, refer to each page below.

.. toctree::
   :maxdepth: 3

   fl_clients/executor
   fl_clients/model_learner
   fl_clients/client_api
   fl_clients/3rd_party_integration