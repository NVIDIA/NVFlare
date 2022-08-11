.. _controllers:

Controllers and Controller API
==============================

Controller/Worker Interactions
------------------------------

NVIDIA FLARE 2.0's collaborative computing is achieved through the Controller/Worker interactions. The following diagram
shows how the Controller and Worker interact.

.. image:: ../resources/Controller.png
    :height: 300px

The Controller is a python object that controls or coordinates the Workers to get a job done. The controller is run on
the FL server (highlighted on the right).

A Worker is capable of performing tasks. Workers run on FL clients.

In its control logic, the Controller assigns tasks to Workers and processes task results from the Workers.

Workers keep asking for the next task to do, executes the task, and submits results to the Controller, until instructed
to exit by the Controller (a special END_RUN task).

Controller API
--------------

The :mod:`Controller API<nvflare.apis.controller_spec>` provides methods for assigning tasks to the Workers (FL clients)
in different ways:

   - Broadcast a task to multiple clients
   - Send a task to a single client
   - Arrange a task to be done by multiple clients in turns

See the included :class:`Controller<nvflare.apis.impl.controller.Controller>` implementation and full reference
implementations of the following controller workflows:

.. toctree::
   :maxdepth: 1

   controllers/scatter_and_gather_workflow
   controllers/cross_site_model_evaluation.rst
   controllers/cyclic_workflow.rst

You can study the source code and use it as a starting point to write your own controller workflows.

.. _tasks:

Task Lifecycle
--------------

The central concept of the Controller API is :class:`Task<nvflare.apis.controller_spec.Task>`.

A :class:`Task<nvflare.apis.controller_spec.Task>` is a piece of work that is assigned by the Controller to client workers. Depending on how the task is assigned (broadcast, send, or relay), the task will be performed by one or more clients.

The Controller's Task Manager manages the task's lifecycle:

    - First, the programmer creates the task, specifying the name and the data of the task.
    - Then, the programmer calls one of the task methods (e.g. broadcast, send, relay, etc.). All these methods do is simply adding the task to the Task Queue. Now the task is waiting for clients to come to retrieve it. Note that there could be multiple tasks in the queue.
    - When a client comes to get the next task, the Task Manager decides which task in the queue should be assigned to the client. The general rule is that the tasks will be examined one by one following their orders in the queue. If the client is a candidate of the task and the task has not been performed by the client, AND the task-specific rule allows the client to be assigned, then the task is assigned to the client, and a new ClientTask record is created and added to the task's client_tasks list. If the before_task_sent callback (CB) is provided, it is called before sending the task to the client.
    - If no task is found for the client, the Task Manager tells the client to try again later.
    - When the client finishes its assigned task and comes back to submit its result, the client_task is found for this client, and then the result_received CB (if provided) is called. The result is recorded into the client_task record, and the client_task is marked as "result received".
    - Eventually the task is completed when one of the following conditions is met:
        - The task itself is timed out (if the task timeout is specified)
        - All assigned tasks received results from clients
        - Task specific exit rule is met (e.g. for broadcast, the minimal-responses are received and waited for enough time after that)
        - The task is cancelled explicitly
        - Fatal error occurred (task data filtering error) and the task is cancelled by the system
    - Once the task is completed, it's completion_status is set based on the condition the task is completed, and the task is removed from the task queue. If the task_done CB is provided, it is called. This is the end of the task's lifecycle.

.. note::

    In NVIDIA FLARE 2.0, the underlying communication is by gRPC: the client always initiates communication by sending
    a request to the server and a receiving response. When we say "server sends task to the client", it is only
    conceptual. With gRPC, the client sends the "ask for next task" request to the server, and the server responds with
    the task data.
