.. _job_configuration:

Predefined Job Configuration Variables
======================================

The following are predefined variables that can be configured in job config files.
The default values of these variables are usually good enough. However, you may change them to different values in some specific cases.

Runner Sync
-----------

When a job is deployed, dedicated job-specific processes are created throughout the system for the execution of the job.
Specifically, a dedicated server process is created to perform server-side logic; and dedicated client processes (one process for each site) are created to perform client-side logic.
This design allows multiple jobs to be running in their isolated space at the same time. The success or failure of a job won't interfere with the execution of other jobs.

The task-based interactions between an FL client and the FL server are done with the ClientRunner on the client side and the ServerRunner on the server side.
When the job is deployed, the order of the job process creation is not guaranteed - the server-side job process may be started before or after any client-side job process.

To ensure that the ClientRunner does not start to fetch tasks from the ServerRunner, the two runners need to be synchronized first.
Specifically, the ClientRunner keeps sending a "runner sync" request to the ServerRunner until a response is received.

The behavior of the "runner sync" process can be configured with two variables:

runner_sync_timeout
^^^^^^^^^^^^^^^^^^^

This variable is for the client-side configuration (config_fed_client.json).

This runner_sync_timeout specifies the timeout value for the "runner sync" request.
If a response is not received from the server within this specified value, then another "runner sync" request will be sent.

The default value is 2.0 seconds.

max_runner_sync_tries
^^^^^^^^^^^^^^^^^^^^^

This variable is for the client-side configuration (config_fed_client.json).

This variable specifies the max number of "runner sync" messages to be sent before receiving a response from the server.
If a response is still not received after this many tries, the client's job process will terminate.

The default value is 30.

The default settings of these two variables mean that if the ClientRunner and the ServerRunner are not synchronized within one minute, the client will terminate.
If one minute is not enough, you can extend these two variables to meet your requirement.

Task Check
----------

After the client is finished with the assigned task, it will send the result to the server, and before sending the result, the client asks the server whether the task is still valid.
This is particularly useful when the result is large and the communication network is slow. If the task is no longer valid, then the client won't need to send the result any more.
The client keeps sending the "task check" request to the server until a response is received.

The behavior of "task check" process can be configured with two variables:

task_check_timeout
^^^^^^^^^^^^^^^^^^

This variable is for the client-side configuration (config_fed_client.json).

This variable specifies the timeout value for the "task check" request.
If a response is not received from the Server within this specified value, then another "task check" request will be sent.

The default value is 5.0 seconds.

task_check_interval
^^^^^^^^^^^^^^^^^^^

This variable is for the client-side configuration (config_fed_client.json).

This variable specifies how long to wait before sending another "task check" request if a response is not received from the server for the previous request.

The default value is 5.0 seconds.

Get Task
--------

The client sends the "get task" request to the server to get the next assigned task.
You can set the get_task_timeout variable to specify how long to wait for the response from the server.
If a response is not received from the server within the specified time, the client will try again.

It is crucial to set this variable to a proper value.
If this value is too short for the server to deliver the response to the client in time, then the server may get repeated requests for the same task.
This can cause the server to run out of memory (since there could be many messages inflight to the same client).

The default value of this variable is 30 seconds. You change its value by setting it in the config_fed_client.json:

``get_task_timeout: 60.0``

Submit Task Result
------------------

The client submits the task result to the server after the task is completed. You can set the submit_task_result_timeout variable to specify how long to wait for the response from the server. If a response is not received from the server within the specified time, the client will try to send the result again until it succeeds.

It is crucial to set this variable to a proper value. If this value is too short for the server to accept the result and deliver a response to the client in time, then the server may get repeated task results for the same task. This can cause the server to run out of memory (since there could be many messages coming to the server).

The default value of this variable is 30 seconds. You change its value by setting it in the config_fed_client.json:

``submit_task_result_timeout: 120.0``

Job Heartbeat
-------------

A task could take the client a long time to finish.
During this time, there is no interaction between the client-side job process and the server-side job process.
In some network environments, this long-time silence could cause the underlying network to drop connections, which could cause some system functions to fail (e.g. any server-initiated messages may not be delivered to the client in a timely fashion).
To prevent this problem, the client's job process sends periodical heartbeats to the server.
The behavior of the heartbeat is controlled by:

job_heartbeat_interval
^^^^^^^^^^^^^^^^^^^^^^

This variable is for the client-side configuration (config_fed_client.json).
This variable specifies how often to send a heartbeat message to the server.

The default value is 30.0 seconds. You can tune this value up or down depending on your communication network's behavior.

Graceful Job Completion
-----------------------

Many components could be involved in the execution of a job. At the end of the job, all components should end gracefully.
For example, a stats report component may still have pending stats records to be processed when the job is done.
If the job process (server-side or client-side) is abruptly terminated when the job's workflow is done, then the pending records would be lost.

To enable graceful completion of components, FLARE will fire the ``EventType.CHECK_END_RUN_READINESS event``.
A component that may have pending tasks can listen to this event and indicate whether it is ready to end.
FLARE will repeat the event until all components are ready to end; or until a configured max time is reached.

end_run_readiness_timeout
^^^^^^^^^^^^^^^^^^^^^^^^^

This variable is for both the server-side (config_fed_server.json) and client-side configuration (config_fed_client.json).
This variable specifies the max time to wait for all components to become ready to end.

The default value is 5.0 seconds

end_run_readiness_check_interval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This variable is for both the server-side (config_fed_server.json) and client-side configuration (config_fed_client.json).
This variable specifies how long to wait before checking component readiness again.

The default value is 0.5 seconds.