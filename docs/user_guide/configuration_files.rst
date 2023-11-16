.. _configuration_files:

###################
Configuration Files
###################

Supported Configuration File Formats
====================================

- `JSON <https://www.json.org/json-en.html>`_
- `YAML <https://yaml.org/>`_
- `Pyhocon <https://github.com/chimpler/pyhocon>`_ - a JSON variant and HOCON (Human-Optimized Config Object Notation) parser for python.
  Supports comments, variable substitution, and inheritance.
- `OmegaConf <https://omegaconf.readthedocs.io/en/2.3_branch/>`_ - a YAML based hierarchical configuration.

Users have the flexibility to use a single format or combine several formats, as exemplified by using config_fed_client.conf and config_fed_server.json together.
If multiple configuration formats coexist, then their usage will be prioritized based on the following search order: ``.json -> .conf -> .yml -> .yaml``

Variable Resolution in Job Configuration
========================================

FLARE jobs are defined with configuration files: ``config_fed_client.json`` and ``config_fed_server.json``.
These two files configure the components (Python objects) used for the server process and the FL client processes.
The component configuration includes information about the class path of the Python object, and arguments for the object's constructor.
The configuration files are processed at the beginning of the server/client job processes to create those components.

Here is a typical example of a job configuration:

.. code-block:: json

   {
      "format_version": 2,
      "executors": [
         {
            "tasks": [
               "train"
            ],
            "executor": {
               "path": "nvflare.app_common.np.np_trainer.NPTrainer",
               "args": {
                  "sleep_time": 1.5,
                  "model_dir": "model"
               }
            }
         }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": [
      ]
   }

As shown in the example above, the ``executor`` component has two args (sleep_time and model_dir) and both are specified explicitly.

Variable Resolution
-------------------

Sometimes, users want to experiment with different arg values of the component, and want to manage those experimental args in a common place (e.g. beginning of the config file) instead of searching for the args from the file to modify them.
This is particularly true if the user has multiple components to experiment with.

FLARE makes this possible with a mechanism called Variable Resolution.
Instead of hard-coding values for each config arg, users can simply use a Variable Reference as the value of the arg, and then define the value of the variable in a separate place (e.g. beginning of the config file).

The following shows the configuration of the above example using variable resolution:

.. code-block:: json

   {
      "format_version": 2,
      "result_dir": "result",
      "sleep_time": 1.5,
      "executors": [
         {
            "tasks": [
               "train"
            ],
            "executor": {
               "path": "nvflare.app_common.np.np_trainer.NPTrainer",
               "args": {
                  "sleep_time": "{sleep_time}",
                  "model_dir": "{result_dir}"
               }
            }
         }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": [
      ]
   }


As you can see from the example, the Variable Definition (Var Def) is a simple JSON element that defines a value for a Variable Name (Var Name).
The Variable Reference (Var Ref) is a string that embeds the referenced Variable Name within curly brackets:  ``{VarName}``.

A var ref can be used within a string with other information.
For example, you could define the ``model_dir`` arg to include a prefix:
      ``/tmp/fl_work/{result_dir}``

You could reference multiple variables in one arg value:
      ``{root_dir}/{result_dir}``

If the arg value contains nothing but a single var ref, it is called a Simple Var Ref (SVR).
Other uses, such as var ref with other info, or multiple var refs, are called Complex Var Ref (CVR).
There is an important difference between a SVR and a CVR when the ref is resolved to compute the arg value: 
a SVR will be resolved to its true type of the corresponding variable definition; whereas a CVR is always resolved into a string with the values of the referenced variables.
The SVR can reference both primitive variables (number, boolean, string) and non-primitives (list and dict), whereas you can only use primitive variables with a CVR!

Predefined System Variables
---------------------------

Referenced variables must be defined. For user-defined variables, usually users define them somewhere in the config file (e.g. at the beginning of the file) as first-level elements, as shown in the above example.

FLARE predefined the following System Variables that are also available for you to use in the job config:

- SITE_NAME - the name of the site (server ot FL client)
- WORKSPACE - the directory of the site's workspace
- ROOT_URL - the url for connecting to the FL server
- SECURE_MODE - whether the communication is in secure mode

Note that system variables are named in UPPERCASE letters. To avoid potential name conflict between user-defined variables and system variables, please name all user-defined variables with lowercase letters.

The next example will show the use of system variables in CellPipe configuration.

OS Environment Variables
------------------------

OS environment variables can be referenced in job configuration via the dollar sign:

      ``{$EnvVarName}``

With this, you can make your job config controlled by OS environment variables.
For example, you can use an environment variable (e.g. NVFLARE_MODEL_DIR) to specify where the trained model will be stored such that system operators can change the model location without needing to change job configurations.
Note that if a variable with the name ``$VarName`` is already defined in the job config, then this definition takes precedence over the corresponding OS environment variable, if any.

The following example shows how to use an OS environment variable to control the location of model_dir:

.. code-block:: json

   {
      "format_version": 2,
      "executors": [
         {
            "tasks": [
               "train"
            ],
            "executor": {
               "path": "nvflare.app_common.np.np_trainer.NPTrainer",
               "args": {
                  "model_dir": "{$NVFLARE_MODEL_DIR}"
               }
            }
         }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": [
      ]
   }

Just like any other var definitions, OS environment variables can be referenced in both SVR and CVR.

Parameterized Variable Definitions
----------------------------------

Before discussing this advanced topic, let's first show an example of job configuration that does not use this technique for comparison:

.. code-block:: json

   {
      "format_version": 2,
      "pipe_token": "pipe_123",
      "executors": [
         {
            "tasks": [
               "train"
            ],
            "executor": {
               "path": "nvflare.app_common.executors.task_exchanger.TaskExchanger",
               "args": {
                  "pipe_id": "task_pipe"
               }
            }
         }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": [
         {
            "id": "task_pipe",
            "path": "nvflare.fuel.utils.pipe.cell_pipe.CellPipe",
            "args": {
               "mode": "passive",
               "site_name": "{SITE_NAME}",
               "token": "{pipe_token}",
               "root_url": "{ROOT_URL}",
               "secure_mode": "{SECURE_MODE}",
               "workspace_dir": "{WORKSPACE}"
            }
         },
         {
            "id": "metric_pipe",
            "path": "nvflare.fuel.utils.pipe.cell_pipe.CellPipe",
            "args": {
               "mode": "passive",
               "site_name": "{SITE_NAME}",
               "token": "{pipe_token}",
               "root_url": "{ROOT_URL}",
               "secure_mode": "{SECURE_MODE}",
               "workspace_dir": "{WORKSPACE}"
            }
         },
         {
            "id": "metric_receiver",
            "path": "nvflare.widgets.metric_receiver.MetricReceiver",
            "args": {
               "pipe_id": "metric_pipe"
            }
         }
      ]
   }


This job requires two pipes, one for task exchange (task_pipe), another for metrics collection (metric_pipe).
If you look at their configuration closely, you will see that: there are many args to configure, and the configs of the two pipes are identical except for their ``id`` values. It is tedious and error-prone to configure many args in multiple places.

One way to improve is to make use of SVR for the args of the two pipes:

.. code-block:: json

   {
      "format_version": 2,
      "pipe_token": "pipe_123",
      "executors": [
         {
            "tasks": [
               "train"
            ],
            "executor": {
               "path": "nvflare.app_common.executors.task_exchanger.TaskExchanger",
               "args": {
                  "pipe_id": "task_pipe"
               }
            }
         }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "pipe_args": {
         "mode": "passive",
         "site_name": "{SITE_NAME}",
         "token": "{pipe_token}",
         "root_url": "{ROOT_URL}",
         "secure_mode": "{SECURE_MODE}",
         "workspace_dir": "{WORKSPACE}"
      },
      "components": [
         {
            "id": "task_pipe",
            "path": "nvflare.fuel.utils.pipe.cell_pipe.CellPipe",
            "args": "{pipe_args}"
         },
         {
            "id": "metric_pipe",
            "path": "nvflare.fuel.utils.pipe.cell_pipe.CellPipe",
            "args": "{pipe_args}"
         },
         {
            "id": "metric_receiver",
            "path": "nvflare.widgets.metric_receiver.MetricReceiver",
            "args": {
               "pipe_id": "metric_pipe"
            }
         }
      ]
   }

In this version of the example, the args for the two pipes are moved into the var def ``pipe_args``, and the components' ``args`` simply reference the var def.
This is better than the original version, but the path of the two pipes still must be repeated for both components.

Using Parameterized Variable Definition, we can further improve it:

.. code-block:: json

   {
      "format_version": 2,
      "pipe_token": "pipe_123",
      "executors": [
         {
         "tasks": [
            "train"
         ],
         "executor": {
            "path": "nvflare.app_common.executors.task_exchanger.TaskExchanger",
            "args": {
               "pipe_id": "task_pipe"
            }
         }
         }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "@pipe_def": {
         "id": "{pipe_id}",
         "path": "nvflare.fuel.utils.pipe.cell_pipe.CellPipe",
         "args": {
         "mode": "passive",
         "site_name": "{SITE_NAME}",
         "token": "{pipe_token}",
         "root_url": "{ROOT_URL}",
         "secure_mode": "{SECURE_MODE}",
         "workspace_dir": "{WORKSPACE}"
         }
      },
      "components": [
         "{@pipe_def:pipe_id=task_pipe}",
         "{@pipe_def:pipe_id=metric_pipe}",
         {
            "id": "metric_receiver",
            "path": "nvflare.widgets.metric_receiver.MetricReceiver",
            "args": {
               "pipe_id": "metric_pipe"
            }
         }
      ]
   }

As you can see here, ``@pipe_def`` is a parameterized variable definition (PVD).
The name of a PVD must start with the ``@`` sign. The PVD is usually defined with references to other variables, and the values can be provided at the time the PVD is referenced.
In this example, the ``@pipe_def`` PVD defines a pipe configuration template that can be resolved to a concrete pipe config.
In the ``components`` section, this PVD is used for the config of the two pipes: task_pipe and metric_pipe.

A PVD can only be referenced with SVR (simple variable reference).
To reference a PVD, you provide values for any variables in the PVD.
In this example, the ``pipe_id`` is the variable that takes two different values for the two different pipes.

The reference to a PVD is in this general format:

      ``{PvdName:N1=V1:N2=V2:...}``

The PvdName is the name of the PVD.
You supply the value of each variable in the PVD using N=V, where N is the name of the variable, and V is the value.
Note that the V can even reference other variables!

Note that if there is a value defined for N outside of the reference, the supplied value in the reference takes precedence.
For example, if your reference supplied a value for ``pipe_token``, then the value you supplied will take precedence over the one defined at the beginning of the file:

      ``"{@pipe_def:pipe_id=task_pipe:pipe_token=pipe_789}"``

In this case, the value of the ``pipe_token`` when creating the pipe ``task_pipe`` will be ``pipe_789``, instead of ``pipe_123`` as defined at the beginning of the file.

Predefined Job Configuration Variables
======================================

The following are predefined variables that can be configured in job config files.
The default values of these variables are usually good enough. However, you may change them to different values in some specific cases.

Runner Sync
-----------

When a job is deployed, dedicated job-specific processes are created throughout the system for the execution of the job.
Specifically, a dedicated server process is created to perform server-side logic; and dedicated client processes (one process for each site) are created to perform client-side logic.
This design allows multiple jobs to be running in their isolated space at the same time. The success or failure of a job won't interfere with the execution of other jobs.

The task-based interactions between a FL client and the FL server is done with the ClientRunner on the client side and the ServerRunner on the server side.
When the job is deployed, the order of the job process creation is not guaranteed - the server-side job process may be started before or after any client-side job process.

To ensure that the ClientRunner does not start to fetch tasks from the ServerRunner, the two runners need to be synchronized first.
Specifically, the ClientRunner keeps sending a "runner sync" request to the ServerRunner until a response is received.

The behavior of the "runner sync" process can be configured with two variables:

runner_sync_timeout
^^^^^^^^^^^^^^^^^^^

This variable is for the client-side configuration (config_fed_client.json).

This runner_sync_timeout specifies the timeout value for the "runner sync" request.
If a response is not received from the Server within this specified value, then another "runner sync" request will be sent.

The default value is 2.0 seconds.

max_runner_sync_tries
^^^^^^^^^^^^^^^^^^^^^

This variable is for the client-side configuration (config_fed_client.json).

This variable specifies the max number of "runner sync" messages to be sent before receiving a response from the server.
If a response is still not received after this many tries, the client's job process will terminate.

The default value is 30.

The default settings of these two variables mean that if the ClientRunner and the ServerRunner are not synched within one minute, the client will terminate.
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
You can set the ``get_task_timeout`` variable to specify how long to wait for the response from the server.
If a response is not received from the server within the specified time, the client will try again.

It is crucial to set this variable to a proper value.
If this value is too short for the server to deliver the response to the client in time, then the server may get repeated requests for the same task.
This can cause the server to run out of memory (since there could be many messages inflight to the same client).

The default value of this variable is 30 seconds. You change its value by setting it in the config_fed_client.json:

``get_task_timeout: 60.0``

Submit Task Result
------------------

The client submits the task result to the server after the task is completed. You can set the ``submit_task_result_timeout`` variable to specify how long to wait for the response from the server. If a response is not received from the server within the specified time, the client will try to send the result again until it succeeds.

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

The default value is 30.0 seconds.

You can tune this value up or down depending on your communication network's behavior.

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
