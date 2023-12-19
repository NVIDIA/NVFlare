.. _variable_resolution:

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
- JOB_ID - Job ID
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