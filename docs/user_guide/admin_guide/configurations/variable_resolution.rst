.. _variable_resolution:

Variable Resolution in Job Configuration
========================================

FLARE jobs are defined with configuration files: ``config_fed_client.json`` and ``config_fed_server.json``.
These two files configure the components (Python objects) used for the server process and the FL client processes.
The component configuration includes information about the class path of the Python object (via ``path`` or ``class_path``), and arguments for the object's constructor.
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

System variables use the same reference syntax as user-defined variables.

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

Suppose a client job needs two event converters. Their configurations repeat the same component path and event prefix, while the component ID and event name differ:

.. code-block:: json

   {
      "format_version": 2,
      "event_prefix": "fed.",
      "executors": [
         {
            "tasks": [
               "train"
            ],
            "executor": {
               "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
               "args": {
                  "execution_mode": "in_process",
                  "task_script_path": "trainer.py"
               }
            }
         }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": [
         {
            "id": "metrics_event_converter",
            "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
            "args": {
               "events_to_convert": [
                  "analytix_log_stats"
               ],
               "fed_event_prefix": "{event_prefix}"
            }
         },
         {
            "id": "status_event_converter",
            "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
            "args": {
               "events_to_convert": [
                  "training_status"
               ],
               "fed_event_prefix": "{event_prefix}"
            }
         }
      ]
   }


The two converter definitions have the same structure. Repeating that structure is tedious and can lead to inconsistent updates.

Simple variable references can remove some repeated scalar values, such as the component path and event prefix:

.. code-block:: json

   {
      "format_version": 2,
      "converter_path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
      "event_prefix": "fed.",
      "executors": [
         {
            "tasks": [
               "train"
            ],
            "executor": {
               "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
               "args": {
                  "execution_mode": "in_process",
                  "task_script_path": "trainer.py"
               }
            }
         }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "components": [
         {
            "id": "metrics_event_converter",
            "path": "{converter_path}",
            "args": {
               "events_to_convert": [
                  "analytix_log_stats"
               ],
               "fed_event_prefix": "{event_prefix}"
            }
         },
         {
            "id": "status_event_converter",
            "path": "{converter_path}",
            "args": {
               "events_to_convert": [
                  "training_status"
               ],
               "fed_event_prefix": "{event_prefix}"
            }
         }
      ]
   }

This removes duplicate scalar values, but the surrounding component structure is still repeated.

Using a Parameterized Variable Definition, we can define the component structure once and supply the values that differ at each reference:

.. code-block:: json

   {
      "format_version": 2,
      "converter_path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
      "event_prefix": "fed.",
      "executors": [
         {
            "tasks": [
               "train"
            ],
            "executor": {
               "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
               "args": {
                  "execution_mode": "in_process",
                  "task_script_path": "trainer.py"
               }
            }
         }
      ],
      "task_result_filters": [],
      "task_data_filters": [],
      "@event_converter": {
         "id": "{component_id}",
         "path": "{converter_path}",
         "args": {
            "events_to_convert": [
               "{event_name}"
            ],
            "fed_event_prefix": "{event_prefix}"
         }
      },
      "components": [
         "{@event_converter:component_id=metrics_event_converter:event_name=analytix_log_stats}",
         "{@event_converter:component_id=status_event_converter:event_name=training_status}"
      ]
   }

Here, ``@event_converter`` is a parameterized variable definition (PVD).
The name of a PVD must start with the ``@`` sign. The PVD is usually defined with references to other variables, and the values can be provided at the time the PVD is referenced.
In this example, the PVD defines an event-converter configuration template that resolves to a concrete component configuration.
The ``components`` section uses it twice, supplying a different ``component_id`` and ``event_name`` for each converter.

A PVD can only be referenced with SVR (simple variable reference).
To reference a PVD, you provide values for any variables in the PVD.
In this example, ``component_id`` and ``event_name`` take different values for the two converters.

The reference to a PVD is in this general format:

``{PvdName:N1=V1:N2=V2:...}``

The PvdName is the name of the PVD.
You supply the value of each variable in the PVD using N=V, where N is the name of the variable, and V is the value.
Note that the V can even reference other variables!

Note that if there is a value defined for N outside of the reference, the supplied value in the reference takes precedence.
For example, a reference can override the ``event_prefix`` defined at the beginning of the file:

``"{@event_converter:component_id=audit_event_converter:event_name=audit_event:event_prefix=global.}"``

In this case, the ``audit_event_converter`` uses ``global.`` instead of the default ``fed.`` prefix.
