.. _application:

########################
NVIDIA FLARE Application
########################

To upload and run your FL application with NVIDIA FLARE, you need to put required files into an application folder.
The structure of the folder needs to be::

    app_folder/
        config/
            config_fed_client.json
            config_fed_server.json
        custom/
            [any of your custom code].py
            [another file with custom code].py
            ...
        resources/
            log.config

.. note::

    The configuration JSON files config_fed_server.json and config_fed_client.json may be in the root folder of the FL
    application or in a sub-folder (for example: config) of the FL application.

***********************
FL server configuration
***********************

``config_fed_server.json`` is the FL server configuration file.

Example:

.. literalinclude:: ../resources/config_fed_server.json
    :language: json

.. csv-table::
    :header: Key, Notes

    format_version, The NVIDIA FLARE version for this config
    server, Specify server-specific attributes like heart_beat_timeout for seconds before the heart beat times out
    task_data_filters, "What filters to apply to data leaving server, see :ref:`filters`"
    task_result_filters, "What filters to apply to data arriving to server, see :ref:`filters`"
    components, All of the Components to use
    workflows, "What Workflows to use, see :ref:`controllers`"

***********************
FL client configuration
***********************

``config_fed_client.json`` is the FL client configuration file.

Example:

.. literalinclude:: ../resources/config_fed_client.json
    :language: json

.. csv-table::
    :header: Key, Notes

    format_version, The NVIDIA FLARE version for this config
    executors, The configuration for Tasks and Executors which now includes Trainers
    task_data_filters, "What filters to apply to data arriving at client, see :ref:`filters`"
    task_result_filters, "What filters to apply to data leaving client, :ref:`filters`"
    components, All of the Components to use


.. _custom_code:

***********
Custom code
***********

You can write your own components and bring your own code (BYOC) following the :ref:`programming_guide`.

To use it in your application, put the code inside the "custom" folder of the application folder and make sure BYOC is
enabled and allowed.

In your server or client config, use path to refer to that component.

Custom code config example
==========================
For example, with a ``SimpleTrainer`` class stored in a file ``my_trainer.py`` inside the custom folder,
the client config should have the following in order to configure it as an Executor::

    ...
    "executor": {
      "path": "my_trainer.SimpleTrainer",
      "args": {}
    },
    ...

.. note::

    In some examples, "name" is used to identify the class of a component instead of "path". When a component is
    specified by "name", the system will scan the ``nvflare`` package for the class name, so this notation may be
    simpler than using the entire path to the class for the component if it is inside the ``nvflare`` package. If the
    component you are using is not in the ``nvflare`` package, please use "path" and make sure that the path is
    accessible with the configured PYTHONPATH.

.. _troubleshooting_byoc:

Troubleshooting BYOC
====================
There is an ``enable_byoc`` flag for each participant that can be set at provisioning, and if that is disabled, even if
you have custom code in your application folder, it will not be loaded. There is also a setting for ``allow_byoc``
through the authorization rule groups. This controls whether or not apps containing BYOC code will be allowed to be
uploaded and deployed.

*********
Resources
*********

A ``log.config`` is needed inside the resources folder.
This file is for the Python logger to use.
If you don't want to customize the log behavior, you can use the same ``log.config`` from one of
the example application folder.

.. literalinclude:: ../resources/log.config
