.. _application:

########################
NVIDIA FLARE Application
########################

The NVIDIA FLARE application defines how the server and client should run.
Note that in the scope of one job, each site will only run one application.

The structure of the app folder needs to be::

    app_folder/
        config/
            config_fed_client.json [required if this app needs to be deployed to clients]
            config_fed_server.json [required if this app needs to be deployed to server]
        custom/
            [any of your custom code].py
            [another file with custom code].py
            ...
        resources/
            log.config

.. note::

    Note that apps can be configured to run on certain sites in a job's deploy_map configuration.
    An application can also be run without a job.
    To do this, simply submit an app as a job and a default deploy map of all sites will be used.

.. note::

    If the same application is going to be deployed on both server and clients, it can contain both
    ``config_fed_server.json`` and ``config_fed_client.json``

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

    Configuration of Executor Tasks is ignored here.

Please follow :ref:`getting_started` to learn more.

.. _troubleshooting_byoc:

Troubleshooting BYOC
====================
In 2.2.1, authorization has been redesigned and BYOC is no longer controlled through settings at provisioning, but
instead by each site's authorization.json (in the local folder of the workspace). BYOC is a right and can be restricted
to certain roles or even orgs or users. See :ref:`federated_authorization` for details.

*********
Resources
*********

A ``log.config`` is needed inside the resources folder.
This file is for the Python logger to use.
If you don't want to customize the log behavior, you can use the same ``log.config`` from one of
the example application folder.

.. literalinclude:: ../resources/log.config
