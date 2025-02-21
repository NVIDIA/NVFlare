.. _logging_configuration:

#####################
Logging Configuration
#####################

FLARE uses python logging with the `dictConfig API <https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig>`_ following the `configuration dictionary schema <https://docs.python.org/3/library/logging.config.html#configuration-dictionary-schema>`_.
FLARE Loggers are designed to follow the package level hierarchy using dot separated logger names in order to faciliate granular control at different levels.

We provide a :ref:`Default Logging Configuration <default_logging_configuration>` file **log_config.json.default** for all NVFLARE sub-systems with pre-configured handlers for console level colors, logs, error logs, structured json logs, and fl training logs.

Overwrite the default configuration by :ref:`Modifying Logging Configurations <modifying_logging_configurations>` files,
or change the logging configuration during runtime by using the :ref:`Dynamic Logging Configuration Commands <dynamic_logging_configuration_commands>` ``configure_site_log`` and ``configure_job_log``.

**********************************
Logging Configuration and Features
**********************************

.. _default_logging_configuration:
Default Logging Configuration
=============================

The default logging configuration json file (**log_config.json.default**, ``default``) is divided into 3 main sections: formatters, handlers, and loggers.
This file can be found at :github_nvflare_link:`log_config.json <nvflare/fuel/utils/log_config.json>`.
See the `configuration dictionary schema <(https://docs.python.org/3/library/logging.config.html#configuration-dictionary-schema)>`_ for more details.

.. code-block:: json

    {
        "version": 1,
        "disable_existing_loggers": false,
        "formatters": {
            "baseFormatter": {
                "()": "nvflare.fuel.utils.log_utils.BaseFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(fl_ctx)s - %(message)s"
            },
            "consoleFormatter": {
                "()": "nvflare.fuel.utils.log_utils.ColorFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(fl_ctx)s - %(message)s"
            },
            "jsonFormatter": {
                "()": "nvflare.fuel.utils.log_utils.JsonFormatter",
                "fmt": "%(asctime)s - %(name)s - %(fullName)s - %(levelname)s - %(fl_ctx)s - %(message)s"
            }
        },
        "filters": {
            "FLFilter": {
                "()": "nvflare.fuel.utils.log_utils.LoggerNameFilter",
                "logger_names": ["custom", "nvflare.app_common", "nvflare.app_opt"]
            }
        },
        "handlers": {
            "consoleHandler": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "consoleFormatter",
                "filters": [],
                "stream": "ext://sys.stdout"
            },
            "logFileHandler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "baseFormatter",
                "filename": "log.txt",
                "mode": "a",
                "maxBytes": 20971520,
                "backupCount": 10
            },
            "errorFileHandler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "baseFormatter",
                "filename": "log_error.txt",
                "mode": "a",
                "maxBytes": 20971520,
                "backupCount": 10
            },
            "jsonFileHandler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "jsonFormatter",
                "filename": "log.json",
                "mode": "a",
                "maxBytes": 20971520,
                "backupCount": 10
            },
            "FLFileHandler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "baseFormatter",
                "filters": ["FLFilter"],
                "filename": "log_fl.txt",
                "mode": "a",
                "maxBytes": 20971520,
                "backupCount": 10,
                "delay": true
            }
        },
        "loggers": {
            "root": {
                "level": "INFO",
                "handlers": ["consoleHandler", "logFileHandler", "errorFileHandler", "jsonFileHandler", "FLFileHandler"]
            }
        }
    }

We use different formatters, filters, and handlers to output log records to the console and various log files, which are described in more detail below.

Formatters
==========

`Formatters <https://docs.python.org/3/library/logging.html#formatter-objects>`_ are used to specify the format of log records.
We provide several useful formatters by default:

BaseFormatter
-------------
The :class:`BaseFormatter<nvflare.fuel.utils.log_utils.BaseFormatter>` is the default formatter serving as the base class for other FLARE formatters.

- All the default `Formatter <https://docs.python.org/3/library/logging.html#logging.Formatter>`_ arguments such as **fmt** with `log record attributes <https://docs.python.org/3/library/logging.html#logrecord-attributes>`_ and the **datefmt** `date format string <https://docs.python.org/3/library/logging.html#logging.Formatter.formatTime>`_ can be specified.
- The **record.name** is shortened to the logger base name, and **record.fullName** is set to the logger full name.

Example configuration and output:

.. code-block:: json

    "baseFormatter": {
        "()": "nvflare.fuel.utils.log_utils.BaseFormatter",
        "fmt": "%(asctime)s - %(name)s - %(fullName)s - %(levelname)s - %(fl_ctx)s - %(message)s",
        "datefmt": "%m-%d-%Y- %H:%M:%S"
    }

.. code-block:: shell

    01-14-2025 14:44:46 - PTInProcessClientAPIExecutor - nvflare.app_opt.pt.in_process_client_api_executor.PTInProcessClientAPIExecutor - INFO - [identity=site-1, run=fc711945-a7cf-4834-9fc4-aa9cb60e327b, peer=example_project, peer_run=fc711945-a7cf-4834-9fc4-aa9cb60e327b, task_name=train, task_id=a16b7a02-b2ea-4eb5-895a-b40d507b2c5c] - execute for task (train)


ColorFormatter
--------------
The :class:`ColorFormatter<nvflare.fuel.utils.log_utils.ColorFormatter>` uses ANSI color codes to format log records based on log level and/or logger names.

We provide the :class:`ANSIColor<nvflare.fuel.utils.log_utils.ANSIColor>` class for commonly used colors and default mappings for log levels.
To customize the colors, use either string of a color name specifed in ANSIColor.COLORS, or an ANSI color code (semicolons can be used for additional ANSI arguments).

- **level_colors**: dict of levelname: ANSI color. Defaults to ANSIColor.DEFAULT_LEVEL_COLORS.
- **logger_colors**: dict of loggername: ANSI color. Defaults to {}.

Example configuration:

.. code-block:: json

    "consoleFormatter": {
        "()": "nvflare.fuel.utils.log_utils.ColorFormatter",
        "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(fl_ctx)s - %(message)s",
        "level_colors": {
            "NOTSET": "grey",
            "DEBUG": "grey",
            "INFO": "grey",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        },
        "logger_colors": {
            "nvflare.app_common": "blue",
            "nvflare.app_opt": "38;5;212"
        }
    }


JsonFormatter
-------------
The :class:`JsonFormatter<nvflare.fuel.utils.log_utils.JsonFormatter>` converts the log records into a json string.

Example configuration and output:

.. code-block:: json

    "jsonFormatter": {
        "()": "nvflare.fuel.utils.log_utils.JsonFormatter",
        "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(fl_ctx)s - %(message)s"
    }

.. code-block:: json

    {"asctime": "2025-01-14 14:44:46,559", "name": "PTInProcessClientAPIExecutor", "fullName": "nvflare.app_opt.pt.in_process_client_api_executor.PTInProcessClientAPIExecutor", "levelname": "INFO", "fl_ctx": "[identity=site-1, run=fc711945-a7cf-4834-9fc4-aa9cb60e327b, peer=example_project, peer_run=fc711945-a7cf-4834-9fc4-aa9cb60e327b, task_name=train, task_id=a16b7a02-b2ea-4eb5-895a-b40d507b2c5c]", "message": "execute for task (train)"}


Filters
=======

`Filters <https://docs.python.org/3/library/logging.html#filter-objects>`_ are used to allow certain log records to pass through based on specified criteria.

LoggerNameFilter
----------------
:class:`LoggerNameFilter<nvflare.fuel.utils.log_utils.LoggerNameFilter>` filters loggers based on a list of logger_names.
Filters utilize the logger hierarchy, so any descendants of the specified names will also be allowed through the filter.

- **logger_names**: list of logger names to allow through filter
- **exclude_logger_names**: list of logger names to disallow through filter (takes precedence over allowing from logger_names)

We leverage this in our FLFilter, which filters loggers related to fl training or custom code.

.. code-block:: json

    "FLFilter": {
        "()": "nvflare.fuel.utils.log_utils.LoggerNameFilter",
        "logger_names": ["custom", "nvflare.app_common", "nvflare.app_opt"]
    }

Handlers
========
`Handlers <https://docs.python.org/3/library/logging.html#handler-objects>`_ are responsible for sending log records to a destination, while applying any specified Formatter or Filters (applied sequentially).

consoleHandler
--------------

The consoleHandler uses the `StreamHandler <https://docs.python.org/3/library/logging.handlers.html#streamhandler>`_ to send logging output to a stream, such as sys.stdout.

Example configuration:

.. code-block:: json

    "consoleHandler": {
        "class": "logging.StreamHandler",
        "level": "DEBUG",
        "formatter": "consoleFormatter",
        "filters": ["FLFilter"],
        "stream": "ext://sys.stdout"
    }


FileHandlers
------------
We use `FileHandlers <https://docs.python.org/3/library/logging.handlers.html#filehandler>`_ to send different formatted and filtered log records to different files.

In the pre-configured handlers, more specifically we utilize the `RotatingFileHandler <https://docs.python.org/3/library/logging.handlers.html#rotatingfilehandler>`_ to rollover to backup files after a certain file size is reached.
FLARE dynamically interprets the ``filename`` to be relative to the either the workspace root directory (for site log files), or the run directory (for job log files).

Example configuration:

.. code-block:: json

    "logFileHandler": {
        "class": "logging.handlers.RotatingFileHandler",
        "level": "DEBUG",
        "formatter": "baseFormatter",
        "filename": "log.txt",
        "mode": "a",
        "maxBytes": 20971520,
        "backupCount": 10
    }

The following log file handlers are pre-configured:

- logFileHandler with baseFormatter to write all logs to ``log.txt``
- errorFileHandler  with baseFormatter and level "ERROR" to write error level logs to ``log_error.txt``
- jsonFileHandler with jsonFormatter to write json formatted logs to ``log.json``
- FLFileHandler with baseFormatter and FLFilter to write fl training and custom logs to ``log_fl.txt``

.. _loggers:
Loggers
=======

Loggers can be configured in the logger section to have a level and handlers.

We define the root logger with INFO level and add the desired handlers.

.. code-block:: json

    "root": {
        "level": "INFO",
        "handlers": ["consoleHandler", "logFileHandler", "errorFileHandler", "jsonFileHandler", "FLFileHandler"]
    }

Given the hierarchical structure of loggers, specific loggers can be configured using their dot separated names.
Furthermore, any intermediate logger parents are already created and are configureable.

When creating loggers for FLARE, we provide several convenience functions to help adhere to the package logger hierarchy:

- :func:`get_obj_logger<nvflare.fuel.utils.log_utils.get_obj_logger>` for classes
- :func:`get_script_logger<nvflare.fuel.utils.log_utils.get_script_logger>` for scripts (if not in a package, default to custom.<script_file_name>)
- :func:`get_module_logger<nvflare.fuel.utils.log_utils.get_module_logger>` for modules


.. _modifying_logging_configurations:
********************************
Modifying Logging Configurations
********************************

Simulator log configuration
===========================

Users can specify a log configuration in the simulator command with the ``-l`` simulator argument:

.. code-block:: shell

    nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 hello-world/hello-numpy-sag/jobs/hello-numpy-sag -l log_config.json

Or using the ``log_config`` argument of the Job API simulator run:

.. code-block:: python

    job.simulator_run("/tmp/nvflare/hello-numpy-sag", log_config="log_config.json")


The log config argument be one of the following:

- path to a log configuration json file (``/path/to/my_log_config.json``)
- preconfigured log mode (``default``, ``concise``, ``verbose``)
- log level name or number (``debug``, ``info``, ``warning``, ``error``, ``critical``, ``30``)


POC log configurations
======================
If you search the POC workspace, you will find the following:

.. code-block:: shell

    find /tmp/nvflare/poc  -name "log_config.json*"

    /tmp/nvflare/poc/server/local/log_config.json.default
    /tmp/nvflare/poc/site-1/local/log_config.json.default
    /tmp/nvflare/poc/site-2/local/log_config.json.default

You can add a ``log_config.json`` to make changes.

We also recommend using the :ref:`Dynamic Logging Configuration Commands <dynamic_logging_configuration_commands>`.

Startup kits log configurations
===============================

The log configuration files are located in the startup kits under the local directory.

If you search for the ``log_config.json.*`` files in the startup kits workspace, you will find the following files:

.. code-block:: shell

    find . -name "log_config.json.*"

    ./site-1/local/log_config.json.default
    ./site-2/local/log_config.json.default
    ./server1/local/log_config.json.default

The server ``log_config.json.default`` is the default logging configuration used by the FL Server and clients. To overwrite the default,
you can change ``log_config.json.default`` to ``log_config.json`` and modify the configuration.

We also recommend using the :ref:`Dynamic Logging Configuration Commands <dynamic_logging_configuration_commands>`.

.. _dynamic_logging_configuration_commands:
**************************************
Dynamic Logging Configuration Commands
**************************************

When running the FLARE system (POC mode or production mode), there are two sets of logs: the site logs and job logs.
The current site log configuration will be used for the site logs as well as the log config of any new job started on that site.
In order to access the generated logs in the workspaces refer to :ref:`access_server_workspace` and :ref:`client_workspace`.

We provide two admin commands to enable users to dynamically configure the site or job level logging when running the FLARE system.
Note these command effects will last until reconfiguration or as long as the corresponding site or job is running.
However these commands do not overwrite the log configuration file in the workspace- the log configuration file can be reloaded using "reload".

- **target**: ``server``, ``client <clients>...``, or ``all``
- **config**: log configuration

    - path to a json log configuration file (``/path/to/my_log_config.json``)
    - predefined log mode (``default``, ``concise``, ``verbose``)
    - log level name/number (``debug``, ``INFO``, ``30``)
    - read the current log configuration file from the workspace (``reload``)

To configure the target site logging (does not affect currently running jobs):

.. code-block:: shell

    configure_site_log target config

To configure the target job logging (the job must be running):

.. code-block:: shell

    configure_job_log job_id target config

See :ref:`operating_nvflare` for how to use commands and :ref:`command_categories` for the default authorization policy.
