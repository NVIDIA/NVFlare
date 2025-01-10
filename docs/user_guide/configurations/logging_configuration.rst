.. _logging_configuration:

#####################
Logging Configuration
#####################

FLARE uses python logging with the dictConfig API (https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig).
FLARE Loggers are designed to follow the package level hierarchy using dot separated logger names in order to faciliate granular control at different levels.

We provide a :ref:`Default Logging Configuration` file **log_config.json.default** for all NVFLARE sub-systems with pre-configured handlers for console colors, logs, error logs, structured json logs, and fl training logs.
You can overwrite the default configuration by :ref:`Modifying Logging Configurations` files.
Users can also change the logging configuration during runtime by using the :ref:`Dynamic Logging Configuration Commands` **configure_site_log** and **configure_job_log**.

**********************************
Logging Configuration and Features
**********************************

Default Logging Configuration
=============================

The default logging configuration json file (**log_config.json.default**):

.. code-block:: json

    {
        "version": 1,
        "disable_existing_loggers": false,
        "formatters": {
            "baseFormatter": {
                "()": "nvflare.fuel.utils.log_utils.BaseFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "colorFormatter": {
                "()": "nvflare.fuel.utils.log_utils.ColorFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "jsonFormatter": {
                "()": "nvflare.fuel.utils.log_utils.JsonFormatter",
                "fmt": "%(asctime)s - %(name)s - %(fullName)s - %(levelname)s - %(message)s"
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
                "formatter": "colorFormatter",
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


We use multiple handlers to output log records to the console and various log files.

- consoleHandler with colorFormatter for level/logname based log coloring to the stdout console
- logFileHandler with baseFormatter to write all logs to log.txt
- errorFileHandler  with baseFormatter to write error level logs to log_error.txt
- jsonFileHandler with jsonFormatter to write json formatted logs to log.json
- FLFileHandler with FLFilter to write fl training and custom logs to log_fl.txt
- root logger set to INFO with the above handlers

For more details see the sections below.


Formatters
==========

`Formatters <https://docs.python.org/3/library/logging.html#logging.Formatter>`_ are used to specify the format of log records.
We provide several useful formatters by default:

BaseFormatter
-------------
The :class:`BaseFormatter<nvflare.fuel.utils.log_utils.base_fed_job.BaseFormatter>` is the default formatter serving as the base class for other FLARE formatters.

- All the default formatter arguments such as **fmt** and **datefmt** can be specified.
- Shortens the **record.name** to the base names, and adds a **record.fullName** property.


ColorFormatter
--------------
The :class:`ColorFormatter<nvflare.fuel.utils.log_utils.base_fed_job.ColorFormatter>` uses ANSI color codes to format log records based on log level and/or logger names.

- **level_colors**: dict of levelname: ANSI color. Defaults to ANSIColor.DEFAULT_LEVEL_COLORS.
- **logger_colors**: dict of loggername: ANSI color. Defaults to {}.

We provide an :class:`ANSIColor<nvflare.fuel.utils.log_utils.base_fed_job.ANSIColor>` class with commonly used colors and default mappings for log levels.
To customize the colors, use either string of a color name specifed in ANSIColor.COLORS, or an ANSI color code (semicolons can be used for additional ANSI arguments).


JsonFormatter
-------------
The :class:`JsonFormatter<nvflare.fuel.utils.log_utils.base_fed_job.JsonFormatter>` converts the log records into a json string.

The `extract_brackets` argument also parses the message and adds a nested **fl_ctx_fields** object with the fl context keys and values.


Filters
=======

`Filters <https://docs.python.org/3/library/logging.html#logging.Filter>`_ are used to allow certain log records to pass through based on a a specified criteria. 

LoggerNameFilter
----------------
:class:`LoggerNameFilter<nvflare.fuel.utils.log_utils.base_fed_job.LoggerNameFilter>` enables a list of logger_names to be filtered.
Filters utilize the logger hierarchy, so any descendants of the specified names will also be allowed.

This is used in our FLFilter, which filters loggers related to fl training or custom code.

.. code-block::

    "FLFilter": {
        "()": "nvflare.fuel.utils.log_utils.LoggerNameFilter",
        "logger_names": ["custom", "nvflare.app_common", "nvflare.app_opt"]
    }


********************************
Modifying Logging Configurations
********************************

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

POC log configurations
======================
Similarly, if you search the POC workspace, you will find the following:

.. code-block:: shell

    find /tmp/nvflare/poc  -name "log_config.json*"

    /tmp/nvflare/poc/server/local/log_config.json
    /tmp/nvflare/poc/site-1/local/log_config.json
    /tmp/nvflare/poc/site-2/local/log_config.json

You can directly modify ``log_config.json`` to make changes.

Simulator log configuration
===========================

Simulator logging configuration uses the default log configuration. If you want to overwrite the default configuration, you can add ``log_config.json`` to
``<simulator_workspace>/startup/log_config.json``.

For example, for hello-numpy-sag examples, the CLI command is:

.. code-block:: shell

    nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 hello-world/hello-numpy-sag/jobs/hello-numpy-sag

If the workspace is ``/tmp/nvflare/hello-numpy-sag/``, then you can add log_config.json in ``/tmp/nvflare/hello-numpy-sag/startup/log_config.json`` to overwrite the default one.

Users can also specify a log configuration file in the command with the ``-l`` simulator argument:

.. code-block:: shell

    nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 hello-world/hello-numpy-sag/jobs/hello-numpy-sag -l log_config.json


**************************************
Dynamic Logging Configuration Commands
**************************************

See :ref:`operating_nvflare` for details on the commands and :ref:`command_categories` for the default authorization policy.

- **target** can be one of server, client <clients>..., or all
- **config** can be a path to a log configuration file, a log level name/number, or "reload" to read the current log config

.. code-block:: shell

    configure_site_log target config

Configures the target site logging (does not affect jobs).


.. code-block:: shell

    configure_job_log job_id target config

Configures the target job logging.