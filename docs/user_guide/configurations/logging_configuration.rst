.. _logging_configuration:

##################################
NVIDIA FLARE Logging Configuration
##################################

NVFLARE uses python logging, specifically fileConfig( configure https://docs.python.org/3/library/logging.config.html) 

We provide default logging configuration files for NVFLARE sub-systems. You can overwrite these logging configurations by modifying the configuration files. 

************************************
Logging configuration files location
************************************

Startup kits log configurations
===============================

The log configuration files are located in the startup kits under the local directory.

If you search for the ``log.config.*`` files in the startup kits workspace, you will find the following files:

.. code-block:: shell

    find . -name "log.config.*"

    ./site-1/local/log.config.default
    ./site-2/local/log.config.default
    ./server1/local/log.config.default

The server ``log.config.default`` is the default logging configuration used by the FL Server and clients. To overwrite the default,
you can change ``log.config.default`` to ``log.config`` and modify the configuration.

POC log configurations
======================
Similarly, if you search the POC workspace, you will find the following:

.. code-block:: shell

    find /tmp/nvflare/poc  -name "log.config*"

    /tmp/nvflare/poc/server/local/log.config
    /tmp/nvflare/poc/site-1/local/log.config
    /tmp/nvflare/poc/site-2/local/log.config

You can directly modify ``log.config`` to make changes.

Simulator log configuration
===========================

Simulator logging configuration uses the default log configuration. If you want to overwrite the default configuration, you can add ``log.config`` to
``<simulator_workspace>/startup/log.config``.

For example, for hello-numpy-sag examples, the CLI command is:

.. code-block:: shell

    nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 hello-world/hello-numpy-sag/jobs/hello-numpy-sag

If the workspace is ``/tmp/nvflare/hello-numpy-sag/``, then you can add log.config in ``/tmp/nvflare/hello-numpy-sag/startup/log.config`` to overwrite the default one. 

Configuration logging
=====================

The default logging file-config based logging configuration is the following:

.. code-block:: shell

    [loggers]
    keys=root

    [handlers]
    keys=consoleHandler

    [formatters]
    keys=fullFormatter

    [logger_root]
    level=INFO
    handlers=consoleHandler

    [handler_consoleHandler]
    class=StreamHandler
    level=DEBUG
    formatter=fullFormatter
    args=(sys.stdout,)

    [formatter_fullFormatter]
    format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

Suppose we would like to change the logging for :class:`ScatterAndGather<nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather>` from to INFO to ERROR,
we can do the following:

.. code-block:: shell

    [loggers]
    keys=root, ScatterAndGather

    [handlers]
    keys=consoleHandler

    [formatters]
    keys=fullFormatter

    [logger_root]
    level=INFO
    handlers=consoleHandler

    [logger_ScatterAndGather]
    level=ERROR
    handlers=consoleHandler
    qualname=ScatterAndGather
    propagate=0

    [handler_consoleHandler]
    class=StreamHandler
    level=DEBUG
    formatter=fullFormatter
    args=(sys.stdout,)

    [formatter_fullFormatter]
    format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
