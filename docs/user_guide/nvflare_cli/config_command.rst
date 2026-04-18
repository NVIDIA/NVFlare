.. _config_command:

#########################
Config Command
#########################

Use ``nvflare config`` to manage local CLI settings stored in
``~/.nvflare/config.conf``.

***********************
Command Usage
***********************

.. code-block:: none

   usage: nvflare config [-h] [-d [STARTUP_KIT_DIR]]
                         [--poc.startup_kit [POC_STARTUP_KIT_DIR]]
                         [--prod.startup_kit [PROD_STARTUP_KIT_DIR]]
                         [-pw [POC_WORKSPACE_DIR]] [--poc.workspace [POC_WORKSPACE_DIR_V2]]
                         [-jt [JOB_TEMPLATES_DIR]]
                         [--job_templates_dir [JOB_TEMPLATES_DIR]] [-debug] [--schema]

*****************
Common Examples
*****************

Set the default POC admin startup kit directory used by ``nvflare job`` and
``nvflare system``:

.. code-block:: shell

   nvflare config --poc.startup_kit /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com

Set the production admin startup kit directory:

.. code-block:: shell

   nvflare config --prod.startup_kit /path/to/prod/admin_startup_kit

Set the default POC workspace:

.. code-block:: shell

   nvflare config --poc.workspace /tmp/nvflare/poc

Set the job template directory for deprecated ``nvflare job list_templates``:

.. code-block:: shell

   nvflare config --job_templates_dir /path/to/job_templates

.. note::

   Most new job creation workflows use exported job folders or ``nvflare recipe``
   instead of the legacy template commands.

Configuration notes:

- The saved config format is normalized to v2 with ``version = 2`` as the first line.
- ``poc.startup_kit`` and ``prod.startup_kit`` should point to admin startup kit
  directories such as ``.../admin@nvidia.com``.
- The legacy compatibility alias ``--startup_kit_dir`` updates ``poc.startup_kit``
  only. It does not set ``prod.startup_kit``.
