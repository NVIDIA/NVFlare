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

   usage: nvflare config [-h] [--poc.workspace [POC_WORKSPACE_DIR]]
                         [-jt [JOB_TEMPLATES_DIR]]
                         [--job_templates_dir [JOB_TEMPLATES_DIR]] [-debug] [--schema]

*****************
Common Examples
*****************

Startup kit paths are managed by :ref:`kit_command`, not ``nvflare config``:

.. code-block:: shell

   nvflare kit add project_admin /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com
   nvflare kit use project_admin

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
- ``startup_kits.active`` and ``startup_kits.entries`` are managed by ``nvflare kit``.
- ``poc.startup_kit`` and ``prod.startup_kit`` are not read or written.
- The legacy compatibility alias ``-pw`` is still accepted for ``--poc.workspace``,
  but it is hidden from help and docs.

For startup kit registration and switching, see :ref:`kit_command`.
