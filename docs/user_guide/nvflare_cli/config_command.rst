.. _config_command:

#########################
Config Command
#########################

Use ``nvflare config`` to manage local CLI settings, including startup kit
registration and activation. Normal users should not need to edit or reason
about the underlying ``~/.nvflare/config.conf`` storage layout.

***********************
Command Usage
***********************

.. code-block:: none

   usage: nvflare config [-h] [--schema] [-d [STARTUP_KIT_DIR]]
                         [-pw [POC_WORKSPACE_DIR]] [-jt [JOB_TEMPLATES_DIR]]
                         {add,use,show,list,remove} ...

*****************
Common Examples
*****************

Register and activate a startup kit:

.. code-block:: shell

   nvflare config add project_admin /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com
   nvflare config use project_admin

Configuration notes:

- The saved config format is normalized to v2 with ``version = 2`` as the first line.
- ``startup_kits.active`` and ``startup_kits.entries`` are managed by ``nvflare config``.
- ``nvflare config show --format json`` and ``nvflare config list --format json``
  include best-effort startup-kit identity, certificate expiration, and local
  stale-path findings for automation.
- ``nvflare config use`` changes global CLI state. Automation should prefer
  per-command ``--kit-id`` or ``--startup-kit`` selectors when running
  server-connected commands.
- ``nvflare config -d/--startup_kit_dir`` remains accepted for compatibility
  with 2.7.x scripts, but is deprecated. Use ``nvflare config add`` and
  ``nvflare config use`` for new workflows.
- ``nvflare config -pw/--poc_workspace_dir`` remains the supported way to set
  the local POC workspace from the root ``config`` command.
- ``nvflare config -jt/--job_templates_dir`` remains accepted for compatibility,
  but job template config is deprecated. Prefer passing custom template
  locations to job commands that need them.
- Development-only spellings such as ``--poc.workspace``, ``--poc.startup_kit``,
  and ``--prod.startup_kit`` are not supported compatibility flags.

For startup kit registration and switching details, see :ref:`kit_command`.
