.. _config_command:

#########################
Config Command
#########################

Use ``nvflare config`` to manage local CLI settings stored in
``~/.nvflare/config.conf``. The root command manages the POC workspace, and
``nvflare config kit`` manages startup kit registration and activation.

***********************
Command Usage
***********************

.. code-block:: none

   usage: nvflare config [-h] [--poc.workspace [POC_WORKSPACE_DIR]]
                         [-debug] [--schema] {kit} ...

*****************
Common Examples
*****************

Set the default POC workspace:

.. code-block:: shell

   nvflare config --poc.workspace /tmp/nvflare/poc

Register and activate a startup kit:

.. code-block:: shell

   nvflare config kit add project_admin /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com
   nvflare config kit use project_admin

Configuration notes:

- The saved config format is normalized to v2 with ``version = 2`` as the first line.
- ``startup_kits.active`` and ``startup_kits.entries`` are managed by ``nvflare config kit``.
- ``poc.startup_kit`` and ``prod.startup_kit`` are not read or written.
- The compatibility alias ``-pw`` is still accepted for ``--poc.workspace``.

For startup kit registration and switching details, see :ref:`kit_command`.
