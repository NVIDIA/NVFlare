.. _config_command:

#########################
Config Command
#########################

Use ``nvflare config kit`` to manage local startup kit registration and
activation. ``nvflare config`` is the parent command namespace for local CLI
settings; normal users should not need to edit or reason about the underlying
``~/.nvflare/config.conf`` storage layout.

***********************
Command Usage
***********************

.. code-block:: none

   usage: nvflare config [-h] [--schema] {kit} ...

*****************
Common Examples
*****************

Register and activate a startup kit:

.. code-block:: shell

   nvflare config kit add project_admin /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com
   nvflare config kit use project_admin

Configuration notes:

- The saved config format is normalized to v2 with ``version = 2`` as the first line.
- ``startup_kits.active`` and ``startup_kits.entries`` are managed by ``nvflare config kit``.

For startup kit registration and switching details, see :ref:`kit_command`.
