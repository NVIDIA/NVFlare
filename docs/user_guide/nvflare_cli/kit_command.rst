.. _kit_command:

################################
Startup Kit Config Subcommands
################################

Use ``nvflare config kit`` to register local startup kit paths and choose the active
admin startup kit used by server-connected CLI commands.

The registry is local to the current machine. ``nvflare config kit`` does not contact
the server. It updates ``~/.nvflare/config.conf`` so commands such as
``nvflare job list``, ``nvflare study list``, and ``nvflare system status`` can
connect without repeated startup-kit arguments.

***********************
Command Usage
***********************

.. code-block:: none

   usage: nvflare config kit [-h]  ...

   kit subcommands:
     add       register a startup kit path
     use       activate a registered startup kit
     show      show the configured active startup kit
     list      list registered startup kits
     remove    remove a local startup kit registration

Running ``nvflare config kit`` without a subcommand prints this help text.

*****************
Common Workflow
*****************

For POC, ``nvflare poc prepare`` registers generated admin/user startup kits
and activates the default Project Admin kit automatically:

.. code-block:: shell

   nvflare poc prepare
   nvflare job list

For a production or manually copied startup kit, register and activate it once:

.. code-block:: shell

   nvflare config kit add cancer_lead /secure/startup_kits/cancer/lead@nvidia.com
   nvflare config kit use cancer_lead
   nvflare job list

To switch local identity, activate another registered ID:

.. code-block:: shell

   nvflare config kit add fraud_org_admin /secure/startup_kits/fraud/org_admin@nvidia.com
   nvflare config kit use fraud_org_admin
   nvflare system status

************************
Startup Kit Resolution
************************

Server-connected commands resolve the startup kit in this order:

1. ``NVFLARE_STARTUP_KIT_DIR`` environment variable, when set.
2. ``startup_kits.active`` from ``~/.nvflare/config.conf``.
3. If neither resolves to a valid admin startup kit, the command fails before
   connecting.

The environment variable is intended for automation. For normal interactive
use, register startup kits with ``nvflare config kit add`` and switch with
``nvflare config kit use``.

*********************
Register a Startup Kit
*********************

.. code-block:: shell

   nvflare config kit add <id> <startup-kit-dir>

Arguments and options:

- ``<id>``: local ID used to refer to the startup kit. IDs can be meaningful
  names such as ``cancer_lead`` or identities such as ``lead@nvidia.com``.
- ``<startup-kit-dir>``: admin/user startup kit directory, such as
  ``.../prod_00/admin@nvidia.com``. Passing the
  ``startup`` subdirectory is also accepted and normalized to the participant
  startup kit directory.
- ``--force``: replace an existing local registration for the same ID.
- ``--schema``: print the command schema as JSON and exit.

For an admin/user startup kit, the path must contain:

- ``startup/fed_admin.json``
- ``startup/client.crt``
- ``startup/rootCA.pem``

Site startup kits are not valid here because they are service identities, not
interactive admin/user identities for CLI sessions.

Example:

.. code-block:: shell

   nvflare config kit add cancer_lead /secure/startup_kits/cancer/lead@nvidia.com

Example output:

.. code-block:: none

   registered_startup_kit: cancer_lead
   path: /secure/startup_kits/cancer/lead@nvidia.com
   next_step: nvflare config kit use cancer_lead

``kit add`` registers the path only. It does not make the kit active.

************************
Activate a Startup Kit
************************

.. code-block:: shell

   nvflare config kit use <id>

``kit use`` validates that the registered path is an admin/user startup kit and
then sets ``startup_kits.active``.

Example:

.. code-block:: shell

   nvflare config kit use cancer_lead

Example output:

.. code-block:: none

   active_startup_kit: cancer_lead
   identity: lead@nvidia.com
   cert_role: lead
   path: /secure/startup_kits/cancer/lead@nvidia.com

**********************
Show Active Startup Kit
**********************

.. code-block:: shell

   nvflare config kit show

Example output:

.. code-block:: none

   active: cancer_lead
   path: /secure/startup_kits/cancer/lead@nvidia.com
   config_file: /home/user/.nvflare/config.conf
   status: ok
   identity: lead@nvidia.com
   cert_role: lead

If ``NVFLARE_STARTUP_KIT_DIR`` is set, ``kit show`` still shows the configured
active kit and prints a warning that normal commands will use the environment
path instead.

**********************
List Registered Kits
**********************

.. code-block:: shell

   nvflare config kit list

Example output:

.. code-block:: none

   active  id               status   identity         cert_role  path
   -------------------------------------------------------------------------------
   *       cancer_lead      ok       lead@nvidia.com  lead       /secure/startup_kits/cancer/lead@nvidia.com
           fraud_org_admin  missing  -                -          /secure/startup_kits/fraud/org_admin@nvidia.com
           old_lab_admin    invalid  -                -          /archive/old_lab/admin@nvidia.com

The active startup kit is marked with ``*``. Status values:

- ``ok``: the path exists and contains a valid startup kit.
- ``missing``: the registered path no longer exists.
- ``invalid``: the path exists but does not contain the required startup kit
  files.

``kit list`` does not contact the server and does not fail just because one
registered path is stale.

*********************
Remove a Registration
*********************

.. code-block:: shell

   nvflare config kit remove <id>

This removes the local registry entry only. It does not delete the startup kit
directory.

If the removed ID was active, ``startup_kits.active`` is cleared and you must
activate another kit before running server-connected commands.

*********************
Config File
*********************

``nvflare config kit`` stores startup kit registrations in
``~/.nvflare/config.conf``:

.. code-block:: none

   version = 2

   startup_kits {
     active = "cancer_lead"

     entries {
       cancer_lead = "/secure/startup_kits/cancer/lead@nvidia.com"
       fraud_org_admin = "/secure/startup_kits/fraud/org_admin@nvidia.com"
     }
   }

Startup kit paths are managed by the ``kit`` subcommands under
``nvflare config``. Other local CLI state in ``~/.nvflare/config.conf`` is an
implementation detail and is not part of the normal startup kit workflow.
