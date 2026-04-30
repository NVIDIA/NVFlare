.. _study_command:

############################
NVIDIA FLARE Study CLI
############################

The ``nvflare study`` command family manages multi-study lifecycle operations on a running
NVFlare server: registering and removing studies, enrolling and removing sites, and managing
study user membership.

These commands are only meaningful when the server is provisioned with ``api_version: 4``
and has multi-study support enabled. For provisioning setup, see :ref:`multi_study_guide`.

.. code-block:: none

   nvflare study -h

   usage: nvflare study [-h] {register,show,list,remove,add-site,remove-site,add-user,remove-user} ...

   study subcommands:
     register       register a new study with initial site enrollment
     show           show the current definition of a study
     list           list all studies visible to the caller
     remove         remove a study and all its configuration
     add-site       enroll additional sites in a study
     remove-site    remove sites from a study
     add-user       add a user to a study's admin list
     remove-user    remove a user from a study's admin list

*****************************
Startup Kit Resolution
*****************************

All ``nvflare study`` commands connect to the server through an admin startup kit. Resolution
is identical to all other server-connected ``nvflare`` commands (``job``, ``system``, etc.):

1. Optional ``--kit-id <id>``: override the active startup kit for this command
   only by using a registered startup-kit ID.
2. Optional ``--startup-kit <path>``: override the active startup kit for this
   command only by using an explicit admin startup-kit directory.
3. ``NVFLARE_STARTUP_KIT_DIR`` environment variable.
4. ``startup_kits.active`` from ``~/.nvflare/config.conf``.
5. If no source resolves to a valid admin startup kit, the command fails before connecting.

The command-line selectors are not required. When provided, they take precedence
over the active startup kit for the current command only and do not change
``startup_kits.active`` in ``~/.nvflare/config.conf``.

A user can register and activate a startup kit once with :ref:`kit_command`:

.. code-block:: shell

   nvflare config add project_admin /path/to/admin@nvidia.com
   nvflare config use project_admin
   nvflare study list --kit-id project_admin

If no source resolves, the command exits with error code 4 and ``"error_code": "STARTUP_KIT_MISSING"``.

*****************************
Role-Based Input Requirements
*****************************

Study site enrollment follows a two-layer role check: first at the CLI (from the caller's
certificate in the startup kit), then authoritatively at the server (from the authenticated
connection properties).

- **project_admin** — manages site enrollment by specifying ``--site-org <org>:<site>`` pairs.
  Using ``--sites`` is rejected.
- **org_admin** — manages only sites in their own organisation by specifying ``--sites``.
  Using ``--site-org`` is rejected.
- Specifying both ``--sites`` and ``--site-org`` in the same command is always rejected.

*********************
Register a Study
*********************

Register a new study and enroll its initial set of sites.

.. code-block:: shell

   # project_admin: register with per-org site groupings
   nvflare study register cancer-research \
       --site-org org_a:hospital-1 \
       --site-org org_a:hospital-2 \
       --site-org org_b:clinic-1

   # org_admin: register and enroll own org's sites
   nvflare study register cancer-research --sites hospital-1 hospital-2

Options:

- ``<name>`` (required positional): name of the study to create.
- ``--site-org <org>:<site>`` (project_admin): one or more ``org:site`` pairs; repeat the flag
  for multiple entries.
- ``--sites <site> [<site> ...]`` (org_admin): one or more sites in the caller's
  organisation. Comma-separated input such as ``--sites hospital-1,hospital-2`` is also
  accepted.

*********************
Show a Study
*********************

Display the current definition of a study, including enrolled sites and admin users.

.. code-block:: shell

   nvflare study show cancer-research

Returns the site-org mapping and the list of admins for the study.

*********************
List Studies
*********************

List all studies the caller has access to.

.. code-block:: shell

   nvflare study list
   nvflare study list --format json

- ``project_admin`` sees all studies.
- ``org_admin`` sees studies in which their organisation has enrolled sites.
- ``lead`` and ``member`` users see studies where they are explicitly mapped.

In JSON mode, the command includes the startup kit selected by the CLI, the
identity authenticated by the server, and per-study submit preflight fields:

.. code-block:: json

   {
     "startup_kit": {
       "source": "active",
       "id": "lead@nvidia.com",
       "path": "/path/to/lead@nvidia.com"
     },
     "identity": {
       "name": "lead@nvidia.com",
       "org": "nvidia",
       "role": "lead"
     },
     "studies": ["cancer-research"],
     "study_details": [
       {
         "name": "cancer-research",
         "role": "lead",
         "capabilities": {"submit_job": true},
         "can_submit_job": true
       }
     ]
   }

``can_submit_job`` reflects the current server-side study visibility/mapping
preflight. It does not expose future custom authorization-policy details; a
later submit may still fail for other server-side validation or policy reasons.

*********************
Remove a Study
*********************

Remove a study and all its configuration. This operation is rejected if any job is currently
running under the study.

.. code-block:: shell

   nvflare study remove cancer-research

***********************
Add Sites to a Study
***********************

Enroll additional sites in an existing study.

.. code-block:: shell

   # project_admin
   nvflare study add-site cancer-research \
       --site-org org_b:clinic-2

   # org_admin
   nvflare study add-site cancer-research --sites clinic-2

Options match ``register`` for ``--site-org`` / ``--sites``.

**************************
Remove Sites from a Study
**************************

Remove sites from a study. The study itself is not deleted.

.. code-block:: shell

   # project_admin
   nvflare study remove-site cancer-research \
       --site-org org_b:clinic-2

   # org_admin
   nvflare study remove-site cancer-research --sites clinic-2

****************************
Add a User to a Study
****************************

Add an existing admin user to a study's admin list.

.. code-block:: shell

   nvflare study add-user cancer-research trainer@org_a.com

- ``<study>`` (required positional): the study to update.
- ``<user>`` (required positional): the admin user to add.

*******************************
Remove a User from a Study
*******************************

Remove a user from a study's admin list. The user is not deleted from the deployment.

.. code-block:: shell

   nvflare study remove-user cancer-research trainer@org_a.com

*****************************
Output Format
*****************************

All ``nvflare study`` commands honour the global ``--format {txt,json}`` flag.
With ``--format json`` (default for automation), every response is a JSON envelope:

.. code-block:: json

   {"status": "ok", "data": { ... }}

Errors are returned as:

.. code-block:: json

   {"status": "error", "error_code": "STUDY_NOT_FOUND", "message": "...", "hint": "...", "exit_code": 1}

Common error codes:

+----------------------------+-----------------------------------------------------------+
| Error code                 | Meaning                                                   |
+============================+===========================================================+
| ``STARTUP_KIT_MISSING``    | No startup kit could be resolved (exit 4)                 |
+----------------------------+-----------------------------------------------------------+
| ``INVALID_ARGS``           | Argument shape violates role contract (exit 4)            |
+----------------------------+-----------------------------------------------------------+
| ``STUDY_NOT_FOUND``        | Named study does not exist                                |
+----------------------------+-----------------------------------------------------------+
| ``STUDY_ALREADY_EXISTS``   | Study name is already registered                          |
+----------------------------+-----------------------------------------------------------+
| ``INVALID_SITE``           | Site is not enrolled or does not belong to the caller org |
+----------------------------+-----------------------------------------------------------+
| ``INVALID_STUDY_NAME``     | Study name fails naming rules                             |
+----------------------------+-----------------------------------------------------------+
| ``STUDY_HAS_RUNNING_JOBS`` | Cannot remove a study while jobs are running              |
+----------------------------+-----------------------------------------------------------+
| ``LOCK_TIMEOUT``           | Registry is busy; another mutation is in progress         |
+----------------------------+-----------------------------------------------------------+

*****************************
Schema Output
*****************************

Any subcommand supports ``--schema`` to print its argument schema as JSON:

.. code-block:: shell

   nvflare study register --schema
