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

1. ``--startup-kit <path>`` — explicit path to the startup kit directory or its ``startup/``
   subdirectory.
2. ``NVFLARE_STARTUP_KIT_DIR`` environment variable.
3. ``~/.nvflare/config.conf`` — reads the ``poc`` target by default; reads ``prod`` when
   ``--startup-target prod`` is given. Written by ``nvflare config``.

A user who has run ``nvflare config`` once does not need to pass any flag — the config file
is consulted automatically. ``--startup-kit`` and ``--startup-target`` are mutually exclusive.

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
   nvflare study register cancer-research --sites hospital-1,hospital-2

Options:

- ``<name>`` (required positional): name of the study to create.
- ``--site-org <org>:<site>`` (project_admin): one or more ``org:site`` pairs; repeat the flag
  for multiple entries.
- ``--sites <site>[,<site>...]`` (org_admin): comma-separated list of sites in the caller's
  organisation.
- ``--startup-target``, ``--startup-kit``: startup kit resolution (see above).

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

- ``project_admin`` sees all studies.
- ``org_admin`` sees studies in which their organisation has enrolled sites.

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
