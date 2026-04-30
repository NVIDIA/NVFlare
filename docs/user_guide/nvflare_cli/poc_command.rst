.. _poc_command:

*****************************************
Proof Of Concept (POC) Command
*****************************************

The ``nvflare poc`` command manages a local proof-of-concept deployment on a
single machine. Separate processes represent the server, clients, and admin
startup kits, which makes POC mode a convenient way to validate job workflows
before a distributed deployment.

***********************
Command Usage
***********************

The POC command provides the subcommands ``config``, ``prepare``, ``add``,
``start``, ``stop``, and ``clean``.

.. code-block:: none

   nvflare poc -h

   usage: nvflare poc [-h] {config,prepare,add,start,stop,clean} ...

*****************
Common Workflow
*****************

1. Optionally run ``nvflare poc config --pw <poc_workspace>`` to choose the
   local workspace path.
2. Run ``nvflare poc prepare`` to create the local workspace and startup kits.
3. Optionally run ``nvflare poc add user`` or ``nvflare poc add site`` to add a
   local participant startup kit.
4. Run ``nvflare poc start`` to start the server and clients.
5. Submit jobs directly with ``nvflare job submit -j <path/to/job>``.
6. Start an admin console explicitly only when you need one.
7. Run ``nvflare poc stop`` to stop the system.
8. Run ``nvflare poc clean`` after the system is stopped.

*******************
Configure Workspace
*******************

Use ``nvflare poc config`` to show or set the local POC workspace path:

.. code-block:: none

   nvflare poc config [-h] [-pw [POC_WORKSPACE_DIR]] [--schema]

Options:

- ``-pw, --pw, --poc_workspace_dir, --poc-workspace-dir``: POC workspace location.
- ``--schema``: print the command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare poc config
   nvflare poc config --pw /tmp/nvflare/poc

*******************
Prepare Workspace
*******************

Use ``nvflare poc prepare`` to provision a local project:

.. code-block:: none

   nvflare poc prepare [-h] [-n [NUMBER_OF_CLIENTS]] [-c [CLIENTS ...]]
                       [-he] [-i [PROJECT_INPUT]] [-d [DOCKER_IMAGE]]
                       [-debug] [--force] [--schema]

Options:

- ``-n, --number_of_clients``: number of sites or clients. Default: ``2``.
- ``-c, --clients``: space-separated client names. If specified,
  ``number_of_clients`` is ignored.
- ``-he, --he``: enable homomorphic encryption in the generated local project.
- ``-i, --project_input``: path to a ``project.yaml`` file. If specified,
  client-count, client-name, and docker-image options are ignored.
- ``-d, --docker_image``: generate docker-oriented startup scripts using the
  specified image. If given without a value, the default image is used.
- ``-debug, --debug``: debug mode.
- ``--force``: overwrite the existing workspace without prompting.
- ``--schema``: print command schema as JSON and exit.

Behavior notes:

- If the workspace already exists and stdin is non-interactive, ``--force`` is
  required.
- ``nvflare poc prepare`` updates ``~/.nvflare/config.conf`` with the POC
  workspace, registers generated admin/user startup kits, and activates the
  default Project Admin kit. Site startup kits stay in the POC workspace and are
  not registered as CLI identities.
- On success, the command prints a JSON result containing the workspace path and
  discovered client list.

Example:

.. code-block:: shell

   nvflare poc prepare -n 2

***************
Add Participant
***************

Use ``nvflare poc add`` to extend the prepared local POC workspace with another
user or site:

.. code-block:: none

   nvflare poc add user [-h] [--org ORG] [--force] [--schema]
                        {org_admin,lead,member} email

   nvflare poc add site [-h] [--org ORG] [--force] [--schema] name

Behavior notes:

- ``poc add user`` and ``poc add site`` require the active startup kit to have
  the ``project_admin`` certificate role. Use ``nvflare config use <id>`` to switch
  back to the POC Project Admin kit before adding users or sites.
- ``poc add user`` adds a secondary admin participant to the persisted POC
  ``project.yml``, dynamically provisions only that new user with the existing
  POC CA, and registers the generated user startup kit in the shared startup
  kit registry. It cannot add another ``project_admin``; the POC Project Admin
  is created by ``poc prepare``.
- ``poc add site`` adds a client participant to the persisted POC
  ``project.yml`` and dynamically provisions only that new site with the
  existing POC CA. The generated site kit is placed in the current POC output
  directory, normally ``prod_00``, and is not registered in
  ``~/.nvflare/config.conf`` because only admin/user kits are CLI identities.
- POC add uses the existing provision state/rootCA and does not regenerate
  existing participant startup kits.
- Use ``--force`` only to replace an existing participant entry in the local
  POC project metadata.

Examples:

.. code-block:: shell

   nvflare poc add user lead bob@nvidia.com --org nvidia
   nvflare config use bob@nvidia.com

   nvflare poc add site site-3 --org nvidia
   nvflare config list
   nvflare poc start -p site-3

**************
Start Services
**************

Use ``nvflare poc start`` to launch services in the prepared POC workspace:

.. code-block:: none

   nvflare poc start [-h] [-p [SERVICE]] [-ex [EXCLUDE]] [-gpu [GPU ...]]
                     [--study STUDY] [--no-wait] [--timeout SECONDS]
                     [-debug] [--schema]

Options:

- ``-p, --service``: participant to start. By default, starts the server and
  clients; admin consoles are excluded unless explicitly requested.
- ``-ex, --exclude``: participant to exclude from startup.
- ``-gpu, --gpu``: GPU device IDs to use as ``CUDA_VISIBLE_DEVICES``.
- ``--study``: study for admin console launches only. Ignored for server and
  client services.
- ``--no-wait``: return after starting processes without waiting for the admin
  server and selected clients to become ready.
- ``--timeout``: seconds to wait for the admin server and selected clients to
  become ready. Defaults to the built-in POC readiness timeout.
- ``-debug, --debug``: debug mode.
- ``--schema``: print command schema as JSON and exit.

Behavior changes:

- Admin console participants are **not started by default**.
- Running ``nvflare poc start`` with no explicit service starts the server and
  clients only.
- By default, the command waits until the admin server accepts connections and
  selected clients are registered before returning ``status: running``.
- Use ``--timeout`` to control this readiness wait.
- With ``--no-wait``, the command returns immediately with ``status: starting``.
- The command returns JSON with ``status``, ``server_url``, ``server_address``,
  ``admin_address``, ``clients``, ``ready_timeout``, ``port_conflict``,
  ``port_preflight``, ``warnings``, and, when readiness was checked or
  explicitly skipped, ``ready``.
- Use ``data.server_address`` and ``data.admin_address`` as the machine-readable
  endpoint addresses for subsequent automation. ``data.server_url`` is kept for
  compatibility with existing clients.
- ``data.port_conflict`` is a best-effort pre-start warning based on local port
  checks. When true, inspect ``data.port_preflight.conflicts`` and
  ``data.warnings`` before submitting jobs to avoid connecting to a different
  running POC system.

Examples:

.. code-block:: shell

   nvflare poc start
   nvflare poc start --timeout 60
   nvflare poc start -p server
   nvflare poc start -p admin@nvidia.com
   nvflare poc start -p admin@nvidia.com --study cancer_research
   nvflare poc start -ex admin@nvidia.com

To start an admin console, specify it explicitly with ``-p``.

Study notes:

- Use ``--study`` only when starting an admin console.
- Named studies require the POC workspace to be prepared from a custom
  ``project.yml`` with ``api_version: 4`` and ``studies:``. If the workspace
  was prepared from the default generated project, only the ``default`` study
  is valid.

*************
Stop Services
*************

Use ``nvflare poc stop`` to stop running POC services:

.. code-block:: none

   nvflare poc stop [-h] [-p [SERVICE]] [-ex [EXCLUDE]] [--no-wait]
                    [-debug] [--schema]

Options:

- ``-p, --service``: participant to stop. By default, stops all running
  services, including admin consoles.
- ``-ex, --exclude``: participant to exclude from stop handling.
- ``--no-wait``: return after requesting shutdown without waiting for completion.
- ``-debug, --debug``: debug mode.
- ``--schema``: print command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare poc stop
   nvflare poc stop -p server
   nvflare poc stop -p site-1
   nvflare poc stop --no-wait

Stopping the server path uses coordinated system shutdown logic. Stopping a
subset of services uses the local stop script flow. By default, the server path
waits for shutdown completion before returning ``status: stopped``. With
``--no-wait``, it returns immediately with ``status: shutdown_initiated``.

****************
Clean Workspace
****************

Use ``nvflare poc clean`` to remove the POC workspace:

.. code-block:: none

   nvflare poc clean [-h] [-debug] [--force] [--schema]

Options:

- ``-debug, --debug``: debug mode.
- ``--force``: stop a running local POC system before removing the workspace.
- ``--schema``: print command schema as JSON and exit.

Behavior notes:

- The workspace is removed only when it is a valid POC directory.
- If the POC system is still running, ``nvflare poc clean`` fails with a hint to
  stop it first. Use ``nvflare poc clean --force`` to stop the local POC system
  and then remove the workspace in one command.

*********************
Workspace Configuration
*********************

The default POC workspace is ``/tmp/nvflare/poc``.

The workspace can also be controlled by:

- ``NVFLARE_POC_WORKSPACE``
- ``~/.nvflare/config.conf`` via ``nvflare poc config --pw <poc_workspace>``

Use ``nvflare poc config`` to show or set the local POC workspace:

.. code-block:: shell

   nvflare poc config
   nvflare poc config --pw /tmp/nvflare/poc

The older root command ``nvflare config -pw <poc_workspace>`` remains accepted
for compatibility, but it is deprecated and prints a warning that points to
``nvflare poc config --pw``.

``nvflare poc prepare`` writes the POC workspace into the local NVFlare config
and registers generated admin/user startup kits in the shared startup kit
registry automatically. If a generated POC identity collides with an existing
startup-kit registration outside the POC workspace, prepare preserves the
existing registration when its path still exists. If the existing registration
points to a path that no longer exists, prepare treats it as stale local POC
state and replaces it. Site startup kits remain in the POC workspace for local
service management.

The default Project Admin startup kit becomes active, so server-connected
commands such as ``nvflare job list`` and ``nvflare system status`` work
without extra startup-kit flags.

In JSON mode, ``nvflare poc prepare`` reports the active-kit transition:
``data.startup_kit.prior_active`` is the startup-kit ID that was active before
prepare, ``data.startup_kit.active`` is the ID active after prepare, and
``data.startup_kit.changed`` indicates whether prepare changed the default
identity. Agents can use this information to restore the user's previous
identity after the POC workflow.

``nvflare poc prepare`` also reports a best-effort local server port preflight
under ``data.port_preflight``. The command checks the generated POC server ports
on the loopback address when the project configuration can be read and lists
unavailable ports in ``data.port_preflight.conflicts``. These conflicts are
warnings for the later ``nvflare poc start`` step; they do not make
``poc prepare`` fail. Since the preflight does not bind wildcard interfaces,
``data.port_preflight.note`` describes the check as best effort.

In JSON mode, ``nvflare poc start`` reports the bound POC endpoints under
``data.server_address`` and ``data.admin_address``. The command waits for
readiness by default unless ``--no-wait`` is used. It also repeats a best-effort
local server port preflight before startup and reports unavailable configured
ports under ``data.port_preflight.conflicts`` with ``data.port_conflict`` set to
``true``. The preflight is loopback-scoped and is intended as an early warning;
startup can still fail if another local bind address conflicts.

Use :ref:`kit_command` to inspect generated POC startup kit registrations or
switch between POC-generated user startup kits.

*********************
JSON Output and Help
*********************

Add ``--format json`` after the subcommand for machine-readable output:

.. code-block:: shell

   nvflare poc prepare -n 2 --format json
   nvflare poc start --format json

Use ``--schema`` for machine-readable command discovery. ``--schema`` always
returns JSON so ``--format json`` is not needed with it:

.. code-block:: shell

   nvflare poc prepare --schema
   nvflare poc start --schema

Human-readable argument errors print help first, followed by the specific
error. JSON mode prints only the JSON error envelope.
