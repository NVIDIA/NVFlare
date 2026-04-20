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

The POC command provides the subcommands ``prepare``, ``prepare-jobs-dir``,
``start``, ``stop``, and ``clean``.

.. code-block:: none

   nvflare poc -h

   usage: nvflare poc [-h] {prepare,prepare-jobs-dir,start,stop,clean} ...

*****************
Common Workflow
*****************

1. Run ``nvflare poc prepare`` to create the local workspace and startup kits.
2. Optionally run ``nvflare poc prepare-jobs-dir`` to link a jobs folder into
   the admin transfer area.
3. Run ``nvflare poc start`` to start the server and clients.
4. Start an admin console explicitly only when you need one.
5. Run ``nvflare poc stop`` to stop the system.
6. Run ``nvflare poc clean`` after the system is stopped.

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
  workspace and startup kit locations.
- On success, the command prints a JSON result containing the workspace path and
  discovered client list.

Example:

.. code-block:: shell

   nvflare poc prepare -n 2

**********************
Prepare Jobs Directory
**********************

Use ``nvflare poc prepare-jobs-dir`` to link a jobs directory into the admin
transfer area:

.. code-block:: none

   nvflare poc prepare-jobs-dir [-h] [-j [JOBS_DIR]] [-debug] [--force] [--schema]

Options:

- ``-j, --jobs_dir``: jobs directory to link.
- ``-debug, --debug``: debug mode.
- ``--force``: overwrite an existing linked jobs directory without prompting.
- ``--schema``: print command schema as JSON and exit.

Behavior notes:

- If the transfer directory already exists and stdin is non-interactive,
  ``--force`` is required.
- Jobs in the linked folder become accessible to the FL admin console.

Example:

.. code-block:: shell

   nvflare poc prepare-jobs-dir -j /path/to/jobs --force

**************
Start Services
**************

Use ``nvflare poc start`` to launch services in the prepared POC workspace:

.. code-block:: none

   nvflare poc start [-h] [-p [SERVICE]] [-ex [EXCLUDE]] [-gpu [GPU ...]]
                     [--study STUDY] [-debug] [--schema]

Options:

- ``-p, --service``: participant to start. By default, starts the server and
  clients; admin consoles are excluded unless explicitly requested.
- ``-ex, --exclude``: participant to exclude from startup.
- ``-gpu, --gpu``: GPU device IDs to use as ``CUDA_VISIBLE_DEVICES``.
- ``--study``: study for admin console launches only. Ignored for server and
  client services.
- ``-debug, --debug``: debug mode.
- ``--schema``: print command schema as JSON and exit.

Behavior changes:

- Admin console participants are **not started by default**.
- Running ``nvflare poc start`` with no explicit service starts the server and
  clients only.
- The command returns JSON with ``status``, ``server_url``, and ``clients``.

Examples:

.. code-block:: shell

   nvflare poc start
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

   nvflare poc stop [-h] [-p [SERVICE]] [-ex [EXCLUDE]] [-debug] [--schema]

Options:

- ``-p, --service``: participant to stop. By default, stops all running
  services, including admin consoles.
- ``-ex, --exclude``: participant to exclude from stop handling.
- ``-debug, --debug``: debug mode.
- ``--schema``: print command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare poc stop
   nvflare poc stop -p server
   nvflare poc stop -p site-1

Stopping the server path uses coordinated system shutdown logic. Stopping a
subset of services uses the local stop script flow.

****************
Clean Workspace
****************

Use ``nvflare poc clean`` to remove the POC workspace:

.. code-block:: none

   nvflare poc clean [-h] [-debug] [--force] [--schema]

Options:

- ``-debug, --debug``: debug mode.
- ``--force``: remove the workspace without an interactive confirmation prompt.
- ``--schema``: print command schema as JSON and exit.

Behavior notes:

- The workspace is removed only when it is a valid POC directory and the system
  is not still running.
- If the POC system is still running, stop it first with ``nvflare poc stop``.

*********************
Workspace Configuration
*********************

The default POC workspace is ``/tmp/nvflare/poc``.

The workspace can also be controlled by:

- ``NVFLARE_POC_WORKSPACE``
- ``~/.nvflare/config.conf`` via ``nvflare config --poc.workspace <poc_workspace>``

``nvflare poc prepare`` writes the POC workspace and startup kit location into
the local NVFlare config automatically.

The saved ``poc.startup_kit`` value is the POC admin startup kit directory,
for example ``.../prod_00/admin@nvidia.com``, not the broader ``prod_00`` root.

*********************
JSON Output and Help
*********************

The top-level CLI supports ``--out-format json`` for machine-readable output:

.. code-block:: shell

   nvflare --out-format json poc prepare -n 2
   nvflare --out-format json poc start

Use ``--schema`` for machine-readable command discovery:

.. code-block:: shell

   nvflare --out-format json poc prepare --schema
   nvflare --out-format json poc start --schema

Human-readable argument errors print help first, followed by the specific
error. JSON mode prints only the JSON error envelope.
