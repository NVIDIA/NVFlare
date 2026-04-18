.. _system_command:

#########################
System Command
#########################

The ``nvflare system`` command group manages a running FL system through the
admin API.

***********************
Command Usage
***********************

.. code-block:: none

   nvflare system -h

   usage: nvflare system [-h]  ...

   system subcommands:
     status     show server and client status
     resources  show server and client resource usage
     shutdown   shut down the FL server
     restart    restart the FL server
     version    show NVFlare version on each remote site
     log-config change logging level on server or client sites

*****************
Common Examples
*****************

Show overall system status:

.. code-block:: shell

   nvflare system status

Show server-only status:

.. code-block:: shell

   nvflare system status server

Show all reported resources:

.. code-block:: shell

   nvflare system resources

Show resources for clients only:

.. code-block:: shell

   nvflare system resources client

Restart the server:

.. code-block:: shell

   nvflare system restart server --force

Shut down the server:

.. code-block:: shell

   nvflare system shutdown server --force

Show deployed NVFlare versions:

.. code-block:: shell

   nvflare system version
   nvflare system version --site server

Change runtime logging:

.. code-block:: shell

   nvflare system log-config concise
   nvflare system log-config --site server DEBUG
   nvflare system log-config --site site-1 msg_only

****************
Status and Resources
****************

``nvflare system status`` reports server and client connectivity.

This command uses the word ``target`` in two different places:

- positional ``target``: ``server`` or ``client``. This tells NVFlare what to query.
- ``--startup-target {poc,prod}``: this tells the CLI which startup kit to use.

Status arguments:

- positional ``target``: optional. ``server`` or ``client``.
- positional ``client_names``: optional list of client names when targeting clients.
- ``--startup-target {poc,prod}``: choose the configured admin startup kit.
- ``--startup_kit``: explicit admin startup kit directory. If provided, it takes precedence over ``--startup-target``.
- ``--schema``: print the command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare system status
   nvflare system status server
   nvflare system status client site-1 site-2
   nvflare system status client site-1 --startup-target prod

In ``nvflare system status client site-1 --startup-target prod``:

- ``client site-1`` means query client ``site-1``.
- ``--startup-target prod`` means use the ``prod`` startup kit.

``nvflare system resources`` reports server and client resource usage.

This command uses the word ``target`` in two different places:

- positional ``target``: ``server`` or ``client``. This tells NVFlare what to query.
- ``--startup-target {poc,prod}``: this tells the CLI which startup kit to use.

Resource arguments:

- positional ``target``: optional. ``server`` or ``client``.
- positional ``client_names``: optional list of client names when targeting clients.
- ``--startup-target {poc,prod}``: choose the configured admin startup kit.
- ``--startup_kit``: explicit admin startup kit directory. If provided, it takes precedence over ``--startup-target``.
- ``--schema``: print the command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare system resources
   nvflare system resources client
   nvflare system resources client site-1 --startup-target prod

In ``nvflare system resources client site-1 --startup-target prod``:

- ``client site-1`` means query client ``site-1``.
- ``--startup-target prod`` means use the ``prod`` startup kit.

**********************
Shutdown and Restart
**********************

Use ``shutdown`` and ``restart`` to control the server process through
the admin channel.

Supported targets:

- ``server``

Control arguments:

- positional ``target``: required. Must be ``server``.
- ``--startup-target {poc,prod}``: choose the configured admin startup kit.
- ``--startup_kit``: explicit admin startup kit directory. If provided, it takes precedence over ``--startup-target``.
- ``--force``: skip the confirmation prompt.
- ``--schema``: print the command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare system shutdown server --force
   nvflare system restart server --force

In non-interactive contexts, ``--force`` is required.

****************
Version
****************

Use ``nvflare system version`` to query the NVFlare version reported by remote
sites.

Version arguments:

- ``--site``: ``server``, a client name, or ``all``. Default: ``all``.
- ``--startup-target {poc,prod}``: choose the configured admin startup kit.
- ``--startup_kit``: explicit admin startup kit directory. Mutually exclusive with ``--startup-target``.
- ``--schema``: print the command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare system version
   nvflare system version --site server
   nvflare system version --site site-1

The command reports site versions, whether they are compatible with the server
version, and which sites are mismatched.

****************
Runtime Logging
****************

Use ``nvflare system log-config`` to change logging on the server or client sites.

.. code-block:: shell

   nvflare system log-config DEBUG
   nvflare system log-config --site server verbose
   nvflare system log-config --site site-1 msg_only

Logging arguments:

- positional ``level``: runtime-required log level or built-in log mode; omitting it returns a CLI error
- ``--site``: ``server``, a client name, or ``all``. Default: ``all``.
- ``--startup-target {poc,prod}``: choose the configured admin startup kit.
- ``--startup_kit``: explicit admin startup kit directory. Mutually exclusive with ``--startup-target``.
- ``--schema``: print the command schema as JSON and exit.

Supported built-in values for positional ``level``:

- ``DEBUG``
- ``INFO``
- ``WARNING``
- ``ERROR``
- ``CRITICAL``
- ``concise``
- ``msg_only``
- ``full``
- ``verbose``
- ``reload``

``level`` is required at runtime. Omitting it does not fail argparse parsing, but the command will return an error.

*********************
JSON Output and Help
*********************

The top-level CLI supports ``--out-format json`` for machine-readable output:

.. code-block:: shell

   nvflare --out-format json system status
   nvflare --out-format json system version --site server

For normal command execution in JSON mode, stdout contains a single JSON
envelope. Human-readable progress and diagnostics are written to stderr.

Use ``--schema`` for machine-readable command discovery:

.. code-block:: shell

   nvflare --out-format json system status --schema
   nvflare --out-format json system shutdown server --schema

Human-readable argument errors print command help first, followed by the
specific error and hint. JSON mode prints only the JSON error envelope.
