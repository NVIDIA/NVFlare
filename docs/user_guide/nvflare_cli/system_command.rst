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
     status         show server and client status
     resources      show server and client resource usage
     shutdown       shut down server, clients, or all
     restart        restart server, clients, or all
     disable-client disable a client from reconnecting
     enable-client  enable a disabled client to reconnect
     version        show NVFlare version on each remote site
     log-config     change logging level on server or client sites

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

Shut down specific clients:

.. code-block:: shell

   nvflare system shutdown client site-1 site-2 --force

Disable a client from reconnecting:

.. code-block:: shell

   nvflare system disable-client site-1 --force

Enable a disabled client:

.. code-block:: shell

   nvflare system enable-client site-1 --force

Show deployed NVFlare versions:

.. code-block:: shell

   nvflare system version
   nvflare system version --site server

Change runtime logging:

.. code-block:: shell

   nvflare system log-config concise
   nvflare system log-config --site server DEBUG
   nvflare system log-config --site site-1 msg_only

.. note::

   All server-connected ``nvflare system`` commands resolve the startup kit in
   this order: ``--kit-id <id>``, ``--startup-kit <path>``,
   ``NVFLARE_STARTUP_KIT_DIR``, then ``startup_kits.active`` from
   ``~/.nvflare/config.conf``. ``--kit-id`` and ``--startup-kit`` are optional
   per-command overrides. When provided, they take precedence over the active
   startup kit for the current invocation only and do not change
   ``startup_kits.active``. Use ``nvflare config add`` and
   ``nvflare config use`` to manage the active startup kit. See
   :ref:`kit_command`.

****************
Status and Resources
****************

``nvflare system status`` reports server and client connectivity.

The positional ``target`` argument means ``server`` or ``client``. It tells
NVFlare what to query.

Status arguments:

- positional ``target``: optional. ``server`` or ``client``.
- positional ``client_names``: optional list of client names when targeting clients.
- ``--schema``: print the command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare system status
   nvflare system status server
   nvflare system status client site-1 site-2

In ``nvflare system status client site-1``, ``client site-1`` means query
client ``site-1``.

``nvflare system resources`` reports server and client resource usage.

The positional ``target`` argument means ``server`` or ``client``. It tells
NVFlare what to query.

Resource arguments:

- positional ``target``: optional. ``server`` or ``client``.
- positional ``client_names``: optional list of client names when targeting clients.
- ``--schema``: print the command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare system resources
   nvflare system resources client
   nvflare system resources client site-1

In ``nvflare system resources client site-1``, ``client site-1`` means query
client ``site-1``.

**********************
Shutdown and Restart
**********************

Use ``shutdown`` and ``restart`` to control server or client processes through
the admin channel.

Supported targets:

- ``server`` — shut down or restart the FL server (closes the admin session).
- ``client`` — shut down or restart one or more clients.
- ``all`` — shut down or restart the server and all clients (closes the admin session).

Control arguments:

- positional ``target``: required. One of ``server``, ``client``, or ``all``.
- positional ``client_names``: optional. One or more client names. Only meaningful when ``target`` is ``client``.
- ``--force``: skip the confirmation prompt.
- ``--no-wait``: return after requesting shutdown or restart without waiting
  for completion.
- ``--timeout SECONDS``: maximum positive seconds to wait for shutdown or
  restart completion. Default: ``30``. Use ``--no-wait`` instead of
  ``--timeout 0`` for fire-and-forget operation.
- ``--schema``: print the command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare system shutdown server --force
   nvflare system shutdown client site-1 site-2 --force
   nvflare system shutdown all --force
   nvflare system shutdown all --force --no-wait
   nvflare system shutdown all --force --timeout 120

   nvflare system restart server --force
   nvflare system restart client site-1 --force
   nvflare system restart all --force
   nvflare system restart server --force --no-wait
   nvflare system restart all --force --timeout 120

In non-interactive contexts, ``--force`` is required.

By default, shutdown waits for the target to stop before returning, and restart
waits for the target to become reachable again before returning. For
``restart all``, this includes waiting for the server to restart and for
previously connected clients to reconnect. With ``--no-wait``, the command
returns immediately with an initiated status. When ``target`` is ``server`` or
``all``, the admin session closes automatically after the shutdown or restart
request is sent.
If the wait exceeds ``--timeout``, the command returns ``TIMEOUT`` with exit
code ``3`` instead of reporting a connection failure.

****************
Client Access Control
****************

Use ``disable-client`` to persistently block a client identity from joining the
running federation. The server removes any active registry entry for the client
and rejects later registration or heartbeat attempts until the client is enabled.
This does not revoke the client's certificate or delete its startup kit. JSON
output includes ``already_disabled`` so callers can distinguish a state change
from an idempotent no-op.

Use ``enable-client`` to remove the disabled flag. The client can rejoin on its
next registration or heartbeat.

The disabled-client policy is stored on the server in
``<server_workspace>/disabled_clients.json`` and is loaded at server startup.
Updates and persistence writes are serialized by the server client-manager lock
and written with a temporary file followed by atomic replacement, so the policy
survives server restart without partially written files. If the file exists but
cannot be loaded, the server fails closed during startup instead of admitting
previously disabled clients.

Client access arguments:

- positional ``client_name``: required. The name of the client to disable or enable.
- ``--force``: skip the confirmation prompt.
- ``--schema``: print the command schema as JSON and exit.

Examples:

.. code-block:: shell

   nvflare system disable-client site-1 --force
   nvflare system enable-client site-1 --force

****************
Version
****************

Use ``nvflare system version`` to query the NVFlare version reported by remote
sites.

Version arguments:

- ``--site``: ``server``, a client name, or ``all``. Default: ``all``.
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

Add ``--format json`` after the subcommand for machine-readable output:

.. code-block:: shell

   nvflare system status --format json
   nvflare system version --site server --format json

stdout contains a single JSON envelope; human-readable progress and
diagnostics go to stderr.

Use ``--schema`` for machine-readable command discovery. ``--schema`` always
returns JSON so ``--format json`` is not needed with it:

.. code-block:: shell

   nvflare system status --schema
   nvflare system shutdown server --schema

Human-readable argument errors print command help first, followed by the
specific error and hint. JSON mode prints only the JSON error envelope.
