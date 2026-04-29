.. _job_cli:

#########################
NVIDIA FLARE Job CLI
#########################

The ``nvflare job`` command family is used to submit, inspect, monitor, and
manage federated learning jobs from an admin startup kit.

Before using server-connected job commands, either run ``nvflare poc prepare``
or activate a registered startup kit with :ref:`kit_command`:

.. code-block:: shell

   nvflare config add project_admin /path/to/admin@nvidia.com
   nvflare config use project_admin

***********************
Command Usage
***********************

.. code-block:: none

   nvflare job -h

   usage: nvflare job [-h]  ...

   job subcommands:
     submit          submit job
     monitor         wait for a job and stream progress to stderr
     list            list jobs on the server
     abort           abort a running job
     meta            get metadata for a job
     logs            retrieve job logs from the server-side log store
     log-config      change logging configuration for a running job
     stats           show running job statistics
     download        download job result
     clone           clone an existing job
     delete          delete a job
     list_templates  [DEPRECATED] use 'nvflare recipe list'
     create          [DEPRECATED] use 'python job.py --export --export-dir <job_folder>' + 'nvflare job submit -j <job_folder>'
     show_variables  [DEPRECATED] use 'nvflare recipe list' or the Job Recipe API

*****************
Common Workflow
*****************

1. Export or prepare a job folder.
2. Submit the job with ``nvflare job submit -j <job_folder>``.
3. Monitor it with ``nvflare job monitor <job_id>``.
4. Inspect metadata, stats, or logs as needed.
5. Download, clone, abort, or delete the job when appropriate.

*****************************
Startup Kit Selection
*****************************

Server-connected job commands use this startup kit resolution order:

1. ``--kit-id <id>``: use a registered startup-kit ID for this command only.
2. ``--startup-kit <path>``: use an explicit admin startup-kit directory for this command only.
3. ``NVFLARE_STARTUP_KIT_DIR`` when set.
4. ``startup_kits.active`` from ``~/.nvflare/config.conf``.
5. If no source resolves to a valid admin startup kit, the command fails before connecting.

``--kit-id`` and ``--startup-kit`` do not change the globally active startup kit.
They are useful for scripts, notebooks, and concurrent workflows that must not
mutate ``~/.nvflare/config.conf``.

****************
Submit a Job
****************

Use ``nvflare job submit`` to submit a pre-built NVFlare job folder:

.. code-block:: shell

   nvflare job submit -j /tmp/nvflare/hello-pt

Submit options:

- ``-j, --job_folder``: job folder path. Defaults to ``./current_job``.
- ``--study``: submit into a named study when the server is configured for multi-study access. If omitted, the literal study name ``default`` is submitted.
- ``-debug, --debug``: keep the temporary copied job folder for inspection.
- ``--schema``: print the command schema as JSON and exit.

Submit returns immediately with a ``job_id``. It does not wait for terminal
job status.

To change job configuration values, edit the exported job files before
submission. Submit-time ``-f/--config_file`` overrides are not supported.

Examples:

.. code-block:: shell

   nvflare config use project_admin
   nvflare job submit -j /tmp/nvflare/hello-pt
   nvflare job list --kit-id project_admin
   nvflare job submit -j /tmp/nvflare/hello-pt --startup-kit /path/to/admin@nvidia.com

Registered startup kit paths must point to the admin startup kit directory
itself, not the broader ``prod_00`` root.

Example JSON success response:

.. code-block:: json

   {"schema_version": "1", "status": "ok", "exit_code": 0, "data": {"job_id": "abc123"}}

If the server is configured for studies, you can target one explicitly:

.. code-block:: shell

   nvflare job submit -j /tmp/nvflare/my_job --study cancer_research

****************
Monitor a Job
****************

Use ``nvflare job monitor`` to wait for a running job and stream progress to
stderr:

.. code-block:: shell

   nvflare job monitor <job_id>
   nvflare job monitor <job_id> --study cancer_research

Monitor options:

- ``job_id``: job ID to monitor.
- ``--timeout``: max seconds to wait. Default: ``0`` (no timeout).
- ``--interval``: poll interval in seconds. Default: ``2``.
- ``--study``: monitor a job in a named study. Use the same study name used
  at submission time. If omitted, the literal study name ``default`` is used.
- ``--stats-target``: where to fetch stats from. Choices: ``server``, ``client``, ``all``. Default: ``server``.
- ``--metric``: extra metric key to surface from stats. Repeatable.
- ``--schema``: print the command schema as JSON and exit.

Exit behavior:

- exit code ``0``: job finished successfully
- exit code ``1``: job reached a terminal failure state: ``FAILED``, ``FINISHED_EXCEPTION``, ``ABORTED``, or ``ABANDONED``
- exit code ``2``: connection or authentication failure prevented monitoring
- exit code ``3``: monitor timeout

This enables CI/CD-style chaining:

.. code-block:: shell

   JOB=$(nvflare job submit -j ./my_job --format json | jq -r .data.job_id)
   nvflare job monitor $JOB && nvflare job download $JOB

*********************
List and Inspect Jobs
*********************

List jobs currently known to the server:

.. code-block:: shell

   nvflare job list

Common list filters:

- ``-n, --name``: filter by job name prefix.
- ``-i, --id``: filter by job ID prefix.
- ``-r, --reverse``: reverse sort order.
- ``-m, --max``: maximum number of results to return.
- ``--study``: list jobs from a named study. If omitted, the literal study name
  ``default`` is used. Values such as ``all`` are passed through to the server
  unchanged.
- ``--schema``: print the command schema as JSON and exit.

Retrieve metadata for a single job:

.. code-block:: shell

   nvflare job meta <job_id>

Use metadata to inspect job identity, lifecycle fields, and server-reported
status information after submission.

``nvflare job meta`` also supports ``--schema``.

******************************
Download, Clone, Abort, Delete
******************************

Download job results:

.. code-block:: shell

   nvflare job download <job_id> -o ./downloads

Clone an existing job:

.. code-block:: shell

   nvflare job clone <job_id>

``nvflare job clone`` clones the full server-side job for reuse. The current
CLI surface takes only the source ``job_id`` and ``--schema``.
It returns ``source_job_id`` and ``new_job_id``. Use the returned
``new_job_id`` to monitor or manage the cloned job.

Abort a running job:

.. code-block:: shell

   nvflare job abort <job_id>
   nvflare job abort <job_id> --force

Delete a job:

.. code-block:: shell

   nvflare job delete <job_id>
   nvflare job delete <job_id> --force

Notes:

- ``abort`` and ``delete`` support ``--force`` to skip the confirmation prompt.
- ``download`` supports ``-o, --output-dir`` to choose the destination directory. Default: current working directory (``./``).
- ``clone``, ``download``, ``abort``, and ``delete`` all support ``--schema``.

**************
Observability
**************

Retrieve job logs from the server-side log store:

.. code-block:: shell

   nvflare job logs <job_id>
   nvflare job logs <job_id> --site site-1
   nvflare job logs <job_id> --site all
   nvflare job logs <job_id> --site all --tail 200
   nvflare job logs <job_id> --site site-1 --since 2026-04-28T10:00:00
   nvflare job logs <job_id> --site all --max-bytes 200000

``job logs`` accepts:

- ``--site server``: return the server job log. This is the default.
- ``--site <client_name>``: return that client's job log after it has been
  streamed to and stored by the server.
- ``--site all``: return the server log and all client logs currently available
  in the server-side log store. If a known job site does not have stored log
  content, the JSON response includes it under ``unavailable``.
- ``--sites`` is accepted as an alias for ``--site`` but still selects one
  target value.
- ``--tail N``: return at most the last N log lines per site.
- ``--since timestamp``: return timestamped log lines at or after the timestamp
  when line timestamps are parseable. Continuation lines following an included
  timestamped line are included.
- ``--max-bytes N``: return at most N UTF-8 bytes per site.
- ``job logs`` also supports ``--schema``.

If no explicit bound is provided, ``job logs`` returns at most the last 500
lines per site. JSON output includes ``logs_truncated``, per-site availability
and line/byte counts under ``sites``, and the applied ``filters``.

In normal human output mode, ``job logs`` prints the log text directly. With
``--site all``, each site is separated by a short header. Use ``--format json``
when a structured ``logs`` dictionary is needed for automation.

``job logs`` does not provide a built-in ``grep`` option. Pipe or post-process
the returned content when text matching is needed.

Client logs are not fetched from client machines at command time. The command
asks the server for logs that were already streamed to the server during the
job. The current implementation reads client logs from the server job
workspace, where streamed client logs are stored as ``<client_name>/log.txt``;
after the job workspace is archived, the same files are read from the stored
job ``workspace`` artifact.

To enable client job log streaming in a portable job, add the job-level log
streamer and receiver components to the job definition:

.. code-block:: python

   from nvflare.app_common.logging.job_log_receiver import JobLogReceiver
   from nvflare.app_common.logging.job_log_streamer import JobLogStreamer

   # Tails each client's job log.txt and streams it to the server.
   recipe.job.to_clients(JobLogStreamer())

   # Receives streamed log chunks on the server and stores them with the job.
   recipe.job.to_server(JobLogReceiver())

System-level logging configuration in ``resources.json.default`` is separate
from this job-level opt-in. Some deployments may configure a server-side
``JobLogReceiver`` globally, but including both components in the job makes the
job self-contained across POC and production deployments.

The ``examples/hello-world/hello-log-streaming`` example shows this pattern.

Change logging configuration for a running job:

.. code-block:: shell

   nvflare job log-config <job_id> DEBUG
   nvflare job log-config <job_id> concise
   nvflare job log-config <job_id> msg_only

``job log-config`` accepts:

- positional ``level``: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``
- log modes: ``concise``, ``msg_only``, ``full``, ``verbose``, ``reload``
- ``--site``: target site name or ``all``. Default: ``all``; specifying
  ``--site all`` explicitly is equivalent to omitting it.
- ``--schema``: print the command schema as JSON and exit

Show running job statistics:

.. code-block:: shell

   nvflare job stats <job_id>

``job stats`` supports ``--site`` to target a specific site or ``all``.
The default is ``all``, so specifying ``--site all`` explicitly is equivalent
to omitting it.
It also supports ``--schema``.

***************************
Recipe-Based Job Creation
***************************

The recommended way to create a new job folder is through the Job Recipe API or
an example ``job.py`` script that supports ``--export``:

.. code-block:: shell

   python job.py --export --export-dir /tmp/nvflare/hello-pt
   nvflare job submit -j /tmp/nvflare/hello-pt

To discover built-in recipes, use:

.. code-block:: shell

   nvflare recipe list

Deprecated commands:

- ``nvflare job create``: retained for compatibility. Prefer ``python job.py --export`` followed by ``nvflare job submit``.
- ``nvflare job list_templates``: use ``nvflare recipe list``.
- ``nvflare job show_variables``: use the Job Recipe API.

Current deprecation notes:

- ``nvflare job create`` still exposes template- and config-oriented arguments for
  legacy workflows.
- ``nvflare job list_templates`` and ``nvflare job show_variables`` remain
  available for backward compatibility but are not the preferred interfaces for
  recipe discovery or job-variable inspection.

*********************
JSON Output and Help
*********************

Add ``--format json`` anywhere after the subcommand for machine-readable output:

.. code-block:: shell

   nvflare job meta <job_id> --format json

``--format json`` may be placed anywhere in the command after the subcommand
name. stdout contains a single JSON envelope; human-readable progress and
diagnostics go to stderr.

Use ``--schema`` for machine-readable command discovery. ``--schema`` always
returns JSON regardless of ``--format``, so the flag is not needed with it:

.. code-block:: shell

   nvflare job submit --schema
   nvflare job monitor --schema

Human-readable argument errors print command help first, followed by the
specific error and hint. JSON mode prints only the JSON error envelope.
