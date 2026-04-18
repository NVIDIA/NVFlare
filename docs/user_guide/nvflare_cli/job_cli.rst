.. _job_cli:

#########################
NVIDIA FLARE Job CLI
#########################

The ``nvflare job`` command family is used to submit, inspect, monitor, and
manage federated learning jobs from a configured admin startup kit.

Before using these commands, configure the startup kit location with
``nvflare config --poc.startup_kit <path>`` or prepare a local POC workspace
with ``nvflare poc prepare``.

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
     logs            retrieve job logs from server workspace
     log-config      change logging configuration for a running job
     stats           show running job statistics
     download        download job result
     clone           clone an existing job
     delete          delete a job
     list_templates  [DEPRECATED] use 'nvflare recipe list'
     create          [DEPRECATED] use 'python job.py --export --export-dir <job_folder>' + 'nvflare job submit'
     show_variables  [DEPRECATED] use 'nvflare recipe list' or the Job Recipe API

*****************
Common Workflow
*****************

1. Export or prepare a job folder.
2. Submit the job with ``nvflare job submit -j <job_folder>``.
3. Monitor it with ``nvflare job monitor <job_id>``.
4. Inspect metadata, stats, or logs as needed.
5. Download, clone, abort, or delete the job when appropriate.

****************
Submit a Job
****************

Use ``nvflare job submit`` to submit a pre-built NVFlare job folder:

.. code-block:: shell

   nvflare job submit -j /tmp/nvflare/hello-pt

Submit options:

- ``-j, --job_folder``: job folder path. Defaults to ``./current_job``.
- ``--target {poc,prod}``: choose the startup kit from ``~/.nvflare/config.conf``. See the startup kit resolution order below.
- ``--startup_kit``: explicit admin startup kit directory, or its ``startup/`` subdirectory. Mutually exclusive with ``--target``.
- ``--study``: submit into a named study when the server is configured for multi-study access. If omitted, the literal study name ``default`` is submitted.
- ``-debug, --debug``: keep the temporary copied job folder for inspection.
- ``--schema``: print the command schema as JSON and exit.

Submit returns immediately with a ``job_id``. It does not wait for terminal
job status.

To change job configuration values, edit the exported job files before
submission. Submit-time ``-f/--config_file`` overrides are not supported.

Startup kit resolution order:

1. ``--startup_kit``
2. ``NVFLARE_STARTUP_KIT_DIR``
3. resolved target via config (explicit ``--target`` value, otherwise default ``poc``):
   - ``poc`` -> ``poc.startup_kit``
   - ``prod`` -> ``prod.startup_kit``

Examples:

.. code-block:: shell

   nvflare job submit -j /tmp/nvflare/hello-pt --target poc
   nvflare job submit -j /tmp/nvflare/hello-pt --target prod
   nvflare job submit -j /tmp/nvflare/hello-pt --startup_kit /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com

``--startup_kit`` must point to the admin startup kit directory itself, not the
broader ``prod_00`` root. For example, use
``/tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com`` rather than
``/tmp/nvflare/poc/example_project/prod_00``.

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

Monitor options:

- ``job_id``: job ID to monitor.
- ``--timeout``: max seconds to wait. ``0`` means no timeout.
- ``--interval``: poll interval in seconds.
- ``--stats-target``: where to fetch stats from. Choices: ``server``, ``client``, ``all``.
- ``--metric``: extra metric key to surface from stats. Repeatable.
- ``--schema``: print the command schema as JSON and exit.

Exit behavior:

- exit code ``0``: job finished successfully
- exit code ``1``: job reached a terminal failure state such as ``FAILED`` or ``ABORTED``
- exit code ``3``: monitor timeout

This enables CI/CD-style chaining:

.. code-block:: shell

   JOB=$(nvflare --out-format json job submit -j ./my_job | jq -r .data.job_id)
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
- ``--study``: list jobs from a named study, or use ``all`` for all studies.
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

Retrieve job logs from the server workspace:

.. code-block:: shell

   nvflare job logs <job_id>
   nvflare job logs <job_id> --tail 200
   nvflare job logs <job_id> --grep ERROR

Current implementation note:

- ``job logs --site`` accepts only ``server``. Default: ``server``. Any other
  value is rejected during argument parsing.
- ``job logs`` also supports ``--tail``, ``--grep``, and ``--schema``.

Change logging configuration for a running job:

.. code-block:: shell

   nvflare job log-config <job_id> DEBUG
   nvflare job log-config <job_id> concise
   nvflare job log-config <job_id> --site all msg_only

``job log-config`` accepts:

- positional ``level``: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``
- log modes: ``concise``, ``msg_only``, ``full``, ``verbose``, ``reload``
- ``--site``: target site name or ``all``. Default: ``all``.
- ``--schema``: print the command schema as JSON and exit

Show running job statistics:

.. code-block:: shell

   nvflare job stats <job_id>
   nvflare job stats <job_id> --site all

``job stats`` supports ``--site`` to target a specific site or ``all``.
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

The top-level CLI supports ``--out-format json`` for machine-readable output:

.. code-block:: shell

   nvflare --out-format json job meta <job_id>

For normal command execution in JSON mode, stdout contains a single JSON
envelope. Human-readable progress and diagnostics are written to stderr.

Use ``--schema`` for machine-readable command discovery:

.. code-block:: shell

   nvflare --out-format json job submit --schema
   nvflare --out-format json job monitor --schema

Human-readable argument errors print command help first, followed by the
specific error and hint. JSON mode prints only the JSON error envelope.
