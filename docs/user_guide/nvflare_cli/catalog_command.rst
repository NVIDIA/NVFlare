.. _catalog_command:

Catalog commands
================

Use ``nvflare catalog read`` for an ad hoc catalog update or to test site
catalog configuration. Sites publish metadata according to their own study
policies. The server must have a ``CatalogManager`` configured in its normal
resources file. The standard server command registry resolves that service
regardless of its component ID; missing configuration returns
``SYSTEM_NOT_READY``. Reads run on
demand; use an external scheduler such as cron to refresh regularly.

For a named Study, catalog search and status require Study membership or a
certificate entitlement, including for project and organization admins. The
site's authorization policy also applies. The default Study follows the site's
authorization policy without a separate membership requirement.

.. code-block:: bash

   nvflare catalog read --study oncology
   nvflare catalog read --study oncology --sites site-a site-a:clinical
   nvflare catalog read --study oncology --no-wait
   nvflare catalog show --study oncology
   nvflare catalog show --study oncology --overview
   nvflare catalog show --study oncology --site site-a --dataset clinical

``--sites`` selects the sites to read; omitting it selects every site in the
study. Name datasets as ``SITE:NAME`` arguments after the options, for example
``site-a:clinical``, to read only those datasets. A selected site with no
``SITE:NAME`` entry reads everything its policy shares. ``--timeout``
sets the task timeout per site in seconds (default 300).

``read`` waits for the job to finish by default and reports its actual terminal
status without downloading job results. Use ``--no-wait`` to return the job ID
immediately after submission. The default exits 0 on successful job completion
and 1 for failed, aborted or unschedulable jobs. A completed job can still have
incomplete site reads; use ``show`` to inspect site health. Warnings describe
failed, degraded and stale sites, including when no datasets are visible.

Human and agent output
----------------------

Both commands use human-readable text by default. For automation, add
``--format json`` to receive exactly one versioned JSON envelope on stdout.
Progress and warnings go to stderr. ``read`` returns ``data.study`` and
``data.job_id`` and, by default, the terminal ``data.status``. With ``--no-wait``,
it returns only the submission information; exit 0 means the job was submitted.
``show`` returns dataset entries and structured site health under ``data``.
By default, every selected dataset includes full published metadata: tables,
columns, keys, relationships, provenance and version information. With both
``--site`` and ``--dataset``, it returns one dataset under ``data.dataset``;
otherwise the complete collection is under ``data.datasets``.

Use ``show --overview`` for the compact SITE/DATASET/NAME/AVAILABLE/VERSION/LAST
CHECKED table. In JSON mode, ``--overview`` omits schema metadata. Both modes
support ``--study``, ``--site`` and ``--dataset`` filters and retain site health
and warnings. ``--overview`` applies to display/export only, not authorization.

One command exports the entire visible Study catalog for offline visualization;
users do not need to enumerate sites or combine per-dataset files:

.. code-block:: bash

   nvflare catalog show --format json > catalog_results.json

.. code-block:: bash

   nvflare catalog read --study oncology --format json
   nvflare catalog read --study oncology --no-wait --format json
   nvflare catalog show --study oncology --format json
   nvflare catalog read --schema
   nvflare catalog show --schema

``--schema`` describes arguments and command behavior without a startup kit or
server connection. ``read`` submits a new job on every invocation and has no
retry-token deduplication. ``show`` is read-only and idempotent.

Exit codes follow the shared CLI convention: 0 success, 1 server/job error,
2 connection/authentication failure, 3 timeout, 4 invalid arguments, and
5 unexpected internal error. Use ``nvflare job meta JOB_ID --study STUDY``
and ``nvflare job logs JOB_ID --study STUDY`` to inspect a failed job.

Catalog commands use the standard startup-kit selection options and the
active kit configured with ``nvflare config``.
