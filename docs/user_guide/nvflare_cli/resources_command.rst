.. _resources_cli:

**********************
Resources
**********************

``nvflare resources`` reads saved resource reports after a job has
finished and the server has finalized them. It shows observed capacity,
recorded CPU work and message payload sent
and received. For the collection methods and interpretation, start with
:ref:`resource_statistics`.

Commands
=========

.. code-block:: shell

   # One job in the default study
   nvflare resources --job JOB_ID

   # One job in a named study
   nvflare resources --job JOB_ID --study STUDY_NAME

   # Add one participant's startup capacity
   nvflare resources --job JOB_ID --site site-1

   # Structured job or participant output
   nvflare resources --job JOB_ID --site site-1 --format json

   # Accessible retained jobs in a study
   nvflare resources --study STUDY_NAME

Arguments
==========

- ``--job``: job ID to inspect, in the default study unless ``--study`` is given.
- ``--study``: select the study containing ``--job``, or query the study when
  ``--job`` is omitted.
- ``--since DATE_OR_TIME`` and ``--until DATE_OR_TIME`` select jobs starting at
  or after ``--since`` and before ``--until``. Use either or both with a study
  query (without ``--job``). Dates mean midnight UTC. ISO timestamps use UTC
  unless an offset is given.
- ``--site``: include one participant's report and startup capacity. Requires ``--job``.
- ``--format json``: return structured output for automation.
- ``--schema``: print the command schema without connecting to a server.

Bare ``nvflare resources`` displays help; in JSON mode, a missing selector
returns an ``INVALID_ARGS`` error envelope. Use the standard startup-kit
selection described in :ref:`job_cli`; no feature-specific setup is required.
A job name comes from existing job metadata and does not replace its ID.

Read the result
================

A job query shows capacity-time, average CPU use and message traffic by
participant. A study query shows each job's totals. An asterisk marks an
incomplete value and a dash means no usable value. See :ref:`resource_statistics`
for sample output, units and rounding.

Study queries check each job's normal resource-read permission before archive
access, rows, totals, or coverage counts. Nonterminal jobs and jobs without a
readable final summary do not contribute to numeric totals. The view covers
only jobs still retained in the job store.

The default wait for resource queries is 300 seconds. A client timeout leaves
the server's archive read running.

Structured output and errors
============================

For a job, the JSON envelope contains ``data.selection`` and ``data.summary``,
plus ``data.participant`` when ``--site`` is requested. A study envelope contains
``data.selection`` and ``data.summary`` with its job rows and computed totals. Reports and archive
readers support only the current schema ``1.0``. The format and example files
are described in :ref:`resource_statistics_reports`.

An unknown job returns ``JOB_NOT_FOUND``. For ``JOB_NOT_FINALIZED``, retry after
the job finishes. Using ``--site`` without ``--job`` returns ``INVALID_ARGS``.
If the selected site has no accepted report, the command returns
``RESOURCE_DATA_UNAVAILABLE``. Repeat the same job query without ``--site`` to see
which participants reported. The job summary still includes their measurements.

For live system resource information, use ``nvflare system resources``.
For running-job training statistics, use ``nvflare job stats``. The same saved
resource reports are available through :ref:`flare_api_resource_utilization`.
