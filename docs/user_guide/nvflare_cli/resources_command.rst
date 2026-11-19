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
- ``--site``: include one participant's report and startup capacity. Requires ``--job``.
- ``--format json``: return structured output for automation.
- ``--schema``: print the command schema without connecting to a server.

Bare ``nvflare resources`` displays help; in JSON mode, a missing selector
returns an ``INVALID_ARGS`` error envelope. Use the standard startup-kit
selection described in :ref:`job_cli`; no feature-specific setup is required.
A job name comes from existing job metadata and does not replace its ID.

Human and automation modes
==========================

Use the same command for people and agents: default text for reading,
``--format json`` for the standard JSON envelope, and ``--schema`` for argument
and behavior discovery. No separate ``nvflare agent resources`` command is
needed. JSON retains the report's native numbers and precision; display
rounding applies only to text. This command does not stream JSON Lines.

Read the result
================

The capacity-time table shows how long CPU, memory and GPU capacity was available.
A job's utilization table shows average CPU use per participant and message
payload in each direction. The study table shows total CPU work per job in
core-hours.
GPU counts are integers. CPU capacity and average CPU use may
be fractional; GPU-hours may also be fractional. Text uses
at most two decimal places without trailing zeros; positive values below
0.01 display as ``<0.01`` rather than zero.
``AVG CPU USED`` divides ``resource_utilization.cpu_consumed.cpu_seconds`` by the shared participant
``duration_seconds``. Totals are calculated at query time. ``*`` marks incomplete values;
``—`` means no usable value. Structured explanations appear once in the owning
participant or totals object's ``issues`` list, tagged by metric and code.
See the output examples in
:ref:`resource_statistics` and the detailed explanations:

- :ref:`resource_statistics_capacity`
- :ref:`resource_statistics_cpu_work`
- :ref:`resource_statistics_message_traffic`
- :ref:`resource_statistics_reports`

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
