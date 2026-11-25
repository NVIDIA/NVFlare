.. _resource_statistics:

Resource Statistics
*******************

After a job finishes, resource statistics shows its participants' CPU, memory,
and GPU capacity, recorded CPU work and message
payload sent and received. Use it to inspect one job, look more closely
at a participant, or compare retained jobs in a study.

A participant is the server or a client site in the job. Reports are collected
automatically; no extra monitoring setup is needed.

Start with a command
====================

.. code-block:: shell

   # All participants in one finished job
   nvflare resources --job JOB_ID

   # Add one participant's startup capacity
   nvflare resources --job JOB_ID --site site-1

   # Return the structured report
   nvflare resources --job JOB_ID --format json

   # All accessible retained jobs in one study
   nvflare resources --study STUDY_NAME

A job query uses the default study unless ``--study STUDY_NAME`` is also
provided. Running ``nvflare resources`` without a selector displays help.
Use the normal administrator startup kit and permissions for your deployment;
see :ref:`CLI options and errors <resources_cli>`, or
:ref:`Python API access <flare_api_resource_utilization>`.

Reports become queryable after server finalization, not while training is
running. For live training statistics use ``nvflare job stats``; for system
monitoring see :ref:`monitoring`.

Read the output
===============

This example uses synthetic data to demonstrate the CLI format.

.. literalinclude:: resource_statistics/examples/job_resources.out
   :language: text

The capacity-time table combines each execution's visible resources and duration.
For example, a GPU count of two over ten minutes contributes one-third of
a GPU-hour. CPU and memory use core-hours and GiB-hours. These values describe
available capacity over time, not how busy those resources were.

``AVG CPU USED`` is different: it divides recorded CPU work by the
shared participant duration. In this example, site-1 used 180 CPU seconds over
600 seconds, averaging 0.3 CPUs despite having 32 logical CPU units visible.
The CLI uses ``cores`` as a short label for logical CPU units. CPU capacity can
be fractional when a CPU-time quota applies. The ``--site`` detail shows the
job-process startup capacity; in task mode this is
the coordinator, not its workers. GPU counts there are integers; capacity-time
can be fractional. Text shows at most two decimal places without
trailing zeros, and positive values below 0.01 as ``<0.01``. JSON keeps full
report precision.

The utilization table groups average CPU use and message payload;
capacity-time is shown separately. Message quantities use GiB, where 1 GiB is 1,073,741,824 bytes.
``PAYLOAD SENT`` and ``PAYLOAD RECEIVED`` are separate counters for the job's
messaging processes. They include repeated transfers and control payloads,
but not network headers. See :ref:`resource_statistics_message_traffic` for
the measurement boundary.

Reports retain startup capacity and accumulated execution capacity-time.
Queries add the recorded capacity-time, without multiplying it again by the
participant duration. Several participants or jobs
can see the same physical hardware, so their sums are not an inventory of
unique machines.

Incomplete information
======================

A number followed by ``*`` has incomplete information. For example, CPU
capacity may use a best-effort count when a container limit cannot be read.
An em dash (``—``) means no usable value. The ``Data gaps`` section explains
the limitation, and ``--format json`` provides the structured details.
Successful measurements have no status label in JSON. Each participant or totals
object has one shared ``issues`` list naming affected metrics; it is empty when
that scope is complete.

A participant may send a valid report with CPU data but no GPU observation.
CPU averages can still be shown. A failed or missing report does not change the training
job's success or failure.

Measurement details
===================

.. toctree::
   :maxdepth: 1

   resource_statistics/capacity
   resource_statistics/cpu_work
   resource_statistics/message_traffic
   resource_statistics/reports

Study results
=============

A study query includes jobs still retained by the job store and permitted by
the caller's existing per-job access policy. Running jobs and jobs without a
readable final resource summary appear as gaps and do not contribute to totals.
The view is not a permanent history of deleted jobs.

The following example shows two jobs with data, one without a usable report,
and one still running. Identical names do not merge jobs: the job ID identifies
each row.

.. literalinclude:: resource_statistics/examples/study_resources.out
   :language: text

This command selects jobs that started in October 2026 (UTC).

.. code-block:: shell

   nvflare resources --study cancer-research --since 2026-10-01 --until 2026-11-01

A job that starts in October and finishes in November contributes its full
recorded usage to this report. When you specify either bound, the report skips
jobs whose recorded start time is missing, invalid, or lacks a time zone.

Collection and availability
===========================

These archived reports cover jobs using NVFlare's server/client job processes.
Standalone simulator execution is outside this guide's scope. Collection uses
existing process access and requires no additional service, privilege, mount,
or operator configuration.

With ``execution_lifetime="task"``, workers are measured in their own execution
environments and folded into one participant report. The same path is used by
each supported task launcher. It does not enable otherwise unsupported launch
modes. Worker records and individual durations remain internal.

Reports are best effort. A process or server crash can leave information
missing; missing data is not assumed to be zero. See
:ref:`resource_statistics_reports` for collection, storage, and failure details.
