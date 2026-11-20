.. _resource_statistics_cpu_work:

CPU Work and Average CPU Used
*****************************

CPU work is processing time recorded for the job process and completed child
processes. It is different from :ref:`CPU capacity <resource_statistics_capacity>`.

Read the values
===============

One second of work on each of two CPUs consumes about two CPU seconds.
Waiting for data adds elapsed time but little CPU work.

The report stores CPU work and one shared participant duration:

.. code-block:: json

   {
     "duration_seconds": 600,
     "resource_utilization": {
       "cpu_consumed": {
         "cpu_seconds": 180
       }
     },
     "issues": []
   }

The CLI calculates:

.. code-block:: text

   AVG CPU USED = CPU seconds / participant duration in seconds
                = 180 / 600
                = 0.3 CPUs

This is not 30 percent of the whole host. It is an average of 0.3 logical CPUs,
regardless of the job's visible CPU capacity. An average can exceed one when
work uses several CPUs.

Inspect a participant with:

.. code-block:: shell

   nvflare resources --job JOB_ID --site site-1
   nvflare resources --job JOB_ID --site site-1 --format json

What is covered
===============

Resident execution covers the job process and its threads, plus completed child
processes whose CPU time has been collected after waiting for them to exit.
Task execution instead adds coordinator self CPU to worker CPU once; those
workers are not also added through the coordinator's waited-child counter.
Custom or still-running subprocesses are not guaranteed to be included.

When NVFlare cannot confirm that a managed child's CPU was accounted for,
it marks the observed CPU work partial, shown as ``*``. An unavailable
counter is shown as ``—``, not zero. A missing GPU observation does not hide
a usable CPU average.

Collection details
==================

On Unix, NVFlare reads ``resource.getrusage(resource.RUSAGE_SELF)`` and
``resource.getrusage(resource.RUSAGE_CHILDREN)`` before and after execution.
Each provides user and system CPU seconds:

.. code-block:: text

   CPU work = change in self user + system time
            + change in waited-child user + system time

The participant's ``duration_seconds`` is the shared elapsed measurement window,
including application waiting. It starts after CPU/RAM and GPU probing,
and ends with the CPU counter read. Workers have their own internal windows.
Their work is summed,
but their durations are not added to the participant duration.
A missing or zero containing duration prevents calculation of an average;
recorded ``cpu_seconds`` remains usable.

``RUSAGE_CHILDREN`` includes terminated children after they have been waited
for. Grandchildren depend on the intervening processes waiting for their
children. A child whose lifetime crosses the measurement boundary may not
align exactly with the elapsed window.

Collection does not isolate the job into a new process group or discover
every custom subprocess. Known live workers, incomplete descendant accounting,
or lost earlier-attempt measurements leave the surviving subtotal partial.
