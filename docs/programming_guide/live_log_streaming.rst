.. _live_log_streaming:

####################
Live Log Streaming
####################

FLARE can stream a job's log files from each client to the server **as the job
runs**, so an operator can ``tail -f`` the server-side copy in real time.
Streamed logs are written under the server workspace and, when a job manager
is available, automatically attached to the job's persisted artifacts.

This page describes how the feature works and how to enable it. To opt out at
a particular site, see :ref:`allow_log_streaming` in
:ref:`site_config`.

Overview
========

The feature is a simple producer / consumer pair built on top of FLARE's
existing object-streaming machinery:

- The **producer** runs inside the client's job subprocess. It tails one or
  more log files (``log.txt``, ``error_log.txt``, custom files) and pushes
  new bytes to the server as they are written.
- The **consumer** runs in the server process. It opens a destination file
  per stream and writes incoming chunks directly, so the file is readable
  with ``tail -f`` while the job is still running. When the stream closes
  (normal end, abort, or idle timeout) the file is handed to the job
  manager for storage.

A third widget, the **system streamer**, lives in the client's
``resources.json`` and saves users from declaring a streamer in every job
config — it auto-injects a ``JobLogStreamer`` into each job before launch.

Components
==========

JobLogStreamer (client, in-job)
-------------------------------

:class:`~nvflare.app_common.logging.job_log_streamer.JobLogStreamer` runs
inside the job subprocess (``CLIENT_JOB``). It belongs in the **job-level**
client configuration (``config_fed_client.json``):

.. code-block:: json

    {
      "components": [
        {
          "id": "log_streamer",
          "path": "nvflare.app_common.logging.job_log_streamer.JobLogStreamer",
          "args": {}
        }
      ]
    }

To stream more than one log file, declare one component per file:

.. code-block:: json

    {
      "components": [
        {
          "id": "log_streamer",
          "path": "nvflare.app_common.logging.job_log_streamer.JobLogStreamer",
          "args": {"log_file_name": "log.txt"}
        },
        {
          "id": "error_log_streamer",
          "path": "nvflare.app_common.logging.job_log_streamer.JobLogStreamer",
          "args": {"log_file_name": "error_log.txt"}
        }
      ]
    }

**Constructor arguments**

``log_file_name`` (str, default ``"log.txt"``)
    Base name of the log file to stream. Must be a relative file name; absolute
    paths and ``..`` traversal are rejected. The actual file is located by
    inspecting the active Python file handler and reusing its directory, so
    streaming works the same way under the simulator and in production
    without any workspace path arithmetic.

``liveness_interval`` (float, default ``10.0``)
    Seconds between heartbeat messages when the log file has produced no new
    bytes. Must be **strictly less than** the receiver's ``idle_timeout`` so
    heartbeats keep the stream alive during quiet periods.

``poll_interval`` (float, default ``0.5``)
    Seconds between polls when the log file has no new data.

**Lifecycle**

The streamer fires on three events:

- ``START_RUN`` — opens the stream and starts a daemon tailing thread.
- ``ABOUT_TO_END_RUN`` — signals the streaming thread to drain and stop, but
  does not block, so post-event log lines still land in the file and are
  picked up by the drain.
- ``END_RUN`` — joins the streaming thread (with a 60-second timeout) so the
  server has received the EOF before the client's ``client_run`` returns.

JobLogReceiver (server)
-----------------------

:class:`~nvflare.app_common.logging.job_log_receiver.JobLogReceiver` opens a
destination file per incoming stream and writes chunks as they arrive. It can
be placed either in **system-level resources** so every job is covered, or
in **job-level configuration** to scope the receiver to a single job.

System-level (``resources.json`` on the server, recommended)::

    {
      "components": [
        {
          "id": "log_receiver",
          "path": "nvflare.app_common.logging.job_log_receiver.JobLogReceiver",
          "args": {}
        }
      ]
    }

Job-level (``config_fed_server.json``) — declare the same component there if
you want the receiver to register only for that job. The widget keys off the
``SYSTEM_START`` event in system mode and ``START_RUN`` in job mode, so the
underlying stream handler is registered exactly once.

In the Job API, you can attach a job-level receiver with:

.. code-block:: python

    job.to_server(JobLogReceiver())

**Constructor arguments**

``dest_dir`` (str, default ``None``)
    Directory where incoming log files are written. Defaults to the system
    temporary directory.

``idle_timeout`` (float, default ``30.0``)
    Seconds without any message (data or heartbeat) before the receiver
    declares the sender dead and closes the stream. Set to ``0`` to disable.

**File layout**

Each chunk is appended to::

    {dest_dir}/{job_id}/{client_name}/{log_file_name}

so an operator can find logs while the job is still running. When the stream
ends successfully, the file is handed to the job manager (if registered) for
permanent storage; otherwise — for example under the simulator — it is
moved into the job's workspace run directory alongside the other artifacts.

If the stream ends with a non-OK return code (e.g. idle timeout), the
**partial** file is retained at the path above and a warning is logged.

SystemLogStreamer (client, system widget)
-----------------------------------------

:class:`~nvflare.app_common.logging.system_log_streamer.SystemLogStreamer` is
a convenience widget that lives in the **client's** ``resources.json`` and
removes the need to declare a streamer in every job:

.. code-block:: json

    {
      "components": [
        {
          "id": "system_log_streamer",
          "path": "nvflare.app_common.logging.system_log_streamer.SystemLogStreamer",
          "args": {}
        }
      ]
    }

On ``BEFORE_JOB_LAUNCH`` (after the deployed job config has been written to
disk but before the subprocess starts), it reads the deployed
``config_fed_client.json``, and if no ``JobLogStreamer`` is already declared
it appends one with the configured arguments. The job subprocess then loads
the modified config and runs ``JobLogStreamer`` as if the user had declared
it explicitly.

When configured for ``error_log.txt``, ``SystemLogStreamer`` also uploads a
post-mortem snapshot from the client parent process on ``JOB_COMPLETED``.
This guarantees error-log delivery for failures that happen so early in the
job that ``JobLogStreamer`` never reaches ``START_RUN``.

The constructor takes the same ``log_file_name``, ``liveness_interval`` and
``poll_interval`` arguments as :class:`JobLogStreamer`; any non-default values
are forwarded to the injected component.

Site control
============

Live log streaming is **enabled by default**. A site can opt out by setting
``"allow_log_streaming": false`` in its ``resources.json``; see
:ref:`allow_log_streaming` for the full description of how each component
behaves when streaming is disabled, including the server-side check that
logs an error if a chunk arrives from a site that has disabled it.

Wire protocol
=============

Streaming uses FLARE's :class:`LogStreamer` over the
``log_streaming`` channel with topic ``live_log``. The stream context carries
the trusted client name and job ID derived from the peer FL context, so
filenames on disk reflect the actual sender — they cannot be spoofed by the
streaming client through the stream payload. Each chunk is a sequence of
log bytes; heartbeats are sent every ``liveness_interval`` seconds when no
bytes have been written.

Behavior under abort
====================

The streaming thread runs in a fresh FL context whose abort signal is never
triggered, so an aborted job's still-buffered log bytes can drain to the
server before the run actually shuts down. Graceful shutdown is signaled
exclusively via the streamer's stop event, set in ``ABOUT_TO_END_RUN`` and
joined in ``END_RUN``. If the join exceeds 60 seconds, a warning is logged
and the run continues to shut down — the server will see the stream close
with an idle-timeout return code rather than EOF, and will retain whatever
partial log has been written so far.
