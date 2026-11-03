.. _auditing:

Auditing
========
NVFLARE records selected user command events and job events through explicit audit calls. Audit output is
separate from application logging: an application log message does not automatically become an audit event.

The audit logging system is primarily implemented in :class:`Auditor<nvflare.fuel.sec.audit.Auditor>`.

Audit File Locations
^^^^^^^^^^^^^^^^^^^^
Without output-root environment variables, each server or client parent writes to
``<workspace>/audit.log`` and each server or client job process writes to
``<workspace>/<job_id>/audit.log``. The job directory name does not include a ``run_`` prefix.

The effective audit root is selected in this order:

1. ``NVFL_AUDIT_ROOT``
2. ``NVFL_LOG_ROOT``
3. ``NVFL_RESULT_ROOT``
4. The workspace defaults described above

With an output-root override, parent files are at ``<root>/<site_name>/audit.log`` and job files are at
``<root>/<site_name>/<job_id>/audit.log``. These paths are constructed by
:meth:`Workspace.get_audit_file_path<nvflare.apis.workspace.Workspace.get_audit_file_path>`.

Parent processes initialize auditing in ``security_init()`` and job processes in ``security_init_for_job()``
from ``nvflare.private.fed.utils.fed_utils``. Auditing is initialized in both secure and non-secure modes.
The file is opened immediately in append mode, so an empty file can exist before the first event and a restart
retains existing records. ``AuditService`` holds one auditor per process; repeated initialization uses the existing
auditor until it is closed. An isolated task worker initializes a job auditor only if none is already present.

The client sub-worker currently initializes the site-level audit path even though it executes within a job.
Consequently, more than one process can target the same file. Simulator runner and worker processes use a no-op
``SimulatorAuditor`` and do not generate these audit files through their normal audit calls.

To access the audit files for a job, run ``download_job <job_id>`` and look in the downloaded job's workspace folder
for the server job and client job logs (by default named ``audit.log``).

Audit File Format
^^^^^^^^^^^^^^^^^^
The audit file is a text file. Each audit call writes headers and an optional message followed by a newline.
The current writer does not escape embedded newlines or header delimiters in values, so a physical line is not
a reliable record boundary for arbitrary input. Messages are appended directly after the headers, without an
``[M:...]`` wrapper. Timestamps use the process's local time without a timezone offset.
Event headers are enclosed in square brackets. The following are some examples of events:

.. code-block::

    [E:b6ac4a2a-eb01-4123-b898-758f20dc028d][T:2022-09-13 13:56:01.280558][U:?][A:_cert_login admin@b.org]
    [E:16392ed4-d6c7-490a-a84b-12685297e912][T:2022-09-14 12:59:47.691957][U:trainer@b.org][A:train.deploy]
    [E:636ee230-3534-45a2-9689-d0ec6c90ed45][R:9dbf4179-991b-4d67-be2f-8e4bac1b8eb2][T:2022-09-14 15:08:33.181712][J:c4886aa3-9547-4ba7-902e-eb5e52085bc2][A:train#39027d22-3c70-4438-9c6b-637c380b8669]received task from server

Event Headers
^^^^^^^^^^^^^^^^^^
Event headers specify meta information about the event. Each header is expressed with the header type (one character),
followed by a colon (:) and the value of the header. The following are defined header types and their values.

.. csv-table::
    :header: Header Type,Meaning,Value
    :widths: 5, 10, 20

    E,Event ID,A UUID for the ID of the event
    T,Timestamp,Local time of the event without a timezone offset
    U,User,Name of the user
    A,Action,User issued command or job's task name and ID
    J,Job,ID of the job
    S,Scope,Name of the job scope
    R,Reference,Caller-supplied reference usually identifying a related event

Most of the headers are self-explanatory, except for the R header. Events can be related. For example, a user command
could cause an event to be recorded on both the server and clients. Similarly, a client's action could cause the server
to act on it (e.g. client submitting task results). The R header records the related event ID on the peer. Reference
event IDs can help to correlate events across the system. The reference is optional and is not always an event ID:
component authorization events currently use a job ID as their reference.

General events contain ``E``, ``T``, ``U`` and ``A``. Job events contain ``E``, ``T`` and ``J``, with ``S`` when
the scope is nonempty and ``A`` in the form ``task_name#task_id`` when a task name is supplied. Both forms include
``R`` only when a reference is supplied. The job-context helper substitutes ``?`` for missing scope and task values.

Current Event Coverage
^^^^^^^^^^^^^^^^^^^^^^
The following table describes existing audit points, rather than guarantees that an operation succeeded.

.. list-table::
    :header-rows: 1
    :widths: 25 35 40

    * - Source
      - Recorded event
      - Interpretation and coverage limits
    * - Server admin dispatch
      - Username and first 100 characters of the command, before handler execution
      - Authentication and authorization filters run first. Rejected requests and unknown commands do not reach this audit filter. The entry does not record handler completion.
    * - Client admin dispatch
      - Forwarded username, request topic and originating server event reference
      - Recorded before client-side processing and authorization checks; receipt does not establish authorization or completion.
    * - Server task dispatch
      - Task assignment event, with its ID attached to the outgoing task
      - Establishes preparation of the assignment, not proof that the client received it.
    * - Client task execution
      - Task receipt, executor invocation and result preparation, including selected failure paths
      - Receipt references the server event. A result-preparation event does not establish successful network delivery.
    * - Server result receipt
      - Result receipt referencing the client result event
      - Recorded before subsequent peer/job/attempt validation; receipt does not establish acceptance.
    * - DXO filtering
      - Applied filter name, source and chain type
      - Recorded when processing returns a truthy valid DXO; skipped filtering does not generate this event.
    * - Component path authorization
      - Selected allow-list policy decisions, including warning-mode allowance
      - Successful records are deduplicated by policy/scope/component keys; these are not a record of every component check.

``Auditor.add_event()`` suppresses actions exactly equal to ``scheduler.check_resource``, ``_check_session``,
``_commands`` or ``__aux_command__``. This exclusion does not apply to ``add_job_event()``.

Write and Lifecycle Guarantees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each auditor serializes its own thread access using a lock and flushes the file after every record. This is not
a cross-process lock, and flushing does not provide the crash-durability guarantee of ``fsync()``. The current
implementation does not implement rotation, retention, centralized export or tamper-evident storage.

Calls through ``AuditService`` before initialization or after service closure return an empty event ID without
writing. An already closed ``Auditor`` also ignores late calls. Closing waits for an admitted write using the
same thread lock. File open, write and flush errors can propagate to callers; there is no dedicated audit-health
or automatic recovery mechanism in this writer.

The :doc:`audit improvement program <../../../audit-improvement-program>` tracks proposed changes and staged
acceptance criteria. Those planned capabilities are not current configuration options or guarantees.
