.. _resource_statistics_message_traffic:

Message Traffic
***************

Message traffic records payload bytes sent and received by the job's messaging
processes. It is a communication-cost proxy, not a network-interface counter
or a guarantee that the receiving application processed the data.

What is counted
===============

The counters include task data, separately transferred tensors, control
payloads, and repeated transfers. A retried chunk counts again because it
requires another transfer. Payload encryption is included when enabled;
CellNet headers and TCP/TLS overhead are not.

The report covers the server or client's job cell, task-worker cells, and managed
external trainers. A cell is NVFlare's messaging endpoint within a process. Transfers
between processes count even when both processes run on one machine. Direct
calls between cells in the same Python process do not count.

Shared parent-process traffic, such as application deployment and parent
relaying, is not assigned to an individual job. Arbitrary custom processes
and communication outside CellNet are also outside this measurement.

Read the output
===============

``PAYLOAD SENT GiB`` and ``PAYLOAD RECEIVED GiB`` show the two directions
independently. Inspect them with:

.. code-block:: shell

   nvflare resources --job JOB_ID
   nvflare resources --job JOB_ID --site site-1 --format json

A participant stores both counters:

.. code-block:: json

   {
     "resource_utilization": {
       "message_traffic": {
         "sent_bytes": 1073741824,
         "received_bytes": 536870912
       }
     },
     "issues": []
   }

This participant sent 1 GiB and received 0.5 GiB during collection.
A successful zero counter is written as ``0``. Missing observations are
not assumed to be zero.

Job and study views sum each direction separately. Do not add sent and
received totals to estimate unique transferred data: a transfer can appear
at both its sending and receiving participant. Relayed data can cross more
than one counted connection.

Collection details
==================

When bytes are counted
----------------------

CellNet counts the actual encoded payload passed to each neighboring endpoint,
after payload encryption when enabled. A send contributes after the local
send call returns without an error. Transport drivers can report failures
later, so this is not proof of remote delivery. The receive counter observes
payload arriving at the receiving cell, before application decoding.

Stream frames and download chunks use these same counters. Their actual
payload lengths count, including retries; no declared model size or logical
message count is needed. Empty payloads contribute zero, even though their
headers still have a transport cost.

The report takes one snapshot of the counters during final collection. It
does not wait for transfers to finish solely for statistics. Traffic after
that snapshot is outside the report, so sent and received values need not
match across participants.

Managed external trainers
-------------------------

A managed trainer can transfer tensors directly without passing them through
the client's job process. Its existing result-completion and shutdown exchanges
return its cumulative sent and received counters. The reporting code adds the latest
snapshot from each managed trainer to the job-process counters; repeated
snapshots replace earlier ones rather than being added again.
Trainer traffic after its latest returned snapshot is outside the report,
including the message that carries that snapshot.

The existing authenticated exchange associates the trainer with its client
and job. No new credential or operator configuration is required. If a
required trainer snapshot is unavailable, the report retains the available
counts and marks them incomplete. This does not reject downloads or change
transfer cleanup.

Task workers
------------

Each task worker snapshots its endpoint counters before closing its Cell and
includes them in its completion diagnostics. When the transport is still active,
the worker marks traffic partial because shutdown can send or receive more bytes.
The coordinator counts the worker's contribution once after settlement.
The two endpoints of a coordinator/worker connection count their own directions;
these totals are not deduplicated network traffic. A missing worker record leaves
the available counters incomplete.

Source reference
----------------

The :github_nvflare_link:`peer counters <nvflare/fuel/f3/traffic_stats.py>`
and :github_nvflare_link:`CellNet transport boundary <nvflare/fuel/f3/cellnet/core_cell.py>`
record bytes. The
:github_nvflare_link:`resource collector <nvflare/private/fed/resource_stats/collector.py>`
assembles the participant report outside CellNet.
