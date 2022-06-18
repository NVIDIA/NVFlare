.. _high_availability:

#####################################
High Availability and Server Failover
#####################################
Previously in NVIDIA FLARE 2.0 and before, the FL server was the single point of failure for the system. Starting with
NVIDIA FLARE 2.1.0, a high availability (HA) solution has been implemented to support multiple FL servers with
automatic cutover when the currently active server becomes unavailable.

The following areas were enhanced for supporting HA:

    - There can now be any number of FL servers (e.g. SP1, SP2, where SP stands for service provider), and only one
      of them is in active service mode (hot), with all others in standby mode (cold).
    - A new service called Overseer was added to oversee the overall availability of its clients (e.g. which SP is the
      hot or active one). Overseer setup is now required in provisioning.
    - State Storage is now used by the Overseer to keep system availability state info.
    - Overseer Agents were added the clients of the Overseer service (FL servers, FL clients, and admins) to constantly communicate
      with the Overseer to report its status and to get the current system status.
    - State Persistence is now used by the SPs (FL servers) to keep job execution state data so that the job execution can
      continue after cutting over to a new server. The State Persistence is shared and accessible by all FL servers.

.. note::

    SP means Service Provider. In NVIDIA FLARE, an SP is an FL server.

*************************
Automatic Server Failover
*************************
The most important feature of HA is automatic cutover to a standby server when the current hot or active server is out of
service, without human intervention. The Overseer and Overseer Agents help support this automatic SP cutover.

Overseer
========
The Overseer provides the authoritative endpoint info of the hot FL server. All other system entities (FL servers, FL
clients, admin clients) constantly communicate (i.e. every 5 seconds) with the Overseer (via the Overseer Agent in
them) to obtain such information, and act on it. Each communication to the Overseer also serves as a heartbeat of the
entity so the Overseer knows that the entity is still alive. If the hot FL Server missed a certain number of
heartbeats in a row, then Overseer will change to another FL Server as the hot SP, if available. Therefore at any
moment, there is at most one hot server.

The endpoint of the Overseer is provisioned and its configuration information is included in the startup kit of each entity.

For security reasons, the Overseer must only accept authenticated communications. In NVIDIA FLARE 2.1.0, the Overseer is
implemented with mTLS authentication.

Overseers maintain a service session id (SSID), which changes whenever any hot SP switch-over occurs, either by admin
commands or automatically.  The following are cases associated with SP switch-over and SSID:

    - If there is only one SP and it's assigned as the hot SP. The overseer associates this hot SP with one unique SSID.
    - If the above hot SP misses heartbeats and the overseer determines it loses communication with that SP, that SP
      becomes offline and is no longer a hot SP.
    - If there is another SP available, the overseer automatically denotes that SP as hot and associates a
      new unique SSID to it.
    - If there is no other SP available after the hot SP becomes offline, the overseer starts reporting no hot
      SP and no SSID.
    - During the above case, when either a new SP or the previous hot SP comes online, the overseer will create a new
      unique SSID and assign that SP as hot.
    - When the hot SP is keeping its online state by maintaining communication with the overseer, newly joined SPs are
      marked as online, but not hot SP.  Newly joined SPs also have no SSIDs associated with them, which means the
      original SSID is maintained with the current hot SP.
    - The overseer will include information on all joined SP and hot SP information in the reply to heartbeats.

Overseer Agent
==============
Overseer Agents provide a convenient and unified interface to communicate with the Overseer. By
following the same interface and behavior defined by the the Overseer Agent specification, users can implement their own
Overseer and Overseer Agent as a drop-in replacement.

The Overseer Agent periodically sends heartbeats to the Overseer. These heartbeats indicate that this Overseer Agent is still
alive and the communication between Overseer and Overseer Agent is not broken. When it fails in sending heartbeats,
the Overseer Agent shall retry until it is told to stop.

Each Overseer Agent includes a special property called "role". This property is set by the callers as the callers, namely FL
servers, FL clients or FL admins, have the knowledge about the role. For example:

    - The Overseer Agents used by FL servers have to tell the Overseer that it sends heartbeats on behalf of an SP, thus
      its role is "server."
    - The Overseer Agents used by admin must indicate its role so Overseers allow admin commands after the agents are
      authenticated and authorized.

The Overseer Agents internally maintain the last reply from Overseers, which is received during the last heartbeat.

Callers can register a callback. The execution of the callback is based on the flag ``conditional_cb``.

    - If the callback is registered with the conditional_cb flag set to True, the callback is called only when the SSID changes.
    - If that flag is set to False, the callback is called immediately every time after the reply of one heartbeat from the
      overseer is received.

Overseer Response Handling
==========================
The Overseer client is responsible for handling Overseer responses (or the lack of responses) properly.

FL Client
---------
No response from Overseer (connection error, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If I already have a hot SP, I’ll continue using it. I’ll keep retrying to communicate with the Overseer to obtain a response.

No hot SP available
^^^^^^^^^^^^^^^^^^^
If I already have a hot SP, I’ll continue using it. I’ll keep retrying to communicate with the Overseer to obtain a
different response.

Hot SP has not changed
^^^^^^^^^^^^^^^^^^^^^^
I’ll continue to use the current hot FL Server.

Hot SP has changed
^^^^^^^^^^^^^^^^^^
I’ll suspend current jobs and abort current running tasks (if any), and try to login to the new hot FL server. After that,
I will resume current jobs, if any. If I run into any communication issues with the new server, I will keep retrying
until success or the hot server endpoint changes again.

FL Server
---------
No response from Overseer (connection error, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I’ll stay in my current mode (hot or cold). I’ll keep retrying to communicate with the Overseer to obtain a response.

No hot SP available
^^^^^^^^^^^^^^^^^^^
I’ll stay in my current mode (hot or cold). I’ll keep retrying to communicate with the Overseer to obtain a
different response.

Hot SP is available
^^^^^^^^^^^^^^^^^^^
If I’m currently cold, and the hot SP is not me, then I stay cold.

If I’m currently hot, and the hot SP is me, then I stay hot.

If I’m currently cold, and the hot SP has changed to me, then I transition to the Cold-to-Hot state. In this state, I
will try to restart the unfinished jobs and get ready for client requests. Once ready, I transition to the hot state. If
any requests are received during the Cold-to-Hot state, I’ll tell them to try later.

If I’m currently hot, and the hot SP has changed to not me, then I transition to the Hot-to-Cold state. In this state,
I will prepare to stop serving the client requests. If any requests are received during the Hot-to-Cold state, I will
tell them I am not in service. This is a transition state to the cold state.

Admin Client
------------
No response from Overseer (connection error, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If I already have a hot SP, I’ll keep using it. I’ll keep retrying to communicate with the Overseer to obtain a response.

No hot SP available
^^^^^^^^^^^^^^^^^^^
If I already have a hot SP, I’ll keep using it. I’ll keep retrying to communicate with the Overseer to obtain a
different response.

Hot SP has not changed
^^^^^^^^^^^^^^^^^^^^^^
I’ll continue to use the current hot FL server.

Hot SP has changed
^^^^^^^^^^^^^^^^^^
I’ll try to login to the new hot FL server. After that, I will issue commands to the new hot server. If I run
into any communication issues with the new server, I will keep retrying until success or the hot server
endpoint changes again.

**************************
Job Execution Continuation
**************************
The secondary feature of HA is the continuation of job execution after SP cutover. NVIDIA FLARE implements a
snapshot-based job continuation mechanism.

    - Once a job is started, the server creates the first snapshot that remembers the basic job state (job ID,
      workspace, etc.).
    - During the execution of the job, the Controller initiates the creation of additional snapshots, based on its
      own control logic. Some controllers may decide not to create additional snapshots. For example, the
      Scatter-and-Gather controller works based on the concept of rounds, and it creates a snapshot after each round;
      whereas the cross-site-validation controller doesn’t create any snapshots.
    - After the SP cutover, the Controller will initiate the restoration of job execution state from the latest snapshot.
      If the Controller didn’t create additional snapshots, then the job will be executed from the beginning after the SP cutover.
    - Note that if clients detect the SP change, they will call "abort_task" to abort the current running task, because
      that task came from the previous SP. If at that moment there is a task running, it will be aborted with the "TASK_ABORTED"
      return code.
    - After the job execution completes, the job snapshot will be deleted from the snapshot storage. If the SP cutover
      occurs after the job execution completes, the completed job will not be migrated over.

************************
HA Running Job Migration
************************
All the FLComponents in the FL workflow have the option to implement the StatePersistable, which is to decide what
kind of data needs to persist and migrate to another server in the case of HA SP cutover. The FL snapshot includes
the current running state of all the FLComponents, the FLContext, and the current Job workspace. Once the HA SP cutover
occurs, the new SP will restore the FLContext, the Job workspace, and all the components' working states. Note that
depending on when the state is persisted, there is potentially a portion of work that may still be lost when the state
is restored.

FLCompoent
==========
Each FLComponent has its implementation to decide what kind of data it needs to persist and migrate, and then how
to restore from the persisted data.

FLContext
=========
FLContext keeps the system running data of the current job. Once HA SP cutover occurs, the same data will be
restored to the next SP. However, any non-serializable data in the FLContext will not be able to migrate and will be
discarded.

Workspace
=========
The running job workspace will also be migrated to the new SP, including the global model generated and
logs. The FLContext has variables indicating the workspace folder structure locations. When setting up the HA
servers, all the servers should choose the same folder locations to start the FL server.
