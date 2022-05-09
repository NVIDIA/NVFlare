###########################################################
Changes and Considerations Converting from NVIDIA FLARE 2.0
###########################################################
There are several breaking changes:

Provisioning Changes
====================
The project.yml file for NVIDIA FLARE 2.1 uses api_version 3, and it is not compatible with previous API versions because
the Provision Tool has been enhanced to support the configurations of additional components. See the example :ref:`project_yml`
that shows how these components are configured in the components section of StaticFileBuilder.

Overseer
--------
The Overseer is a new system component that determines the hot FL server at any time.
The name of the overseer must be unique and in the format of fully qualified domain names.  During
provisioning time, if the name is specified incorrectly, either being duplicate or containing incompatible
characters, the provision command will fail with an error message.

NVFLARE 2.1 comes with HTTPS-based overseer.  Users are welcome to change the name and port arguments of the overseer
in project.yml to fit their deployment environment.

The Overseer will receive a Startup kit, which includes the start.sh shell script, its certificate and private key,
root CA certificate, privileged user list file and a signature file.  The signature file contains signatures of each
file, signed by root CA.  This is to ensure the privileged user list file is not tampered.

Overseer Agent
--------------
This is the component that communicates with the Overseer on the client's behalf.
Overseer agent config info is included in the Startup Kits of FL Servers, FL Clients, and Admin Clients.

The provisioning tool generates the overseer agent section in fed_server.json, fed_client.json, and admin.json with
information gathered from the project.yml file.  For example, if the overseer agent section specifies the listening
port 7443, the overseer agent section of all fed_server.json, fed_client.json and admin.json contains that port
information.

The other important requirement is this agent must be able to communicate with the overseer specified above.  Users
may implement their own overseer based on their deployment environment.  In that case, users also need to implement
their own overseer agent.

For NVFLARE 2.1, the provisioning tool will automatically map parameters specified in overseer into the arguments for
the HttpOverseerAgent.

When DummyOverseerAgent is specified, the provisioning tool will include all arguments into the overseer agent section of generated json files.

Study Storage
-------------
The Study definition is stored in a persistent store. The Study Storage config specifies the Python object that manages the access to the store.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server’s Startup Kit.

Study Manager
-------------
The Study Manager config specifies the Python object that manages the retrieval of Study Definition objects stored in the Study Storage.

The system reserved component id, study_manager, is used to denote the Study Manager in the project.yml file.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server’s Startup Kit.

Job Storage
-----------
The Job definition is stored in a persistent store. The Job Storage config specifies the Python object that manages the access to the store.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server’s Startup Kit.

Job Definition Manager
----------------------
The Job Definition Manager config specifies the Python object that manages the access and manipulation of Job Definition objects stored in the Job Storage.

The system reserved component id, job_manager, is used to denote the Job Definition Manager in the project.yml file.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server’s Startup Kit.

Job Execution State Storage
---------------------------
The Job Execution State is stored in a persistent store. The Job Execution State Storage config specifies the Python
object that manages the access to the store.

This configuration is included in the fed_server.json of the Server’s Startup Kit.

Job Scheduler
-------------
The Job scheduler is responsible for determining the next job to run. Job scheduler config specifies the Job scheduler Python object.

The system reserved component id, job_scheduler, is used to denote the Job Scheduler in the project.yml file.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server’s Startup Kit.

Resource Manager
----------------
The Resource Manager is responsible for managing job resources on FL Client. Resource Manager config specifies the Resource Manager Python object.

The system reserved component id, resource_manager, is used to denote the Resource Manager in the project.yml file.

This component is specified as one item in the components.client section.

This configuration is included in the fed_client.json of the FL Client’s Startup Kit.

Resource Consumer
-----------------
The Resource Consumer is responsible for consuming and/or initializing job resources on FL Client. The Resource Consumer
config specifies the Resource Consumer Python object.

This configuration is included in the fed_client.json of the FL Client’s Startup Kit.

The system reserved component id, resource_consumer, is used to denote the Resource Consumer in the project.yml file.

This component is specified as one item in the components.client section.


Architecture Changes
====================
In 2.0, the FL server runs in a single process that does both management and execution of the job (one job at a time).
In 2.1, multiple jobs can be running concurrently, and each job runs in a separate child process. The main
process will only manage jobs (schedule, monitor, and manage job message routing, and command processing).

In 2.0, the FL client already runs the job in a separate child process. In 2.1, this will continue to be the case but
expanded to multiple child processes - one for each job.

The running of a job is still called a run. The job ID (a UUID that is automatically generated) is the value of "run
number" when the job is running (in 2.0, the user had to set an integer run number manually in order to deploy and
start a run).

Job Preparation
===============
In 2.0, admin users (researchers) conduct a RUN by first creating apps, and then uploading/deploying/starting the
apps interactively via admin commands. As discussed above, this will change in 2.1, where the user simply submits
jobs and lets the system do the rest.

This means that the user, instead of creating only apps, will need to create jobs. A Job is a folder that contains
one or more apps, and the extra meta.json file. Please look under Job above to see sample contents of meta.json as
well as the keys that are required. The “deploy_map” will specify which sites each app should be deployed to, and all
of the specified apps must be in the Job folder.

Admin Client
============
The Admin Client has been integrated with Overseer Agent. It can now dynamically change to use the new FL server in case
an SP cutover happens.

Due to automated job execution, the following commands are no longer supported/needed in 2.1:

    - set_run_number
    - deploy_app
    - start_app
    - abort

New commands are introduced to help user understand the states of submitted jobs:

    - submit_job: submit a prepared job to the system. A unique Job ID is returned to the user if the submission is successful. The user can use this ID to query the status of the job later.
    - list_jobs: list jobs that are in the system already (flags can be used for filtering)
    - abort_job: abort job if it is already running or dispatched
    - delete_job: delete job from the system
    - get_job_result: after the job is finished, its result will be stored in the Job Store. This command downloads the result to the user’s machine.

Some commands are modified for the job-centric behavior:

    - The "abort" command now requires a job ID since there can be multiple jobs running (the command renamed to "abort_job").
    - The “abort_task” command now requires a job ID since there can now be multiple jobs running at the same time.

Some commands were also added for high availability to gain more insight into the system:

    - list_sp: list the information for all SPs
    - get_active_sp: get the information for the current active SP
    - promote_sp: promote a specified SP to become the active SP (promote_sp example1.com:8002:8003)
