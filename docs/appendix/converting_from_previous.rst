###########################################################
Changes and Considerations Converting from NVIDIA FLARE 2.0
###########################################################
There are several breaking changes:

Provisioning Changes
====================
The project.yml file for NVIDIA FLARE 2.1 uses api_version 3, and it is not compatible with previous API versions because
the Provision Tool has been enhanced to support the configurations of additional components. See the example :ref:`project_yml`
that shows how these components are configured in the components section of StaticFileBuilder.

For details on the components and their configurations, see :ref:`concepts_and_system_components`.

Architecture Changes
====================
In 2.0, the FL server runs in a single process that does both management and execution of the job (one job at a time).
In 2.1, multiple jobs can be running concurrently, and each job runs in a separate child process. The main
process will only manage jobs (schedule, monitor, and manage job message routing, and command processing).

In 2.0, the FL client already runs the job in a separate child process. In 2.1, this will continue to be the case but
expanded to multiple child processes - one for each job.

The job ID (a UUID that is automatically generated) is the identifier for the job that is created when the job is submitted.
This replaces "run number" for operating the system (in 2.0, the user had to set an integer run number manually in order to deploy and
start a run).

Job Preparation
===============
In 2.0, admin users (researchers) conduct a RUN by first creating apps, and then uploading/deploying/starting the
apps interactively via admin commands. As discussed above, this changes in 2.1, where the user simply submits the
jobs and lets the system do the rest.

This means that the user, instead of creating only apps, will need to create jobs (an app that is uploaded as job will
automatically be converted into a simple job for compatibility without creating a job). A job is a folder that contains
one or more apps, and the extra meta.json file. Please look under job above to see sample contents of meta.json as
well as the keys that are required. The “deploy_map” will specify which sites each app should be deployed to, and all
of the specified apps must be in the job folder.

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
    - download_job: after the job is finished, its resulting workspace will be stored in the Job Store. This command downloads the job and workspace to the user’s machine.

Some commands are modified for the job-centric behavior:

    - The "abort" command now requires a job ID since there can be multiple jobs running (the command has been renamed to "abort_job").
    - The “abort_task” command now requires a job ID since there can now be multiple jobs running at the same time.

Some commands were also added for high availability to gain more insight into the system:

    - list_sp: list the information for all SPs
    - get_active_sp: get the information for the current active SP
    - promote_sp: promote a specified SP to become the active SP (promote_sp example1.com:8002:8003)
