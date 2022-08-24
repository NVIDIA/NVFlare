.. _multi_job:

###########################################
Jobs: Defining Jobs and Multi-Job Execution
###########################################
Newly introduced in NVIDIA FLARE 2.1.0, Jobs now organize and streamline the running of apps to allow for multi-job
execution and the running of multiple experiments in parallel.

********
Concepts
********

Job
===
In NVIDIA FLARE 2.1.0, to be able to run multiple experiments in parallel, the system needs to be able to make decisions
as to when to run which experiments.

To be able to do this, the system now has to know everything about the experiment: which app(s)
go to which clients or server, what are the resource requirements for this experiment, etc.
The total definition of such needed information is called a Job.

Many underlying assumptions about the operation of the system changed, with the admin now submitting a job and letting the
system manage the rest, instead of before when the admin uploaded apps, set the run number, deployed, then started the app
at the server and client sites.

Jobs now contain all of the :ref:`apps<application>` and the information of which apps to deploy to which sites as the
:ref:`deploy_map<deploy_map>` inside a meta.json that should be included with a job to be submitted::

    JOB_FOLDER:
        - meta.json
        - APP_FOLDER_FOR_SERVER    (required to have config/config_fed_server.json)
        - APP_FOLDER_FOR_CLIENT1   (required to have config/config_fed_client.json)
        - APP_FOLDER_FOR_CLIENT2   (required to have config/config_fed_client.json)
        - APP_FOLDER_FOR_CLIENT... (required to have config/config_fed_client.json)

.. note::

   For backward compatibility with previous apps, a single app may be submitted as a job, and a meta.json will
   automatically be created for it with the app being deployed to all participants. As such, apps can have both
   config_fed_server.json and config_fed_client.json and can be deployed to multiple participants.

Here is an example for meta.json::

    {
      "name": "try_algo1",
      "resource_spec": {
        "client1": {
          "num_gpus": 1,
          "mem_per_gpu": "16G"
        },
        "client2": {
          "num_gpus": 1,
          "mem_per_gpu": "16G"
        }
      },
      "deploy_map": {
        "hello-numpy-sag-server": [
          "server"
        ],
        "hello-numpy-sag-client": [
          "client1",
          "client2"
        ],
        "hello-numpy-sag-client3": [
          "client3"
        ]
      },
      "min_clients": 2,
      "mandatory_clients": [
        "client1",
        "client2"
      ]
    }

Pay attention to the following:

    - name: user provided name for the job
    - resource_spec: resources required to perform this job at each site
    - deploy_map: what apps go to which sites (see :ref:`deploy_map`)
    - min_clients: minimum clients required for this job
    - mandatory_clients: mandatory clients required for this job

The system also keeps additional information about the job such as:

    - Submitter name
    - Time of submission
    - Current status of the job (submitted, approved, running, completed, etc.)
    - Location of the final result

Resources
=========
For a job to be runnable, the system must have sufficient resources: all relevant sites of the job must be able to
support the job’s specified resource requirements. Since resource is a generic concept - anything could be regarded
as a resource - NVIDIA FLARE 2.1.0 itself does not define any specific resources. Instead, NVIDIA FLARE provides a general
framework for resource definition and interpretation.

.. _deploy_map:

Deploy Map
==========
The ``deploy_map`` is a map of which apps in the job being uploaded will be deployed to which FL client sites. Back in
NVIDIA FLARE before 2.1.0, the admin command "deploy_app" was used to manually perform app deployment with the option
to specify which sites to deploy to. Because the JobRunner now automatically picks up and handles the deployment and
running of apps, it needs information about which sites each app should be deployed to, and it gets it from the
``deploy_map`` section of meta.json.

Each app specified in the ``deploy_map`` must be included in the job being uploaded as an app folder directly in the job
folder with meta.json.

There is only one server, and only one app can be deployed to it for the Job, so "server" can appear only once in
the ``deploy_map``.

The ``deploy_map`` cannot be empty, so the following is not allowed::

    "deploy_map": {}

When specified as a site name, "@ALL" carries a special meaning of all sites to deploy to. If "@ALL" is used, there
should be no other apps being deployed to the sites. This means the following example of ``deploy_map`` is not allowed::

    "deploy_map": {
        "app1": ["@ALL"], "app2": ["site-1"]
    }

If an empty list of sites is specified for an app in the ``deploy_map``, then that app is to be deployed to no sites,
and no validation is done other than checking that the folder exists. This is the case for "app2" in the following valid
example of ``deploy_map`` for a job containing app1 and app2::

    "deploy_map": {
        "app1": ["@ALL"], "app2": []
    }

Resource-less Jobs
==================
Similarly, for simple FL jobs or in POC mode, resources are not a concern. In this case, the resource spec can be
omitted from the job definition. The FL client always answers "Yes" when asked whether it can run a job without
required resources.

Resource-Based Job Automation
=============================
Each job specifies resource requirements (the resource_spec in the meta.json), which is expressed as a Python dictionary: the key/value pairs can specify any arbitrary requirement;

There is a Job Scheduler on the Server, which decides whether a job is runnable. It asks these clients
whether they can run the job, given the resource requirements (note: the job could have different requirements for
different clients).

On each client, there is a Resource Manager component, it will check whether the resource requirements coming from a job
can be satisfied (using a check_resources method).

If runnable clients meet the job’s client requirements (minimum number of clients and mandatory clients), then the
job is runnable for the system, and the job is dispatched to these clients.

When checking resources, some clients might reserve resources. (like running an instance from the cloud).

After checking all the clients and if the Job Scheduler decides the job is not runnable. The client’s Resource
Manager will be called to cancel the resources it might have reserved for the job (using the cancel_resources method in
Resource Manager).

The Job Scheduler is invoked periodically to try to run as many jobs as possible.

Once a job is dispatched to a client, the Resource Manager is called to allocate the required resources
(using the allocate_resources method). Once the job is started on the client, it will call the Resource Consumer to consume the
resources.

Once the job is finished (completed normally or aborted), the Resource Manager is called again to free the resources (using
the free_resources method).


Example of GPU-based job automation
-----------------------------------
Here is an example of GPU-based job automation, where a job is deployed only if clients have enough GPUs.

First, the resource requirement of GPUs is defined as the key/value pair of “num_gpus”/integer in the job’s
resource_spec, say, “num_gpus”: 2.

Second, the Resource Manager on the Client decides whether it has 2 GPUs when called. This could be done by
statically configuring available GPUs at the start of the Resource Manager, or it might be able to auto-detect. Here
we use a simple Resource Manager that takes the 1st approach: it has a list of available GPU IDs. When called to
check resource requirements, it simply checks whether the list contains at least 2 GPU IDs.

Third, if the Job Scheduler decides to run the job, the Resource Manager will be called to allocate the 2 required
GPUs - it will return a list of 2 GPU IDs and remove them from the list of available GPUs .

Fourth, when the job is started (in a separate “bubble”), the Resource Consumer will be called to consume the
resources (which is the list of 2 GPU device IDs). In this case, this Resource Consumer simply sets the
CUDA_VISIBLE_DEVICES system variable to the 2 GPU IDs. This ensures that each concurrent job will be using different
GPU devices.

Finally, when the job is finished, the Resource Manager is called to free the allocated resources. In this case, it
simply puts the 2 GPU IDs back to its list.


Job Runner
==========
The Job Runner is responsible for managing jobs at runtime. It is responsible for:

    - Deciding when to schedule a new job
    - Monitoring the progress of running jobs
    - Managing job execution state and ensuring the server and clients are in sync

The Job Runner periodically checks if there are new submitted / approved jobs from the job
manager. If there are jobs have not been run, Job runner sends the job candidates to the job scheduler to check for
the job readiness. Once the job scheduler returns the job which satisfies the running condition and resource
requirements for the clients, the job runner will dispatch the FL application for the server and each client to the
corresponding destination. Then the job runner will start the FL server application and client applications to run
the job.

The job runner keeps track of the running jobs and the corresponding job ids. Once a job finishes running, or the
job execution got aborted, the job runner will remove the job id from the running_jobs table.

One-Shot Execution
------------------
Once submitted, a job only has one chance to be executed, whether the execution succeeds or not. Once executed, the
job status will be updated and won’t be scheduled again. If the user wants to run the same job again, the user can
use the “clone job” command to make a new job from an existing job; or the user can submit the same job definition
again.

System State Self Healing
-------------------------
It is important for the FL server and clients to be in sync in terms of job execution. However, in a distributed
system, it is impossible to keep all parts of the system in sync at all times. For example, when deploying or
starting the job, some clients may succeed while others may fail. NVIDIA FLARE implements a heartbeat-based mechanism for
the system to keep in sync most of the time. In case they become out of sync, the mechanism can also gradually bring the parties
back in sync.

Each FL client periodically sends heartbeat messages to the FL server. The message contains the job IDs of the
jobs that the client is running. The server keeps the job IDs of the jobs that each site should be running. If
there is a discrepancy with the client running a job that should not be running, the server will ask the client to
abort it.
