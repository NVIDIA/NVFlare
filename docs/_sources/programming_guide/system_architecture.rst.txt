###################
System Architecture
###################

NVIDIA FLARE is designed with the idea that less is more, using a spec based design principle to focus on what is
essential (solutions to hard, tedious problems others do not want to solve) and to allow other people to be able to do
what they want to do in real world applications. FL is an open ended space, so the spec based design allows others to
bring their own implementations and solutions for various components.

.. _concepts_and_system_components:

******************************
Concepts and System Components
******************************

Spec-based Programming for System Service Objects
=================================================
NVIDIA FLARE 2.1.0 needs additional services to implement the HA feature:
storage, overseer, job definition management, etc. There are many ways to implement such services. For example,
storage could be implemented with a file system, AWS S3, or some database technologies. Similarly, job definition
management could be done with simple file reading or a sophisticated solution with a database or search engine.

To allow any of these solutions in NVIDIA FLARE, we take a spec-based approach for such objects. Each such service will
provide an interface definition (the spec), and all implementations must follow the spec. The spec defines the
required behaviors for the implementation. At the same time, we provide some implementations that follow these specs.

Spec definitions and provided implementations are below for the different system components.

.. _system_components:

System Components
=================
See the example :ref:`project_yml` for how these components are configured in StaticFileBuilder.

Overseer
--------
The Overseer is a system component newly introduced in 2.1.0 that determines the hot FL server at any time for high availability.
The name of the Overseer must be unique and in the format of fully qualified domain names.  During
provisioning time, if the name is specified incorrectly, either being duplicate or containing incompatible
characters, the provision command will fail with an error message. It is possible to use a unique hostname rather than
FQDN, with the IP mapped to the hostname by having it added to ``/etc/hosts``.

NVIDIA FLARE 2.1.0 comes with HTTPS-based overseer.  Users are welcome to change the name and port arguments of the overseer
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

The other important requirement is this agent must be able to communicate with the Overseer specified above.  Users
may implement their own Overseer based on their deployment environment.  In that case, users also need to implement
their own Overseer Agent.

:class:`Overseer Agent Spec<nvflare.apis.overseer_spec.OverseerAgent>`

NVIDIA FLARE provides two implementations:

    - :class:`HttpOverseerAgent<nvflare.ha.overseer_agent.HttpOverseerAgent>` to work with the Overseer server. For NVIDIA
      FLARE 2.1.0, the provisioning tool will automatically map parameters specified in Overseer into the arguments for
      the HttpOverseerAgent.
    - :class:`DummyOverseerAgent<nvflare.ha.dummy_overseer_agent.DummyOverseerAgent>` is a dummy agent that simply
      returns the configured endpoint as the hot FL server. The dummy agent is used when a single FL server is configured
      and no Overseer server is necessary in an NVIDIA FLARE system. When DummyOverseerAgent is specified, the provisioning
      tool will include all arguments into the Overseer Agent section of generated json files.

Job Definition Manager
----------------------
The Job Definition Manager config specifies the Python object that manages the access and manipulation of Job Definition objects stored in the Job Storage.

The system reserved component id, job_manager, is used to denote the Job Definition Manager in the project.yml file.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server’s Startup Kit.

:class:`Job Definition Manager Spec<nvflare.apis.job_def_manager_spec.JobDefManagerSpec>`

NVIDIA FLARE provides a simple implementation that is based on scanning of job definition objects:

    - :class:`Simple Job Def Manager<nvflare.apis.impl.job_def_manager.SimpleJobDefManager>`

Job Storage
^^^^^^^^^^^
The Job definition is stored in a persistent store (used by Simple Job Def Manager). The Job Storage config specifies the Python object that manages the access to the store.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server’s Startup Kit.

.. note::

   The default storage is `FilesystemStorage<nvflare.app_common.storages.filesystem_storage.FilesystemStorage>` and is
   configured to use paths available in the file system to persist data. Other implementations can be used instead that
   may need to take other arguments or configurations.

Job Scheduler
-------------
The Job scheduler is responsible for determining the next job to run. Job scheduler config specifies the Job scheduler Python object.

The system reserved component id, job_scheduler, is used to denote the Job Scheduler in the project.yml file.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server’s Startup Kit.

:class:`Job Scheduler Spec<nvflare.apis.job_scheduler_spec.JobSchedulerSpec>`

NVIDIA FLARE provides a default implementation of the Job Scheduler that does resource based scheduling as described in the beginning:

    - :class:`Default Job Scheduler<nvflare.app_common.job_schedulers.job_scheduler.DefaultJobScheduler>`

Storage
-------
Storage is used in Job Storage and Job Execution State Storage. See the specific sections for more details.

:class:`Storage Spec<nvflare.apis.storage.StorageSpec>`

NVIDIA FLARE provides two simple storage implementations:

    - :class:`File System Storage<nvflare.app_common.storages.filesystem_storage.FilesystemStorage>`
    - :class:`AWS S3 Storage<nvflare.app_common.storages.s3_storage.S3Storage>`

Resource Manager
-----------------
The Resource Manager is responsible for managing job resources on FL Client. Resource Manager config specifies the Resource Manager Python object.

The system reserved component id, resource_manager, is used to denote the Resource Manager in the project.yml file.

This component is specified as one item in the components.client section.

This configuration is included in the fed_client.json of the FL Client’s Startup Kit.

:class:`Resource Manager Spec<nvflare.apis.resource_manager_spec.ResourceManagerSpec>`

NVIDIA FLARE provides a simple resource manager that manages resources as a list of items:

    - :class:`List Resource Manager<nvflare.app_common.resource_managers.list_resource_manager.ListResourceManager>`

Resource Consumer
-----------------
The Resource Consumer is responsible for consuming and/or initializing job resources on FL Client. The Resource Consumer
config specifies the Resource Consumer Python object.

This configuration is included in the fed_client.json of the FL Client’s Startup Kit.

The system reserved component id, resource_consumer, is used to denote the Resource Consumer in the project.yml file.

This component is specified as one item in the components.client section.

:class:`Resource Consumer Spec<nvflare.apis.resource_manager_spec.ResourceConsumerSpec>`

NVIDIA FLARE provides a GPU resource consumer:

    - :class:`GPU Resource Consumer<nvflare.app_common.resource_consumers.gpu_resource_consumer.GPUResourceConsumer>`

Snapshot Persisting
-------------------
The Job Execution State is persisted in snapshots with the Job Execution State Storage.

Job Execution State Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Job Execution State is stored in a persistent store. The Job Execution State Storage config specifies the Python
object that manages the access to the store.

This configuration is included in the fed_server.json of the Server’s Startup Kit.