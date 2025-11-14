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
NVIDIA FLARE needs additional services for storage, job definition management, etc. There are many ways to implement such services. For example,
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

Job Definition Manager
----------------------
The Job Definition Manager config specifies the Python object that manages the access and manipulation of Job Definition objects stored in the Job Storage.

The system reserved component id, job_manager, is used to denote the Job Definition Manager in the project.yml file.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server's Startup Kit.

:class:`Job Definition Manager Spec<nvflare.apis.job_def_manager_spec.JobDefManagerSpec>`

NVIDIA FLARE provides a simple implementation that is based on scanning of job definition objects:

    - :class:`Simple Job Def Manager<nvflare.apis.impl.job_def_manager.SimpleJobDefManager>`

Job Storage
^^^^^^^^^^^
The Job definition is stored in a persistent store (used by Simple Job Def Manager). The Job Storage config specifies the Python object that manages the access to the store.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server's Startup Kit.

.. note::

   The default storage is `FilesystemStorage<nvflare.app_common.storages.filesystem_storage.FilesystemStorage>` and is
   configured to use paths available in the file system to persist data. Other implementations can be used instead that
   may need to take other arguments or configurations.

Job Scheduler
-------------
The Job scheduler is responsible for determining the next job to run. Job scheduler config specifies the Job scheduler Python object.

The system reserved component id, job_scheduler, is used to denote the Job Scheduler in the project.yml file.

This component is specified as one item in the components.server section.

This configuration is included in the fed_server.json of the Server's Startup Kit.

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

This configuration is included in the fed_client.json of the FL Client's Startup Kit.

:class:`Resource Manager Spec<nvflare.apis.resource_manager_spec.ResourceManagerSpec>`

NVIDIA FLARE provides a simple resource manager that manages resources as a list of items:

    - :class:`List Resource Manager<nvflare.app_common.resource_managers.list_resource_manager.ListResourceManager>`

Resource Consumer
-----------------
The Resource Consumer is responsible for consuming and/or initializing job resources on FL Client. The Resource Consumer
config specifies the Resource Consumer Python object.

This configuration is included in the fed_client.json of the FL Client's Startup Kit.

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

This configuration is included in the fed_server.json of the Server's Startup Kit.