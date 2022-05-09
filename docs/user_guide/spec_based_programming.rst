#################################################
Spec-based Programming for System Service Objects
#################################################
NVIDIA FLARE 2.1 needs additional services to implement the HA feature:
storage, overseer, job definition management, etc. There are many ways to implement such services. For example,
storage could be implemented with a file system, AWS S3, or some database technologies. Similarly, job definition
management could be done with simple file reading or a sophisticated solution with a database or search engine.

To allow any of these solutions in NVIDIA FLARE, we take a spec-based approach for such objects. Each such service will
provide an interface definition (the spec), and all implementations must follow the spec. The spec defines the
required behaviors for the implementation. At the same time, we provide some implementations that follow these specs.

Here are the spec definitions and provided implementations:

Overseer Agent
--------------
:class:`Overseer Agent Spec<nvflare.apis.overseer_spec.OverseerAgent>`

NVIDIA FLARE provides two implementations:

    - :class:`HttpOverseerAgent<nvflare.ha.overseer_agent.HttpOverseerAgent>` to work with the Overseer server
    - :class:`DummyOverseerAgent<nvflare.ha.dummy_overseer_agent.DummyOverseerAgent>` is a dummy agent that simply
      returns the configured endpoint as the hot FL server. The dummy agent is used when a single FL server is configured
      and no Overseer server is necessary in an NVIDIA FLARE system.

Job Definition Manager
----------------------
:class:`Job Definition Manager Spec<nvflare.apis.job_def_manager_spec.JobDefManagerSpec>`

NVIDIA FLARE provides a simple implementation that is based on scanning of job definition objects:

    - :class:`Simple Job Def Manager<nvflare.apis.impl.job_def_manager.SimpleJobDefManager>`


Job Scheduler
-------------
:class:`Job Scheduler Spec<nvflare.apis.job_scheduler_spec.JobSchedulerSpec>`

NVIDIA FLARE provides a default implementation of the Job Scheduler that does resource based scheduling as described in the beginning:

    - :class:`Default Job Scheduler<nvflare.app_common.job_schedulers.job_scheduler.DefaultJobScheduler>`

Storage
-------
:class:`Storage Spec<nvflare.apis.storage.StorageSpec>`

NVIDIA FLARE provides two simple storage implementations:

    - :class:`File System Storage<nvflare.app_common.storages.filesystem_storage.FilesystemStorage>`
    - :class:`AWS S3 Storage<nvflare.app_common.storages.s3_storage.S3Storage>`

Study Manager
-------------
:class:`Study Manager Spec<nvflare.apis.study_manager_spec.StudyManagerSpec>`

NVIDIA FLARE provides a simple implementation:

    - :class:`Study Manager<nvflare.apis.impl.study_manager.StudyManager>`

Resource Manager and Consumer
-----------------------------
:class:`Resource Manager Spec<nvflare.apis.resource_manager_spec.ResourceManagerSpec>` and :class:`Resource Consumer Spec<nvflare.apis.resource_manager_spec.ResourceConsumerSpec>`

NVIDIA FLARE provides a simple resource manager that manages resources as a list of items:

    - :class:`List Resource Manager<nvflare.app_common.resource_managers.list_resource_manager.ListResourceManager>`

NVIDIA FLARE provides a GPU resource consumer:

    - :class:`GPU Resource Consumer<nvflare.app_common.resource_consumers.gpu_resource_consumer.GPUResourceConsumer>`
