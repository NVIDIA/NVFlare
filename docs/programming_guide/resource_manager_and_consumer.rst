.. _resource_manager_and_consumer:

#######################################
Resource Manager and Resource Consumer
#######################################
NVFlare introduced the concept of :ref:`job`, resource manager and resource consumer in version 2.1.

Each job has a meta.json that can specify "deploy_map", "min_clients", "mandatory_clients" and "resource_spec".

A user can specify the job resource requirement in meta.json and configure the corresponding resource manager and consumer.


During :ref:`job scheduling <job_scheduler_configuration>`, the server side will ask each client if the resource requirement can be satisfied. Each client will call the check_resources
method with their configured ResourceManager. The "resource_spec" specified in the job config will be passed in as an argument. Check resources should
figure out if the local client site resources are enough to run this job. If it can, then this job will be scheduled; Otherwise, the job will stay in the queue.

Note that this check is solely done by the ResourceManager, so if the resource manager tells the FL server that the resource is enough, NVFlare will
assume that it is OK to start the job on that site.

If outside of NVFlare another process occupied the resources and the resources became unavailable that might lead to the failure of job execution at runtime.

How to Specify the Job Resource Requirement
===========================================
With the job concept, users can specify how many resources this job requires at runtime.

The resource spec is a dict that maps site name to require resources, for example:

.. code-block::

    {
        "resource_spec": {
            "site-1": { "num_of_gpus": 1, "mem_per_gpu_in_GiB": 1 },
            "site-2": { "num_of_gpus": 1, "mem_per_gpu_in_GiB": 1 }
        }
    }

The full meta.json will then look like:

.. code-block::

    {
        "name": "hello-pt",
        "resource_spec": {
            "site-1": { "num_of_gpus": 1, "mem_per_gpu_in_GiB": 1 },
            "site-2": { "num_of_gpus": 1, "mem_per_gpu_in_GiB": 1 }
        },
        "min_clients" : 2,
        "deploy_map": {
            "app": [
                "@ALL"
            ]
        }
    }

How to Configure Resource Manager and Consumer
==============================================

Each site can configure its own resource manager and consumer using the "resources.json" inside the "local" folder.

For example, the default in POC looks like:

.. code-block::

    {
        "format_version": 2,
        "client": {
            "retry_timeout": 30,
            "compression": "Gzip"
        },
        "components": [
            {
                "id": "resource_manager",
                "path": "nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager",
                "args": { "num_of_gpus": 1, "mem_per_gpu_in_GiB": 4 }
            },
            {
                "id": "resource_consumer",
                "path": "nvflare.app_common.resource_consumers.gpu_resource_consumer.GPUResourceConsumer",
                "args": {}
            }
        ]
    }

This means they specify this site has 1 GPU and memory per GPU is 1 GiB.
If you do not have any GPU you can set the num_of_gpus and mem_per_gpu_in_GiB to 0.

GPUResourceManager (:mod:`nvflare.app_common.resource_managers.gpu_resource_manager`) and GPUResourceConsumer (:mod:`nvflare.app_common.resource_consumers.gpu_resource_consumer`)

.. note::

    Make sure each client has the same resource manager and resource consumer class, even though the arguments can be different.

GPUResourceManager and GPUResourceConsumer
==========================================

With the default ``ignore_host=False``, GPUResourceManager validates the configured GPU count and memory
against the host using nvidia-smi at initialization. When ``CUDA_VISIBLE_DEVICES`` contains GPU IDs,
the startup check is restricted to those GPUs. With ``ignore_host=True``, it skips host discovery and
trusts the configured capacity. Exclusive host-independent pools can use a nonempty numeric
``CUDA_VISIBLE_DEVICES`` mask to select GPU IDs.

The manager tracks reservations against that configured capacity; it does not continuously measure
host GPU usage. Process mode's ``GPUResourceConsumer`` separately checks host GPUs and free memory
at consumption time, then sets ``CUDA_VISIBLE_DEVICES`` for the subprocess.

Reservations coordinate only jobs tracked by this client's resource manager. Other processes,
containers, or independent clients can still use the same GPUs. Exclusive allocation does not
provide a host-wide GPU lock or enforce a physical memory quota. Operators must prevent overlapping
use or rely on an external scheduler. See the :ref:`Docker GPU policies <docker_gpu_assignment>`
for the no-op consumer, disjoint client pools, and parent-restart requirements.


How to Write Your Own Resource Manager and Consumer
===================================================

You can easily write your own resource manager and consumer following the API specification:

.. code-block:: python

    class ResourceConsumerSpec(ABC):
        @abstractmethod
        def consume(self, resources: dict):
            pass


    class ResourceManagerSpec(ABC):
        @abstractmethod
        def check_resources(self, resource_requirement: dict, fl_ctx: FLContext) -> Tuple[bool, str]:
            """Checks whether the specified resource requirement can be satisfied.
            Args:
                resource_requirement: a dict that specifies resource requirement
                fl_ctx: the FLContext
            Returns:
                A tuple of (check_result, token).
                check_result is a bool indicates whether there is enough resources;
                token is for resource reservation / cancellation for this check request.
            """
            pass

        @abstractmethod
        def cancel_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext):
            ""Cancels reserved resources if any.
            Args:
                resource_requirement: a dict that specifies resource requirement
                token: a resource reservation token returned by check_resources
                fl_ctx: the FLContext
            Note:
                If check_resource didn't return a token, then don't need to call this method
            """
            pass

        @abstractmethod
        def allocate_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext) -> dict:
            """Allocates resources.
            Note:
                resource requirements and resources may be different things.
            Args:
                resource_requirement: a dict that specifies resource requirement
                token: a resource reservation token returned by check_resources
                fl_ctx: the FLContext
            Returns:
                A dict of allocated resources
            """
            pass

        @abstractmethod
        def free_resources(self, resources: dict, token: str, fl_ctx: FLContext):
            """Frees resources.
            Args:
                resources: resources to be freed
                token: a resource reservation token returned by check_resources
                fl_ctx: the FLContext
            """
            pass

        @abstractmethod
        def report_resources(self, fl_ctx) -> dict:
            """Reports resources."""
            pass


A more friendly interface (AutoCleanResourceManager) is provided as well:

.. code-block:: python

    class AutoCleanResourceManager(ResourceManagerSpec, FLComponent, ABC):

        @abstractmethod
        def _deallocate(self, resources: dict):
            """Deallocates the resources.
            Args:
                resources (dict): the resources to be freed.
            """
            raise NotImplementedError

        @abstractmethod
        def _check_required_resource_available(self, resource_requirement: dict) -> bool:
            """Checks if resources are available.
            Args:
                resource_requirement (dict): the resource requested.
            Return:
                A boolean to indicate whether the current resources are enough for the required resources.
            """
            raise NotImplementedError

        @abstractmethod
        def _reserve_resource(self, resource_requirement: dict) -> dict:
            """Reserves resources given the requirements.
            Args:
                resource_requirement (dict): the resource requested.
            Return:
                A dict of reserved resources associated with the requested resource.
            """
            raise NotImplementedError

        @abstractmethod
        def _resource_to_dict(self) -> dict:
            raise NotImplementedError
