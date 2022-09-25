.. _dynamic_provisioning:

******************************
Dynamic Provisioning
******************************

In version 2.2, two additional options, ``--add_user`` and ``--add_client``, were added to the ``nvflare provision`` command to allow for adding to an existing project.
Both of these commands take a yaml file to define the additional participant to provision. Type ``nvflare provision -h`` to see the details for the ``nvflare provision``
command.

Sample user.yaml for ``--add_user``:

.. code-block:: yaml

    name: new_user@nvidia.com
    org: nvidia
    roles:
    - super
 
Sample client.yaml for ``--add_client``:

.. code-block:: yaml

    name: new-site
    org: nvidia
    enable_byoc: true
    components:
    resource_manager:    # This id is reserved by system.  Do not change it.
    path: nvflare.app_common.resource_managers.list_resource_manager.ListResourceManager
    args:
        resources:
        gpu: [0, 1, 2, 3]
    resource_consumer:    # This id is reserved by system.  Do not change it.
    path: nvflare.app_common.resource_consumers.gpu_resource_consumer.GPUResourceConsumer
    args:
        gpu_resource_key: gpu
 

After running ``provision`` with ``--add_user`` or ``--add_client``, the new user or client will be included in the prod_NN folder.

To permanently include users or clients, please update the project.yml.

This feature is still a work in progress as information to servers and the overseer are not working in the federated way yet.
