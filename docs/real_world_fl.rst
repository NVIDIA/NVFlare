.. _real_world_fl:

#############################
Real-World Federated Learning
#############################

This section shows how to use NVIDIA FLARE to deploy and operate an FL system.
A reference application will be used here to show provisioning and basic operation
of the system through the FLARE console and FLARE API. To deploy on the cloud, see :ref:`cloud_deployment`.

For instructions on how to set up the :ref:`nvflare_dashboard_ui` added in 2.2.1 to
help gather information to provision a project and distribute startup kits, see :ref:`dashboard_api`.

For more details on what you can do with apps with custom components and
the flexibility that the Controller and Worker APIs bring, see the :ref:`programming_guide`.

You can also see some `example applications <https://github.com/NVIDIA/NVFlare/tree/2.3/examples>`_ integrating with
`Clara Train <https://docs.nvidia.com/clara/clara-train-sdk/>`_ and
`MONAI <https://github.com/Project-MONAI/tutorials/tree/master/federated_learning/nvflare>`_
to see the capabilities of the system and how it can be operated.

.. toctree::
   :maxdepth: 1

   real_world_fl/overview
   real_world_fl/operation
   real_world_fl/application
   real_world_fl/job
   real_world_fl/workspace
   real_world_fl/cloud_deployment
   user_guide/federated_authorization
