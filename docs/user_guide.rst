###############################################
User Guide - Provision and Operate an FL System
###############################################

This user guide shows how to use NVIDIA FLARE to deploy and operate an FL system on multiple sites with a
Provision-Start-Operating procedure. A reference application will be used here to show provisioning and basic operation
of the system through the admin client. You will find information here about the Open Provision API for setting up the
system with all the available components (including setup for High Availability and Jobs) and the Admin API for
operating FL. For more details on what you can do with apps with custom components and
the flexibility that the Controller and Worker APIs bring, see the :ref:`programming_guide`.

You can also see some `example applications <https://github.com/NVIDIA/NVFlare/tree/main/examples>`_ integrating with
`Clara Train <https://docs.nvidia.com/clara/clara-train-sdk/>`_ and
`MONAI <https://github.com/Project-MONAI/tutorials/tree/master/federated_learning/nvflare>`_
to see the capabilities of the system and how it can be operated.

.. toctree::
   :maxdepth: 1

   user_guide/overview
   user_guide/application
   user_guide/job
   user_guide/workspace
   user_guide/provisioning_tool
   user_guide/system_architecture
   user_guide/high_availability
   user_guide/authorization
   user_guide/admin_commands
