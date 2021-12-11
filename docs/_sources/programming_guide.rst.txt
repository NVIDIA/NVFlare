.. _programming_guide:

#####################################################
Programming Guide - Developing Apps with NVIDIA FLARE
#####################################################

This guide contains details on the key concepts, objects, and information you should know to implement your own components.

NVIDIA FLARE is designed with a componentized architecture which allows users to bring customized components into the
framework.

If you want to use your components in an FL application, you will need to change the config in the application folder.
Please refer to :ref:`user_guide/application:NVIDIA FLARE Application` for more details.

.. toctree::
   :maxdepth: 1

   programming_guide/controllers
   programming_guide/executor
   programming_guide/event_system
   programming_guide/fl_context
   programming_guide/filters
   programming_guide/shareable
   programming_guide/data_exchange_object
   programming_guide/fl_component
   programming_guide/best_practices

Code Structure
==============

Different components can be built on top of the APIs(:mod:`nvflare.apis`) in NVIDIA FLARE core, and you can now
implement your own custom workflows. The concepts of aggregator, learnable, persistors, and shareable_generator which
were fixed in the workflow have been decoupled from the core of NVIDIA FLARE and moved to :mod:`nvflare.app_common`.
This is also the package containing the reference implementation of the :ref:`scatter_and_gather_workflow`, and all of
this can be used in your own workflow.

    - :mod:`nvflare.apis` - the generic class definitions
    - :mod:`nvflare.app_common` - higher level controllers, workflows, and algorithms
    - :mod:`nvflare.fuel` - supporting components of the provisioning and admin systems
    - :mod:`nvflare.lighter` - configuration, scripts, and Builders to support the provisioning tool
    - :mod:`nvflare.poc` - configurations for the poc tool
    - :mod:`nvflare.private` - low-level implementation of the platform and communication
    - :mod:`nvflare.security` - authorization policies
    - :mod:`nvflare.widgets` - widgets that extend base functionality in the aux communication channel
