#####################################################
Programming Guide - Developing Apps with NVIDIA FLARE
#####################################################

This guide introduces key objects and components you should know to implement your own components.

NVIDIA FLARE is designed with a componentized architecture which allows users to bring customized components into the
framework.

Different components can be built on top of the APIs(:mod:`nvflare.apis`) in NVIDIA FLARE core, and you can now
implement your own custom workflows. The concepts of aggregator, learnable, persistors, and shareable_generator which
were fixed in the workflow have been decoupled from the core of NVIDIA FLARE and moved to :mod:`nvflare.app_common`.
This is also the package containing the reference implementation of the :ref:`Scatter and Gather Workflow`, and all of
this can be used in your own workflow.

If you want to use your components in an FL application, you will need to change the config in the application folder.
Please refer to :ref:`user_guide/application:NVIDIA FLARE Application` for more details.

.. toctree::
   :maxdepth: 1

   programming_guide/fl_context
   programming_guide/fl_component
   programming_guide/controllers
   programming_guide/filters
   programming_guide/executor
   programming_guide/event_system
   programming_guide/shareable
   programming_guide/data_exchange_object
   programming_guide/best_practices
