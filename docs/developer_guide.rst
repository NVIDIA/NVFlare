.. _developer_guide:
.. _programming_guide:

##############################
Architecture & Developer Guide
##############################

This guide is for developers who need to understand FLARE internals, build custom workflows,
or extend the platform. For higher-level usage, see the :ref:`User Guide <user_guide>`.

System Architecture
===================

.. toctree::
   :maxdepth: 1

   programming_guide/system_architecture
   flare_system_architecure
   cellnet_architecture

Core Concepts
=============

.. toctree::
   :maxdepth: 1

   user_guide/core_concepts/job
   user_guide/core_concepts/workspace
   user_guide/core_concepts/application
   programming_guide/fl_model
   programming_guide/fl_context
   programming_guide/fl_component
   programming_guide/event_system
   programming_guide/fed_job_api
   user_guide/nvflare_cli/fl_simulator
   user_guide/data_scientist_guide/poc

Workflows & Controllers
=======================

.. toctree::
   :maxdepth: 1

   programming_guide/workflows_and_controllers
   programming_guide/controllers/model_controller
   programming_guide/controllers/scatter_and_gather_workflow
   programming_guide/controllers/cyclic_workflow
   programming_guide/controllers/client_controlled_workflows
   programming_guide/controllers/cross_site_model_evaluation
   programming_guide/controllers/initialize_global_weights

Advanced Topics
===============

.. toctree::
   :maxdepth: 1

   programming_guide/filters
   programming_guide/component_configuration
   programming_guide/resource_manager_and_consumer
   programming_guide/global_model_initialization
   programming_guide/timeouts
   programming_guide/dashboard_api

Hierarchical Architecture
=========================

.. toctree::
   :maxdepth: 1

   programming_guide/hierarchical_architecture
   programming_guide/hierarchical_communication

3rd-Party Integration
=====================

.. toctree::
   :maxdepth: 1

   programming_guide/execution_api_type/3rd_party_integration

Low-Level APIs
==============

These are foundational APIs that higher-level abstractions (Client API, FLARE API) are built on top of.
Most users do not need these directly, but they are available for advanced customization.

.. toctree::
   :maxdepth: 1

   deprecated/FLAdminAPI
   programming_guide/execution_api_type/executor
   programming_guide/shareable
   programming_guide/data_exchange_object
   programming_guide/controllers/controllers
   programming_guide/execution_api_type
