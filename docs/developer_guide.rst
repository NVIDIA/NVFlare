.. _developer_guide:
.. _programming_guide:

##############################
Architecture & Developer Guide
##############################

This guide is for developers who need to understand FLARE internals, build custom workflows,
or extend the platform. For higher-level usage, see the :ref:`User Guide <user_guide>`.

System Architecture
===================

- :doc:`System Architecture Overview <programming_guide/system_architecture>`
- :doc:`FLARE System Architecture <flare_system_architecure>`
- :doc:`CellNet Architecture <cellnet_architecture>`

Core Concepts
=============

- :doc:`Job <user_guide/core_concepts/job>`
- :doc:`Workspace <user_guide/core_concepts/workspace>`
- :doc:`Application <user_guide/core_concepts/application>`
- :doc:`FLModel <programming_guide/fl_model>`
- :doc:`FLContext <programming_guide/fl_context>`
- :doc:`FLComponent <programming_guide/fl_component>`
- :doc:`Event System <programming_guide/event_system>`
- :doc:`FedJob API <programming_guide/fed_job_api>`
- :doc:`FL Simulator <user_guide/nvflare_cli/fl_simulator>`
- :doc:`POC <user_guide/data_scientist_guide/poc>`

Workflows & Controllers
=======================

- :doc:`Workflows and Controllers <programming_guide/workflows_and_controllers>`
- :doc:`Model Controller <programming_guide/controllers/model_controller>`
- :doc:`Scatter and Gather <programming_guide/controllers/scatter_and_gather_workflow>`
- :doc:`Cyclic Workflow <programming_guide/controllers/cyclic_workflow>`
- :doc:`Client-Controlled Workflows <programming_guide/controllers/client_controlled_workflows>`
- :doc:`Cross-Site Model Evaluation <programming_guide/controllers/cross_site_model_evaluation>`
- :doc:`Initialize Global Weights <programming_guide/controllers/initialize_global_weights>`

Advanced Topics
===============

- :doc:`Filters <programming_guide/filters>`
- :doc:`Component Configuration <programming_guide/component_configuration>`
- :doc:`Resource Manager and Consumer <programming_guide/resource_manager_and_consumer>`
- :doc:`Global Model Initialization <programming_guide/global_model_initialization>`
- :doc:`Timeouts <programming_guide/timeouts>`
- :doc:`Dashboard API <programming_guide/dashboard_api>`

Hierarchical Architecture
=========================

- :doc:`Hierarchical Architecture <programming_guide/hierarchical_architecture>`
- :doc:`Hierarchical Communication <programming_guide/hierarchical_communication>`

3rd-Party Integration
=====================

- :doc:`3rd-Party Integration <programming_guide/execution_api_type/3rd_party_integration>`

Low-Level APIs
==============

These are foundational APIs that higher-level abstractions (Client API, FLARE API) are built on top of.
Most users do not need these directly, but they are available for advanced customization.

- :doc:`FLAdminAPI <deprecated/FLAdminAPI>`
- :doc:`Executor <programming_guide/execution_api_type/executor>`
- :doc:`Shareable <programming_guide/shareable>`
- :doc:`Data Exchange Object <programming_guide/data_exchange_object>`
- :doc:`Controllers <programming_guide/controllers/controllers>`
- :doc:`Execution API Types <programming_guide/execution_api_type>`
