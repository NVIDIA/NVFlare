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

.. toctree::
   :maxdepth: 1
   :hidden:

   programming_guide/system_architecture
   flare_system_architecure
   cellnet_architecture

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

.. toctree::
   :maxdepth: 1
   :hidden:

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

- :doc:`Workflows and Controllers <programming_guide/workflows_and_controllers>`
- :doc:`Model Controller <programming_guide/controllers/model_controller>`
- :doc:`Scatter and Gather <programming_guide/controllers/scatter_and_gather_workflow>`
- :doc:`Cyclic Workflow <programming_guide/controllers/cyclic_workflow>`
- :doc:`Client-Controlled Workflows <programming_guide/controllers/client_controlled_workflows>`
- :doc:`Cross-Site Model Evaluation <programming_guide/controllers/cross_site_model_evaluation>`
- :doc:`Initialize Global Weights <programming_guide/controllers/initialize_global_weights>`

.. toctree::
   :maxdepth: 1
   :hidden:

   programming_guide/workflows_and_controllers
   programming_guide/controllers/model_controller
   programming_guide/controllers/scatter_and_gather_workflow
   programming_guide/controllers/cyclic_workflow
   programming_guide/controllers/client_controlled_workflows
   programming_guide/controllers/cross_site_model_evaluation
   programming_guide/controllers/initialize_global_weights

Advanced Topics
===============

- :doc:`Filters <programming_guide/filters>`
- :doc:`Component Configuration <programming_guide/component_configuration>`
- :doc:`Resource Manager and Consumer <programming_guide/resource_manager_and_consumer>`
- :doc:`Global Model Initialization <programming_guide/global_model_initialization>`
- :doc:`Timeouts Reference <programming_guide/timeouts>`
- :doc:`Dashboard API <programming_guide/dashboard_api>`
- :doc:`Unsafe Component Detection <user_guide/admin_guide/security/unsafe_component_detection>`

.. toctree::
   :maxdepth: 1
   :hidden:

   programming_guide/filters
   programming_guide/component_configuration
   programming_guide/resource_manager_and_consumer
   programming_guide/global_model_initialization
   programming_guide/timeouts
   programming_guide/dashboard_api
   user_guide/admin_guide/security/unsafe_component_detection

Large Models & LLM
==================

Techniques for federated training and fine-tuning of large models, including LLMs.

**Deployment & Optimization:**

- :doc:`Notes on Large Models <user_guide/admin_guide/deployment/notes_on_large_models>` -- Deployment considerations for large model training
- :doc:`Message Quantization <programming_guide/message_quantization>` -- Reducing message size via quantization
- :doc:`File Streaming <programming_guide/file_streaming>` -- Streaming large files between participants
- :doc:`Tensor Downloader <programming_guide/tensor_downloader>` -- Efficient model parameter transfer
- :doc:`Memory Management <programming_guide/memory_management>` -- Controlling memory usage during training
- :doc:`Decomposer for Large Objects <programming_guide/decomposer_for_large_object>` -- Serializing large objects efficiently

**LLM Fine-Tuning:**

- `Federated SFT (Supervised Fine-Tuning) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/llm_hf>`_ -- Federated SFT with HuggingFace
- `Federated PEFT (Parameter-Efficient Fine-Tuning) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/llm_hf>`_ -- LoRA and other PEFT methods
- `NeMo SFT Integration <https://github.com/NVIDIA/NVFlare/tree/main/integration/nemo/examples/supervised_fine_tuning>`_ -- Federated SFT with NeMo
- `NeMo PEFT Integration <https://github.com/NVIDIA/NVFlare/tree/main/integration/nemo/examples/peft>`_ -- Federated PEFT with NeMo
- `NeMo Prompt Learning <https://github.com/NVIDIA/NVFlare/tree/main/integration/nemo/examples/prompt_learning>`_ -- Federated prompt tuning with NeMo

.. toctree::
   :maxdepth: 1
   :hidden:

   user_guide/admin_guide/deployment/notes_on_large_models
   programming_guide/message_quantization
   programming_guide/memory_management
   programming_guide/tensor_downloader
   programming_guide/file_streaming
   programming_guide/decomposer_for_large_object

Hierarchical Architecture
=========================

- :doc:`Hierarchical Architecture <programming_guide/hierarchical_architecture>`
- :doc:`Hierarchical Communication <programming_guide/hierarchical_communication>`

.. toctree::
   :maxdepth: 1
   :hidden:

   programming_guide/hierarchical_architecture
   programming_guide/hierarchical_communication

3rd-Party Integration
=====================

- :doc:`3rd-Party Integration <programming_guide/execution_api_type/3rd_party_integration>`

.. toctree::
   :maxdepth: 1
   :hidden:

   programming_guide/execution_api_type/3rd_party_integration

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

.. toctree::
   :maxdepth: 1
   :hidden:

   deprecated/FLAdminAPI
   programming_guide/execution_api_type/executor
   programming_guide/shareable
   programming_guide/data_exchange_object
   programming_guide/controllers/controllers
   programming_guide/execution_api_type

Troubleshooting
===============

- :doc:`Common Errors & Solutions <troubleshooting/common_errors>`
- :doc:`Timeout Troubleshooting <user_guide/timeout_troubleshooting>`
- :doc:`Debugging Guide <troubleshooting/debugging_guide>`
- :doc:`FAQ <faq>`

.. toctree::
   :maxdepth: 1
   :hidden:

   troubleshooting/common_errors
   user_guide/timeout_troubleshooting
   troubleshooting/debugging_guide
   faq
