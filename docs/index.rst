.. _user_guide:

############
NVIDIA FLARE
############

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Welcome

   welcome
   release_notes/flare_272
   release_notes/previous

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started

   installation
   quickstart
   run_mode

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   user_guide/data_scientist_guide/client_api_usage
   user_guide/data_scientist_guide/job_recipe
   user_guide/data_scientist_guide/available_recipes
   user_guide/data_scientist_guide/flare_api
   programming_guide/experiment_tracking
   user_guide/data_scientist_guide/federated_xgboost/federated_xgboost
   user_guide/data_scientist_guide/flower_integration/flower_integration
   CLI Tools <user_guide/nvflare_cli/nvflare_cli>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Deployment

   user_guide/admin_guide/deployment/overview
   programming_guide/provisioning_system
   user_guide/admin_guide/deployment/dashboard_ui
   user_guide/admin_guide/deployment/cloud_deployment
   user_guide/admin_guide/deployment/aws_eks
   user_guide/admin_guide/deployment/containerized_deployment
   user_guide/admin_guide/deployment/helm_chart

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Operations & Configuration

   user_guide/admin_guide/deployment/operation
   user_guide/admin_guide/monitoring
   user_guide/admin_guide/configurations/logging_configuration
   user_guide/admin_guide/configurations/configurations
   user_guide/admin_guide/configurations/communication_configuration
   user_guide/admin_guide/configurations/variable_resolution
   user_guide/admin_guide/configurations/server_port_consolidation

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Security

   flare_security_overview
   user_guide/admin_guide/security/terminologies_and_roles
   Identity & Access Control <user_guide/admin_guide/security/identity_security>
   user_guide/admin_guide/security/site_policy_management
   Network & Communication <user_guide/admin_guide/security/communication_security>
   Data Privacy & Filters <user_guide/admin_guide/security/data_privacy_protection>
   user_guide/admin_guide/security/auditing
   user_guide/admin_guide/security/unsafe_component_detection
   security_faq

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Confidential Computing

   Overview <user_guide/confidential_computing/index>
   user_guide/confidential_computing/on_premises/index
   user_guide/confidential_computing/on_premises/cc_architecture
   user_guide/confidential_computing/on_premises/cc_deployment_guide
   user_guide/confidential_computing/on_premises/attestation
   user_guide/confidential_computing/on_premises/hashicorp_vault_trustee_kbs_deployment
   user_guide/confidential_computing/azure/index
   user_guide/confidential_computing/azure/azure_confidential_virtual_machine_deployment
   user_guide/confidential_computing/azure/confidential_azure_container_instances_deployment

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Edge Development

   Edge Overview <user_guide/edge_development/index>
   user_guide/edge_development/flare_mobile
   user_guide/edge_development/mobile_android

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Large Models & LLM

   user_guide/admin_guide/deployment/notes_on_large_models
   programming_guide/message_quantization
   programming_guide/memory_management
   programming_guide/tensor_downloader
   programming_guide/file_streaming
   programming_guide/decomposer_for_large_object

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Architecture & Developer Guide

   Overview <developer_guide>
   programming_guide/system_architecture
   flare_system_architecure
   cellnet_architecture
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
   programming_guide/workflows_and_controllers
   programming_guide/controllers/model_controller
   programming_guide/controllers/scatter_and_gather_workflow
   programming_guide/controllers/cyclic_workflow
   programming_guide/controllers/client_controlled_workflows
   programming_guide/controllers/cross_site_model_evaluation
   programming_guide/controllers/initialize_global_weights
   programming_guide/filters
   programming_guide/component_configuration
   programming_guide/resource_manager_and_consumer
   programming_guide/global_model_initialization
   programming_guide/timeouts
   programming_guide/dashboard_api
   programming_guide/hierarchical_architecture
   programming_guide/hierarchical_communication
   programming_guide/execution_api_type/3rd_party_integration
   deprecated/FLAdminAPI
   programming_guide/execution_api_type/executor
   programming_guide/shareable
   programming_guide/data_exchange_object
   programming_guide/controllers/controllers
   programming_guide/execution_api_type

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Troubleshooting

   user_guide/timeout_troubleshooting
   user_guide/nvflare_cli/preflight_check
   faq

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples & Tutorials

   example_applications_algorithms
   tutorials
   self-paced-training/index
   user_guide/researcher_guide/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   API Reference <apidocs/modules>
   glossary
   release_notes/extra_270
   publications_and_talks
   contributing

NVIDIA FLARE (NVIDIA Federated Learning Application Runtime Environment) is a domain-agnostic, open-source, extensible SDK that allows
researchers and data scientists to adapt existing ML/DL workflows (PyTorch, RAPIDS, Nemo, TensorFlow) to a federated paradigm; and enables
platform developers to build a secure, privacy preserving offering for a distributed multi-party collaboration.

Getting Started
===============
To get started with NVIDIA FLARE:

1. Read the :ref:`Welcome <welcome>` page for a quick overview of FLARE and its capabilities
2. Follow the :ref:`installation` guide to set up your environment
3. Run through the :ref:`quickstart` guide to try your first example
4. Explore more examples in the :ref:`Examples & Tutorials <examples_tutorials>` section

For New Users
=============
If you are new to FLARE, we recommend starting with the :ref:`Client API <client_api>` and :ref:`Job Recipe API <job_recipe>` --
these higher-level APIs let you convert existing ML training code to federated with minimal changes.

For Deployment & Security
=========================
When you are ready for production, the **Deployment** section covers provisioning and deployment options,
and the **Operations & Configuration** section covers admin commands, monitoring, and system configuration.
The :ref:`Security <security>` section covers authentication, authorization, data privacy, and auditing.

For Developers
==============
For deep dives into FLARE architecture, custom workflows, and low-level APIs, see the
:ref:`Architecture & Developer Guide <developer_guide>` and the :doc:`API Reference <apidocs/modules>`.
