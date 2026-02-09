.. _user_guide:

############
NVIDIA FLARE
############

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Overview

   welcome
   release_notes/flare_272
   industry_use_cases

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started

   installation
   quickstart
   migration_guide

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   user_guide/data_scientist_guide/client_api_usage
   user_guide/data_scientist_guide/job_recipe
   user_guide/data_scientist_guide/available_recipes
   user_guide/data_scientist_guide/flare_api
   user_guide/data_scientist_guide/flower_integration/flower_integration
   programming_guide/experiment_tracking
   Federated XGBoost <user_guide/data_scientist_guide/federated_xgboost/federated_xgboost>
   user_guide/data_scientist_guide/data_preparation
   CLI Tools <user_guide/nvflare_cli/nvflare_cli>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples & Tutorials

   example_applications_algorithms
   tutorials
   self-paced-training/index
   Research Papers <user_guide/researcher_guide/index>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Large Models & LLM

   Federated LLM Fine-Tuning <programming_guide/llm_fine_tuning>
   programming_guide/message_quantization
   programming_guide/memory_management
   programming_guide/tensor_downloader
   programming_guide/file_streaming
   programming_guide/decomposer_for_large_object

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Edge & Mobile

   Mobile Training (iOS / Android) <user_guide/edge_development/mobile_training>
   Mobile SDK Reference <user_guide/edge_development/flare_mobile>
   Hierarchical FL <programming_guide/hierarchical_architecture>
   programming_guide/hierarchical_communication

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Deployment & Operations

   user_guide/admin_guide/deployment/overview
   programming_guide/provisioning_system
   user_guide/admin_guide/deployment/dashboard_ui
   user_guide/admin_guide/deployment/cloud_deployment
   user_guide/admin_guide/deployment/aws_eks
   Running FLARE in Docker <user_guide/admin_guide/deployment/containerized_deployment>
   Running FLARE in Kubernetes <user_guide/admin_guide/deployment/helm_chart>
   Preflight Check <user_guide/nvflare_cli/preflight_check>
   user_guide/admin_guide/deployment/operation
   user_guide/admin_guide/monitoring
   user_guide/admin_guide/configurations/logging_configuration
   System Configuration <user_guide/admin_guide/configurations/system_configuration>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Security & Compliance

   flare_security_overview
   user_guide/admin_guide/security/terminologies_and_roles
   Identity & Access Control <user_guide/admin_guide/security/identity_security>
   user_guide/admin_guide/security/site_policy_management
   Network & Communication <user_guide/admin_guide/security/communication_security>
   Data Privacy & Filters <user_guide/admin_guide/security/data_privacy_protection>
   Differential Privacy <user_guide/admin_guide/security/differential_privacy>
   user_guide/admin_guide/security/auditing
   Confidential Computing <user_guide/confidential_computing/index>
   security_faq

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Developer Guide

   developer_guide

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   API Reference <apidocs/modules>
   glossary
   publications_and_talks
   release_notes/previous
   contributing

NVIDIA FLARE (Federated Learning Application Runtime Environment) is an open-source Python SDK
for federated learning. Add a few lines to your existing training code and run it across distributed
sites -- from 2 hospitals to millions of edge devices.

**New here?** Start with :ref:`Installation <installation>`, then :ref:`Quick Start <quickstart>` to run your first FL job.

**Already using FLARE?** See the :ref:`User Guide <client_api>` for APIs, or :ref:`Examples & Tutorials <example_applications>` for code.

**Ready for production?** The **Deployment & Operations** section covers provisioning and infrastructure.
**Security & Compliance** covers authentication, privacy, and confidential computing.

**Building custom workflows?** See the :ref:`Developer Guide <developer_guide>` or the :doc:`API Reference <apidocs/modules>`.
