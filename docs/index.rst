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

NVIDIA FLARE (Federated Learning Application Runtime Environment) is an open-source SDK for
federated learning. It helps ML practitioners adapt existing training workflows (PyTorch,
TensorFlow, XGBoost, scikit-learn, NeMo) to a federated setting with minimal code changes,
and enables platform teams to deploy secure, privacy-preserving multi-party collaboration.

Choose Your Path
================

New to FLARE (ML Practitioners)
-------------------------------

Start here if you want to federate an existing training script.

- :doc:`Welcome <welcome>` -- What FLARE is and what it supports
- :doc:`Installation <installation>` -- Install FLARE and set up your environment
- :doc:`Quick Start <quickstart>` -- Run a Hello World example and convert your ML code
- :ref:`Client API <client_api>` -- Recommended high-level API for federated training
- :ref:`Job Recipe API <job_recipe>` -- Pre-built recipes for common FL workflows
- :doc:`Migration Guide <migration_guide>` -- Upgrade between FLARE versions
- :ref:`Examples & Tutorials <example_applications>` -- End-to-end examples and tutorials

Deployment & Security (Production Teams)
----------------------------------------

Start here if you are deploying FLARE in an organization or consortium.

- :doc:`Deployment Overview <user_guide/admin_guide/deployment/overview>` -- Provisioning, Docker/Kubernetes, cloud deployment, dashboard
- :doc:`Admin Commands <user_guide/admin_guide/deployment/operation>` -- Operating and managing a running FL system
- :doc:`System Configuration <user_guide/admin_guide/configurations/system_configuration>` -- Configuration files and settings
- :doc:`Preflight Check <user_guide/nvflare_cli/preflight_check>` -- Pre-launch validation
- :doc:`Security Overview <flare_security_overview>` -- Authentication, authorization, privacy, auditing
- :doc:`Confidential Computing <user_guide/confidential_computing/index>` -- Hardware-backed TEEs for end-to-end IP protection

Developers (Advanced / Contributors)
-------------------------------------

Start here if you want to extend FLARE or build custom workflows.

- :ref:`Developer Guide <developer_guide>` -- Architecture deep-dives, controllers, filters, and extension points
- :doc:`API Reference <apidocs/modules>` -- Full Python API documentation
- :doc:`Contributing <contributing>` -- How to contribute to NVIDIA FLARE

Explore by Use Case
===================

- :doc:`Industry Use Cases <industry_use_cases>` -- Real-world deployments across healthcare, finance, government, and more
- :ref:`Large Models & LLM <llm_fine_tuning>` -- Federated fine-tuning, memory management, and optimization for large models
- :ref:`Edge & Mobile <mobile_training>` -- Mobile training (iOS/Android) and hierarchical FL for large-scale deployments
