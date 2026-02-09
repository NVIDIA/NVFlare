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
   user_guide/data_scientist_guide/data_preparation
   CLI Tools <user_guide/nvflare_cli/nvflare_cli>

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
   production_readiness
   user_guide/admin_guide/deployment/operation
   user_guide/admin_guide/monitoring
   user_guide/admin_guide/configurations/logging_configuration
   System Configuration <user_guide/admin_guide/configurations/system_configuration>
   operations/performance_tuning
   operations/upgrade_guide

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
   user_guide/admin_guide/security/auditing
   Confidential Computing <user_guide/confidential_computing/index>
   user_guide/admin_guide/security/regulatory_guidance
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

NVIDIA FLARE (NVIDIA Federated Learning Application Runtime Environment) is a domain-agnostic, open-source, extensible SDK that allows
researchers and data scientists to adapt existing ML/DL workflows (PyTorch, RAPIDS, Nemo, TensorFlow) to a federated paradigm; and enables
platform developers to build a secure, privacy preserving offering for a distributed multi-party collaboration.

Getting Started
===============
To get started with NVIDIA FLARE:

1. Read the :ref:`Welcome <welcome>` page for a quick overview of FLARE and its capabilities
2. Follow the :ref:`installation` guide to set up your environment
3. Run through the :ref:`quickstart` guide with a Hello World example
4. Browse :ref:`Industry Use Cases <industry_use_cases>` relevant to your domain
5. Explore more in the **Examples & Tutorials** section

For New Users
=============
If you are new to FLARE, we recommend starting with the :ref:`Client API <client_api>` and :ref:`Job Recipe API <job_recipe>` --
these higher-level APIs let you convert existing ML training code to federated with minimal changes.
If you have existing centralized ML code, see the :ref:`Migration Guide <migration_guide>`.

For Deployment & Security
=========================
When you are ready for production, the **Deployment & Operations** section covers provisioning, deployment, admin commands,
monitoring, and system configuration.
Review the :ref:`Production Readiness Checklist <production_readiness>` before going live.
The **Security & Compliance** section covers authentication, authorization, data privacy, and regulatory guidance.

For Developers
==============
For deep dives into FLARE architecture, custom workflows, and low-level APIs, see the
:ref:`Developer Guide <developer_guide>`, or browse the :doc:`API Reference <apidocs/modules>`.
