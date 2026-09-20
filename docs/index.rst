.. _user_guide:

############
NVIDIA FLARE
############

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Overview

   welcome
   What's New <whats_new>
   roadmap
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
   :caption: Adapt & Build

   Agent Skills <user_guide/agent_skills/index>
   Choose an API Path <user_guide/data_scientist_guide/api_selection>
   Client API <user_guide/data_scientist_guide/client_api_usage>
   Job Recipe <user_guide/data_scientist_guide/job_recipe>
   Recipe API <user_guide/data_scientist_guide/recipe_api>
   Available Recipes <user_guide/data_scientist_guide/available_recipes>
   Collaboration API (Technical Preview) <user_guide/data_scientist_guide/collab_api>
   FLARE API <user_guide/data_scientist_guide/flare_api>
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
   Multi-Study Support <user_guide/admin_guide/multi_study_guide>
   Distributed Provisioning <user_guide/nvflare_cli/distributed_provisioning>
   user_guide/admin_guide/deployment/dashboard_ui
   user_guide/admin_guide/deployment/cloud_deployment
   Deploy Prepare <user_guide/nvflare_cli/deploy_command>
   Running FLARE in Docker <user_guide/admin_guide/deployment/containerized_deployment>
   Running FLARE in Kubernetes <user_guide/admin_guide/deployment/helm_chart>
   Running FLARE on Slurm <user_guide/admin_guide/deployment/slurm_job_launcher>
   Deploying FLARE on OpenShift <user_guide/admin_guide/deployment/openshift>
   Brev Scripted Deployment Quickstart <user_guide/admin_guide/deployment/brev_scripted_deployment>
   Brev Kubernetes Helm Deployment <user_guide/admin_guide/deployment/brev_deployment>
   Preflight Check <user_guide/nvflare_cli/preflight_check>
   user_guide/admin_guide/deployment/operation
   user_guide/admin_guide/monitoring
   user_guide/admin_guide/configurations/logging_configuration
   Live Log Streaming <programming_guide/live_log_streaming>
   Site Configuration Metadata <user_guide/admin_guide/configurations/site_config>
   System Configuration <user_guide/admin_guide/configurations/system_configuration>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Security & Compliance

   system_architecture/security_overview
   user_guide/admin_guide/security/terminologies_and_roles
   Identity & Access Control <user_guide/admin_guide/security/identity_security>
   Per-Job Certificates <user_guide/admin_guide/security/per_job_certificates>
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

`NVIDIA FLARE™ <https://developer.nvidia.com/flare>`_ (NVIDIA Federated
Learning Application Runtime Environment) is a domain-agnostic, open-source,
and extensible SDK for federated learning and other federated-computing
applications. It supports collaborative training and evaluation, analytics and
statistics, site-local data processing, and custom distributed workflows.

NVIDIA FLARE brings approved computation to participating sites. In a properly
designed and governed federation, raw datasets remain at their source while
each site controls which tasks may run and which results may leave. Its general
federated-computing core provides workflow orchestration, task execution,
communication, security, and lifecycle services from local development through
production deployment.

Choose Your Path
================

Run Your First Federation
-------------------------

Start here if you are new to NVIDIA FLARE or want to adapt an existing
training or data-processing workflow.

- :doc:`Quick Start <quickstart>` -- Run a complete two-client federation,
  understand its result, and choose the next path.
- :doc:`Agent Skills <user_guide/agent_skills/index>` -- Use maintained coding-agent
  workflows for supported training projects, federated statistics, optimization,
  and diagnosis.
- :ref:`Choose an API Path <api_selection>` -- Select the Client API, Recipe,
  Collaboration API (Technical Preview), or lower-level components for a manual
  or custom integration.
- :doc:`Installation <installation>` -- Review supported environments and
  alternative installation methods.
- :ref:`Examples and Tutorials <example_applications>` -- Continue with runnable
  applications, feature tutorials, and the example catalog.

Research and Design Custom Workflows
------------------------------------

Start here if you develop federated algorithms, study privacy or scale, or
reproduce published work.

- :ref:`Collaboration API (Technical Preview) <collab_api>` -- Build custom
  server-controlled or peer-to-peer algorithms with remote Python method calls.
- :ref:`Available Recipes <available_recipes>` -- Review maintained workflow
  builders and their extension points.
- :ref:`Examples and Tutorials <example_applications>` -- Find algorithm and
  workflow implementations.
- :doc:`Research Papers <user_guide/researcher_guide/index>` -- Browse published
  research and reference implementations.
- :ref:`Security Overview <flare_security_overview>` -- Select privacy,
  governance, authorization, audit, and confidential-computing controls.

Deployment & Security (Production Teams)
----------------------------------------

Start here if you are deploying FLARE in an organization or consortium.

- :doc:`Deployment Overview <user_guide/admin_guide/deployment/overview>` -- Provisioning, Docker/Kubernetes, cloud deployment, dashboard
- :doc:`Admin Commands <user_guide/admin_guide/deployment/operation>` -- Operating and managing a running FL system
- :doc:`System Configuration <user_guide/admin_guide/configurations/system_configuration>` -- Configuration files and settings
- :doc:`Preflight Check <user_guide/nvflare_cli/preflight_check>` -- Pre-launch validation
- :doc:`Security Overview <system_architecture/security_overview>` -- Authentication, authorization, privacy, auditing
- :doc:`Confidential Computing <user_guide/confidential_computing/index>` -- Hardware-backed TEEs for end-to-end IP protection

Developers (Advanced / Contributors)
-------------------------------------

Start here if you want to extend FLARE or build custom workflows.

- :ref:`Developer Guide <developer_guide>` -- Architecture deep-dives, controllers, filters, and extension points
- :doc:`API Reference <apidocs/modules>` -- Full Python API documentation
- :doc:`Contributing <contributing>` -- How to contribute to NVIDIA FLARE

Explore by Use Case
===================

- :ref:`Federated Statistics <federated_statistics>` -- Compute aggregate
  statistics while source records stay at participating sites.
- :ref:`Large Models and LLMs <llm_fine_tuning>` -- Fine-tune and evaluate
  large models across sites.
- :ref:`Edge and Mobile <mobile_training>` -- Build mobile and hierarchical
  federated-learning applications.
- :doc:`Industry Use Cases <industry_use_cases>` -- Explore applications in
  healthcare, life sciences, financial services, automotive, and other fields.
- :doc:`Welcome <welcome>` -- Review the concepts, capabilities, and execution
  environments behind these paths.
