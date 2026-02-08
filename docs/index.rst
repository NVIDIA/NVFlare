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
   user_guide/data_scientist_guide/available_recipes
   run_mode

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   user_guide/data_scientist_guide/client_api_usage
   user_guide/data_scientist_guide/job_recipe
   programming_guide/fed_job_api
   user_guide/data_scientist_guide/flare_api
   programming_guide/experiment_tracking
   user_guide/data_scientist_guide/federated_xgboost/federated_xgboost
   user_guide/data_scientist_guide/flower_integration/flower_integration
   user_guide/nvflare_cli/fl_simulator
   user_guide/data_scientist_guide/poc
   CLI Tools <user_guide/nvflare_cli/nvflare_cli>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Deployment & Operations

   user_guide/admin_guide/deployment/overview
   programming_guide/provisioning_system
   user_guide/admin_guide/deployment/dashboard_ui
   user_guide/admin_guide/deployment/operation
   user_guide/admin_guide/deployment/cloud_deployment
   user_guide/admin_guide/deployment/aws_eks
   user_guide/admin_guide/deployment/containerized_deployment
   user_guide/admin_guide/deployment/helm_chart
   user_guide/admin_guide/deployment/notes_on_large_models
   user_guide/admin_guide/monitoring
   user_guide/admin_guide/configurations/configurations
   user_guide/admin_guide/configurations/communication_configuration
   user_guide/admin_guide/configurations/logging_configuration
   user_guide/admin_guide/configurations/variable_resolution
   user_guide/admin_guide/configurations/server_port_consolidation

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Security

   flare_security_overview
   user_guide/admin_guide/security/terminologies_and_roles
   user_guide/admin_guide/security/identity_security
   user_guide/admin_guide/security/site_policy_management
   user_guide/admin_guide/security/authorization_policy_previewer
   user_guide/admin_guide/security/communication_security
   user_guide/admin_guide/security/data_privacy_protection
   user_guide/admin_guide/security/serialization
   user_guide/admin_guide/security/auditing
   user_guide/admin_guide/security/unsafe_component_detection

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Confidential Computing

   user_guide/confidential_computing/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Edge Development

   user_guide/edge_development/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Architecture & Developer Guide

   developer_guide

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Troubleshooting

   user_guide/timeout_troubleshooting
   programming_guide/memory_management
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
When you are ready for production, the :ref:`Deployment & Operations <deployment_operations>` section covers provisioning, deployment, and operations.
The :ref:`Security <security>` section covers authentication, authorization, data privacy, and auditing.

For Developers
==============
For deep dives into FLARE architecture, custom workflows, and low-level APIs, see the
:ref:`Architecture & Developer Guide <developer_guide>` and the :ref:`API Reference <apidocs/modules>`.
