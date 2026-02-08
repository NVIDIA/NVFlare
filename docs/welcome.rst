.. _welcome:

############################
Welcome to NVIDIA FLARE
############################

What is Federated Learning?
===========================

Federated Learning is a distributed learning paradigm where training occurs across multiple clients, each with their own local datasets.
This enables the creation of common robust models without sharing sensitive local data, helping solve issues of data privacy and security.

The FL server orchestrates the collaboration by sending an initial model to clients. Clients train on their local data and send
model updates back for aggregation into a global model. After multiple rounds, a robust global model is developed -- all without
any raw data leaving its source.

.. image:: resources/fl_diagram.png
    :height: 350px
    :align: center

**Types of Federated Learning:**

- **Horizontal FL** -- Clients hold different data samples over the same features
- **Vertical FL** -- Clients hold different features over overlapping data samples
- **Swarm Learning** -- Decentralized FL where clients perform aggregation without a central server

What is NVIDIA FLARE?
=====================

**NVIDIA FLARE** (NVIDIA Federated Learning Application Runtime Environment) is a domain-agnostic, open-source,
extensible Python SDK that allows researchers, data scientists, and data engineers to adapt existing ML/DL and
compute workflows to a federated paradigm.

FLARE brings computing to distributed datasets rather than copying data to a central location.
Data remains within each compute node, with only pre-approved results shared among collaborators.

.. image:: resources/flare_overview.png
    :height: 400px

Key Features
============

**Built for Productivity**

- **Client API** -- Convert existing ML/DL training code to federated with minimal code changes
- **Job Recipe API** -- Pre-built recipes for FedAvg, FedProx, SCAFFOLD, XGBoost, Cyclic, and more
- **FL Simulator** -- Rapid development and prototyping on a single machine
- **POC Mode** -- Multi-process simulation of a federated network on one host
- **FLARE API** -- Run and monitor jobs directly from Python code or notebooks
- **Dashboard** -- Web UI for project setup, approval, and deployment artifact distribution
- **Experiment Tracking** -- Built-in support for MLflow, Weights & Biases, and TensorBoard

**Built for Security & Privacy**

- **Secure Provisioning** -- TLS certificate-based authentication
- **Authorization Policies** -- Fine-grained, site-controlled authorization
- **Privacy Preservation** -- Differential privacy, homomorphic encryption, private set intersection
- **Confidential Computing** -- Hardware-backed TEEs with AMD SEV-SNP and NVIDIA GPU support
- **Audit Logging** -- Complete audit trail for accountability

**Built for Scale**

- **Framework Agnostic** -- PyTorch, TensorFlow, XGBoost, scikit-learn, and more
- **Cross-Silo to Edge** -- From a handful of hospital sites to millions of mobile devices
- **Hierarchical Architecture** -- Multi-region, tiered FL for large-scale deployments
- **Multi-Job Execution** -- Concurrent job execution with resource management
- **3rd-Party Integration** -- FlareAgent for seamless integration with external systems

**Built for Customization**

- **Layered Architecture** -- Every layer is pluggable and customizable
- **Specification-Based APIs** -- Build alternative implementations following well-defined specs
- **Rich Examples** -- Extensive library of FL algorithms, workflows, and application examples

Product Lines
=============

FLARE consists of three product categories:

**FLARE Core**
    The full federated learning platform: communication infrastructure, workflows, controllers,
    Client API, Recipe API, FL Simulator, provisioning, deployment, and management tools.

**FLARE Confidential AI**
    Confidential Federated AI with hardware-backed security. Leverages Trusted Execution Environments
    (AMD SEV-SNP, Intel TDX) and NVIDIA GPU confidential computing for end-to-end IP protection.
    Supports both on-premises and Azure cloud deployments.

**FLARE Edge**
    Federated learning at the edge, supporting millions of devices with hierarchical architecture,
    asynchronous aggregation (FedBuff), device simulation, and mobile SDKs for Android and iOS
    (via ExecuTorch).

High-Level Architecture
=======================

The FLARE architecture comprises three main layers:

- **Foundation Layer** -- Communication infrastructure (CellNet/F3), messaging protocols, streaming, privacy preservation, and secure platform management
- **Application Layer** -- Building blocks for federated learning including federation workflows, learning algorithms, and execution APIs
- **Tooling Layer** -- FL Simulator and POC CLI for experimentation, plus deployment and management tools for production

For detailed architecture information, see :ref:`flare_system_architecture`.

Design Principles
=================

- **Less is more** -- We solve unique challenges by building an open platform that enables others to solve their specific problems
- **Design to specification** -- Every component and API is spec-based, so alternative implementations can be easily constructed
- **Build for real-world scenarios** -- Components handle unexpected events and fail gracefully
- **Keep the system general-purpose** -- Layered packaging with minimal dependencies enables diverse federated computing use cases
- **Client system friendly** -- Runs anywhere with minimal environmental dependencies and does not interfere with the deployment environment

What is New in 2.7.2
====================

NVIDIA FLARE 2.7.2 brings the Job Recipe API to general availability, introduces the Tensor-based Downloader for
efficient large model handling, and adds comprehensive timeout and memory management documentation.

**Highlights:**

- **Job Recipe API -- Generally Available**: Unified recipe architecture covering FedAvg, FedOpt, SCAFFOLD, Cyclic, XGBoost, and more across all major frameworks
- **Tensor-based Downloader**: Memory-efficient pull-based model transfer using safetensors format for large model training
- **Server-Side Memory Cleanup**: Automatic garbage collection and heap trimming to prevent RSS growth in long-running jobs
- **Edge Development**: New hierarchical architecture with EDIP protocol, mobile SDKs, and device simulation for FL at scale
- **Confidential Computing**: End-to-end IP protection with AMD SEV-SNP + NVIDIA GPU TEEs

See :doc:`release_notes/flare_272` for full release notes.
See :doc:`release_notes/previous` for previous releases.

Roadmap
=======

NVIDIA FLARE continues to evolve with a focus on:

- **Simplified APIs** -- Making federated learning more accessible with higher-level abstractions and improved developer experience
- **LLM & Generative AI** -- Enhanced support for federated fine-tuning of large language models, including PEFT and quantization workflows
- **Edge & Mobile** -- Expanding edge device support with new platform integrations and improved scalability
- **Confidential Computing** -- Broader hardware TEE support (Intel TDX) and additional cloud provider integrations
- **Performance & Scale** -- Continued optimization for large-scale deployments, large models, and high-throughput scenarios
- **Ecosystem Integration** -- Deeper integration with ML frameworks, MLOps platforms, and data engineering tools

For the latest updates, visit the `NVIDIA FLARE GitHub <https://github.com/NVIDIA/NVFlare>`_.


.. toctree::
   :hidden:

   fl_introduction
   flare_overview
   whats_new
   real_world_fl
