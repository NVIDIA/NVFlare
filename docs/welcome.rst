.. _welcome:

############################
Welcome to NVIDIA FLARE
############################

What is Federated Learning?
===========================

Federated learning lets multiple sites collaboratively train or evaluate models
using their local data. Sites perform training or evaluation where the data
resides and exchange model updates, metrics, or other approved results instead
of collecting the underlying training examples in one place.

This approach can enable collaboration where privacy, regulation, data
sovereignty, intellectual property, data ownership, or the cost of moving data
makes central collection impractical. Federated learning includes centralized
workflows such as federated averaging, decentralized workflows such as swarm
learning, and horizontal or vertical collaboration patterns.

Federated learning is one application of the broader **federated computing**
paradigm. Federated computing brings approved computation to data distributed
across parties or locations. It also supports evaluation, analytics,
statistics, site-local data processing, and custom multi-site computation.

What is NVIDIA FLARE?
=====================

`NVIDIA FLARE™ <https://developer.nvidia.com/flare>`_ (NVIDIA Federated
Learning Application Runtime Environment) is a domain-agnostic, open-source,
and extensible SDK for federated learning and other federated-computing
applications. It allows researchers and data scientists to adapt existing ML
and DL workflows to a federated paradigm and enables platform developers to
build secure, privacy-preserving solutions for distributed multi-party
collaboration.

NVIDIA FLARE builds its federated-learning capabilities on a general
federated-computing core. The core provides workflow orchestration, task
execution, communication, security, and lifecycle services shared by training,
evaluation, analytics, statistics, and custom distributed applications.

Because applications integrate through extensible Python APIs, NVIDIA FLARE is
not limited to a fixed model family or framework. Maintained integrations and
examples include PyTorch, TensorFlow, JAX, scikit-learn, XGBoost, Hugging Face,
NeMo, and Flower.

Keep Data at Its Source
=======================

NVIDIA FLARE moves approved computation to participating sites. In a properly
designed and governed federation, raw datasets remain at their source and only
outputs permitted by the collaboration and each site's policies leave the
site. Application owners and site operators define and enforce those policies
for their data, regulatory requirements, and threat model.

NVIDIA FLARE supplies controls for identity, authorization, secure
communication, auditing, privacy-preserving techniques, and confidential
computing. These controls support a collaboration's security and governance
design; using NVIDIA FLARE alone does not determine which data or results an
application is allowed to transmit.

From Development to Production
==============================

Applications use the same core programming model across three stages:

- **Simulator** runs server and client logic on one system for rapid
  application development and validation.
- **Proof of Concept (POC)** simulates a production deployment on one local
  host with separate processes and locally generated startup kits.
- **Production** runs across provisioned sites with production identities,
  authorization, networking, policies, and operations.

NVIDIA FLARE provides high-level Client and Recipe APIs, the Collaboration API
(Technical Preview), and lower-level controllers, executors, events, filters,
and communication components. Researchers can implement new algorithms and
workflows, data scientists can adapt existing applications, and platform teams
can provision and operate multi-site systems.

Continue from Here
==================

Run and understand a complete local federation in :doc:`Quick Start
<quickstart>`. The :doc:`documentation home <index>` then maps the maintained
paths for Agent Skills, API selection, examples and tutorials, custom workflow
research, deployment and security, and project contribution.

For broader product material, visit the `NVIDIA FLARE website
<https://nvidia.github.io/NVFlare/>`_ and `developer site
<https://developer.nvidia.com/flare>`_. See :doc:`Industry Use Cases
<industry_use_cases>` for applications and supporting material, and
:doc:`What's New <whats_new>` for current release information.


.. toctree::
   :hidden:

   fl_introduction
   flare_overview
   whats_new
   real_world_fl
