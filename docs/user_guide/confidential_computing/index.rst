.. _confidential_computing:

##################################################
Federated Confidential Computing with NVIDIA FLARE
##################################################

.. admonition:: Technical Preview

   Federated Confidential Computing with NVIDIA FLARE is a **technical preview**.
   Supported configuration targets do not imply hardware acceptance for every
   platform, software revision, or workload. Consult the selected deployment's
   validation status before approving it for use.

Federated learning keeps training data at participating sites, but models and
updates still cross organizational boundaries. Model owners, data owners, and
infrastructure operators can have different interests and administrative rights.

FLARE combines hardware-backed confidential execution with participant
attestation and deployment-specific workload authorization. An approved
deployment can protect code, model weights, and data in use from the host
operator. Its guarantees also depend on trusted guest software, application
behavior, storage, and the policy governing released outputs.

Three protection goals
======================

**Build explicit trust between participants.** A TLS certificate identifies a
participant; it does not establish the state of its execution environment.
Attestation lets participants verify signed evidence against configured
requirements. Workload authorization determines which approved application may
receive a particular secret. These checks serve different purposes.

**Protect aggregation at the server.** A protected server processes individual
model updates inside an approved Trusted Execution Environment (TEE). The
aggregation application can still see those updates, while the host operator
stays outside the protected execution boundary. This is distinct from a
cryptographic secure-aggregation protocol. Authorized aggregate outputs can
still reveal information and require their own privacy policy.

**Protect training and model IP at client sites.** A model owner can supply
approved code and weights for execution on a data owner's confidential compute.
The data owner controls access to local data. Neither party should assume that
a TEE makes the application benign: the approved application can read its inputs
and may disclose them through permitted outputs.

Getting started
===============

Start with :ref:`cc_getting_started` to choose a deployment approach, identify
the prerequisites, and prepare your workload before configuring provisioning.

.. toctree::
   :maxdepth: 1

   getting_started

Architecture and security
=========================

Read :ref:`cc_architecture_overview` for responsibilities, trust boundaries,
attestation, and protection limits. The detailed Confidential Containers (CoCo)
analysis covers its deployment-specific security model and remaining gaps.

.. toctree::
   :maxdepth: 1

   architecture
   Confidential Containers (CoCo) security architecture <coco_security_architecture>

Azure Confidential Computing Deployment
=======================================

FLARE integrates with Azure confidential virtual machines and confidential
Azure Container Instances (ACI), using Microsoft Azure Attestation for
participant evidence. Azure image and resource preparation is separate from
NVFlare's ``azure_cc`` provisioning configuration.

* :ref:`Confidential ACI deployment <confidential_azure_container_instances_deployment>`:
  run an NVFlare server and its aggregation workload inside confidential ACI.
* :ref:`Azure confidential VM deployment <azure_confidential_virtual_machine_deployment>`:
  create and attest a confidential VM, then install NVFlare and its startup kit.

Deployment guides and configuration reference
==============================================

After choosing a deployment and reviewing its security model, use the
corresponding operator guide. The unified deployment reference describes the
shared provisioning configuration; it is not the introductory guide.

.. toctree::
   :maxdepth: 1

   on_premises/index
   azure/index
   deployment
