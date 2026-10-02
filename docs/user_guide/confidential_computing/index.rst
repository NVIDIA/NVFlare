.. _confidential_computing:

###############################
FLARE Confidential Federated AI
###############################

.. admonition:: FLARE Confidential Federated AI

   This feature is in **Technical Preview**.
   CVM Builder is included in the NVIDIA FLARE source tree.

Introduction
============

Federated Learning faces critical trust challenges even among collaborating organizations.

- **Trust of participants** is difficult to establish
- participants may worry about **code tampering** during execution.
- **Model owners** are concerned about **model theft** and **model tampering** that could compromise their intellectual property.
- **Data owners** fear **model inversion attacks** that could extract training data and
- **data leakage** through gradients or model parameters or other accidental code changes

Traditional federated learning relies on organizational trust agreements, but these cannot guarantee runtime security or
prevent malicious behavior during model training and aggregation.

Security Risks in Federated Learning
------------------------------------

Federated learning operations face multiple security risks throughout the entire lifecycle:

**Deployment-Time Risks**

At deployment time, code is particularly vulnerable when introduced into an untrusted or unverified environment.
An untrusted host or malicious host owner can intercept the model by:

- Modifying the application code before execution begins
- Tampering with the execution environment
- Delaying the activation of security mechanisms such as attestation and encryption
- Injecting malicious code during the deployment phase before protections are activated

Without strict controls over when and how models are decrypted or loaded, attackers can gain early access before protections
are in place, making deployment a critical point of exposure.

**Runtime Risks**

Even after secure deployment, model Intellectual Property (IP) and training data remain exposed to runtime threats:

- **Compromised participant machines** - Attackers may exploit vulnerabilities to gain remote access
- **Unauthorized access** - Direct access or network access to remote training machines
- **Network-based leaks** - Interception of model parameters or gradients during transmission
- **Storage-based leaks** - Extraction from disk-based model checkpoints or intermediate results
- **Memory extraction** - Copying models or data directly from system memory
- **Insider threats** - Malicious participants or administrators with physical or logical access

These risks exist regardless of organizational trust agreements and cannot be fully mitigated through traditional security measures alone.

What is Confidential Computing?
-------------------------------

Confidential Computing leverages hardware-based Trusted Execution Environments (TEEs) to protect data and code during execution.

- **VM-based confidential computing** uses technologies like **AMD SEV-SNP** (Secure Encrypted Virtualization-Secure Nested Paging) and **Intel TDX** (Trust Domain Extensions) to create isolated, encrypted virtual machines where memory is protected from the host OS, hypervisor,and even administrators.
- **NVIDIA GPU Confidential Computing** extends this protection to GPU workloads, enabling encrypted data transfer between CPU and GPU with hardware-accelerated encryption (H100 and Blackwell GPUs).

These technologies provide a hardware root of trust through attestation, allowing participants to verify that workloads
are running in genuine secure environments before sharing sensitive data or models.

Risk Mitigation with Confidential Computing
-------------------------------------------

FLARE's Confidential Computing integrations provide mechanisms for protecting workloads from an untrusted host:

- **Protected Aggregation on Server** - A TEE can isolate server-side computation and plaintext model updates from the server's host operator. This is not the same as a cryptographic secure-aggregation protocol and does not by itself prevent model inversion or inference from authorized outputs.
- **IP Protection on Client** - An approved confidential deployment can protect model code and weights from the client host operator, provided that the application, guest software, key-release policy, storage, and output handling preserve that boundary.
- **Controlled Workload Admission** - Attestation and workload authorization can restrict execution to approved code and configurations. Approval does not establish that code is benign: an admitted application can disclose data or secrets that it is allowed to use.

Data-use restrictions, differential privacy, output review, application security, and any required cryptographic secure
aggregation remain separate design requirements. See :ref:`coco_security_architecture` for the CoCo integration's
specific guarantees, assumptions, and residual threats.

Deployment-specific lockdown can include encrypted storage, disabled login, blocked SSH access, and restricted network
ports. CVM Builder and CoCo enforce these controls differently; consult the selected architecture and verify the
effective configuration rather than assuming that enabling a TEE automatically enables every control.

The selected deployment must establish and validate protection throughout the lifecycle:

- **Deployment Protection** - Bind approved workload content and configuration to attestation-based authorization before releasing secrets.
- **Runtime Protection** - Use the TEE to isolate guest memory from the host, with reviewed guest software and workload behavior.
- **Storage Protection** - Encrypt and authenticate confidential persistent state; TEE memory protection alone does not protect host-backed files.
- **Trust Establishment** - Verify evidence against explicit platform and workload requirements before sharing sensitive material.
- **Access Control Lockdown** - Review interactive access, guest-agent APIs, network endpoints, and output channels for the selected deployment.

Operational Risks Even with Confidential Computing
--------------------------------------------------

While Confidential Computing significantly enhances security, certain operational risks remain that require additional safeguards:

- **Deployment-time Code Injection** - If an attacker can modify the application code at deployment time before the CVM is launched, they could add code to copy encryption keys, model checkpoints, or leak data during execution
- **Application-level Vulnerabilities** - If an attacker compromises the application running inside the TEE (through bugs, backdoors, or malicious updates), the TEE protection cannot prevent IP leakage
- **Host-level Storage Vulnerabilities** - Model checkpoints written to host disk storage may be accessible from the host filesystem, bypassing runtime memory protection
- **Side-channel Attacks** - Sophisticated attacks may exploit timing, power consumption, or other side channels to extract information

.. warning::

   **Critical Design Requirement:**

   Even with Confidential Computing, without proper design of the CVM to extend the chain of trust from hardware
   to the application workload, confidential computing attestation will **NOT** be able to detect deployment-time
   code modifications or tampering. The deployment must extend the chain of trust from verified platform evidence to
   authenticated application content and enforced workload policy. Not every application byte is necessarily part of
   the CPU launch measurement.

These risks require additional safeguards including:

- Secure deployment pipelines with code integrity verification through attestation before CVM activation
- Encrypted persistent storage with proper key management
- CVM access and network lockdown to prevent unauthorized entry points
- Regular security audits and vulnerability assessments

This comprehensive approach enables organizations to collaborate on federated learning while maintaining strong IP protection guarantees.


FLARE Confidential Federated AI Overview
========================================

NVIDIA FLARE provides Confidential Federated AI capabilities through hardware-backed security. Deployment paths include
CVM Builder, Confidential Containers (CoCo) on Kubernetes, and Azure Confidential Computing. They share some NVFlare
attestation components but have different boot, packaging, and key-release mechanisms; do not mix their runbooks.

Confidential Containers on Kubernetes
-------------------------------------

The CoCo integration runs protected NVFlare clients and, optionally, the server inside Kata confidential VMs. The
``2.9`` implementation includes all four targets: AMD SEV-SNP or Intel TDX, each with or without an NVIDIA confidential GPU.
**TDX remains experimental pending hardware acceptance:** encrypted NVFlare execution, attestation-gated key release,
and peer-proof generation/verification remain unverified end to end on TDX. See :ref:`coco_security_architecture`
for implementation scope and validation status.
Trusted provisioning builds, signs, and encrypts workload images; independently
administered Trustee services authorize their decryption using approved platform references and workload policy.

Start with :ref:`coco_security_architecture` for the complete security model, trust boundaries, attack analysis, and
links to role-specific deployment procedures. CoCo uses per-workload encrypted images and InitData-bound guest policy,
not CVM Builder's generic root-and-vault packaging.

On-Premises IP Protection Deployment
------------------------------------

FLARE's on-premises Confidential Federated AI solution provides comprehensive IP protection for organizations that need to protect proprietary models and training code during federated collaboration. This solution leverages confidential virtual machines (CVMs) with:

- **Intel TDX** - Confidential VMs running on Intel Trust Domain Extensions
- **AMD SEV-SNP CPU with optional NVIDIA GPU** - Confidential VMs running on AMD processors with Secure Encrypted Virtualization, optionally paired with a supported NVIDIA confidential-computing GPU

- **End-to-End IP Protection** - Model code, weights, and training algorithms are protected throughout the entire lifecycle, from deployment through execution to result storage
- **Attestation-Based Trust** - Hardware-backed attestation verifies the integrity of execution environments before model IP is released to client sites
- **Secure Deployment Pipeline** - Ensures only certified, unmodified training code is deployed to confidential VMs, preventing deployment-time tampering
- **CVM Lockdown** - Comprehensive access control hardening on both server and client CVMs (primarily on client side) including disabled login, blocked SSH access, and restricted network ports to prevent unauthorized access to the protected environment

This solution is intended for organizations with high-value proprietary models collaborating with partners who may have different security postures or trust levels.


Azure Confidential Computing Deployment
---------------------------------------

For organizations seeking cloud-based confidential federated learning, FLARE supports running federated learning workloads on Azure Confidential Computing infrastructure. This deployment option provides:

**Trust Establishment Among Participants**

Azure Confidential Computing enables participants to establish explicit trust through:

- **Remote Attestation** - Each participant can verify that the FL server is running in a genuine confidential environment before submitting updates
- **Hardware Root of Trust** - Azure's confidential computing infrastructure provides cryptographic proof of the execution environment's integrity
- **Transparent Security Posture** - Participants can independently verify the security properties of the federated learning environment

This deployment model is suitable for organizations that prioritize data privacy and secure aggregation while training code and model architectures can be shared among trusted participants.


Choosing the Right Deployment
=============================

- Use **Confidential Containers on Kubernetes** when protected workloads are deployed as Pods on an untrusted cluster; start with :ref:`coco_security_architecture`.
- Use **On-Premises IP Protection / CVM Builder** when deploying the standalone root-and-vault CVM workflow.
- Use **Azure Confidential Computing** when the primary concern is data privacy and secure aggregation among trusted collaborators


.. toctree::
   :maxdepth: 2

   coco_security_architecture
   on_premises/index
   azure/index
