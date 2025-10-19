.. _confidential_computing:

################################
FLARE Confidential Federated AI
################################

.. admonition:: FLARE Confidential Federated AI

   This feature is in **Technical Preview**.
   Reach out to the NVIDIA FLARE team for CVM build scripts: federatedlearning@nvidia.com


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
--------------------------------------

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

Even after secure deployment, model IP and training data remain exposed to runtime threats:

- **Compromised participant machines** - Attackers may exploit vulnerabilities to gain remote access
- **Unauthorized access** - Direct access or network access to remote training machines
- **Network-based leaks** - Interception of model parameters or gradients during transmission
- **Storage-based leaks** - Extraction from disk-based model checkpoints or intermediate results
- **Memory extraction** - Copying models or data directly from system memory
- **Insider threats** - Malicious participants or administrators with physical or logical access

These risks exist regardless of organizational trust agreements and cannot be fully mitigated through traditional security measures alone.

What is Confidential Computing?
--------------------------------

Confidential Computing leverages hardware-based Trusted Execution Environments (TEEs) to protect data and code during execution.

- **VM-based confidential computing** uses technologies like **AMD SEV-SNP** (Secure Encrypted Virtualization-Secure Nested Paging) and **Intel TDX** (Trust Domain Extensions) to create isolated, encrypted virtual machines where memory is protected from the host OS, hypervisor,and even administrators.
- **NVIDIA GPU Confidential Computing** extends this protection to GPU workloads, enabling encrypted data transfer between CPU and GPU with hardware-accelerated encryption (H100 and Blackwell GPUs).

These technologies provide a hardware root of trust through attestation, allowing participants to verify that workloads
are running in genuine secure environments before sharing sensitive data or models.

Risk Mitigation with Confidential Computing
--------------------------------------------

FLARE's Confidential Computing solution addresses the federated learning security risks through three key mechanisms:

- **Secure Aggregation on Server** - The FL server operates within a TEE to aggregate client updates securely, preventing model inversion attacks and ensuring aggregated model parameters cannot be intercepted or tampered with
- **IP Protection on Client** - Model code and weights are protected within confidential VMs on client sites, preventing model theft and unauthorized access to proprietary algorithms or pre-trained models
- **Data Leakage Prevention on Client** - Pre-approved, certified training code runs in isolated TEEs, ensuring that only authorized computations occur and preventing malicious code from exfiltrating training data

FLARE's IP protection solution includes CVM lockdown features that disk encryption, disable login access, block SSH connections, and restrict
network ports to prevent unauthorized access to the protected environment. These lockdown features apply to both server and client CVMs,
with primary focus on client-side protection where model IP is most vulnerable.

FLARE's solution provides end-to-end security throughout the entire lifecycle:

- **Deployment Protection** - Attestation-based verification ensures only certified, unmodified code packages are deployed to confidential VMs
- **Runtime Protection** - TEEs protect model IP and training code during execution, preventing extraction or reverse engineering. CVM access is locked down with disabled login, SSH, and controlled network ports to prevent unauthorized access
- **Storage Protection** - Integration with encrypted storage solutions and key management systems protects model checkpoints and intermediate results
- **Trust Establishment** - Remote attestation allows model owners to verify the security posture of client environments before releasing valuable IP, ensuring compliance with confidential computing requirements
- **Access Control Lockdown** - Comprehensive CVM hardening includes disabling interactive login, blocking SSH access, restricting network ports to only essential communication channels, and preventing unauthorized administrative access

Operational Risks Even with Confidential Computing
---------------------------------------------------

While Confidential Computing significantly enhances security, certain operational risks remain that require additional safeguards:

- **Deployment-time Code Injection** - If an attacker can modify the application code at deployment time before the CVM is launched, they could add code to copy encryption keys, model checkpoints, or leak data during execution
- **Application-level Vulnerabilities** - If an attacker compromises the application running inside the TEE (through bugs, backdoors, or malicious updates), the TEE protection cannot prevent IP leakage
- **Host-level Storage Vulnerabilities** - Model checkpoints written to host disk storage may be accessible from the host filesystem, bypassing runtime memory protection
- **Side-channel Attacks** - Sophisticated attacks may exploit timing, power consumption, or other side channels to extract information

.. warning::

   **Critical Design Requirement:**

   Even with Confidential Computing, without proper design of the CVM to extend the chain of trust from hardware
   to the application workload, confidential computing attestation will **NOT** be able to detect deployment-time
   code modifications or tampering. The CVM must be designed to ensure that attestation verifies the entire execution
   stack—from hardware through the application layer—to provide meaningful security guarantees.

These risks require additional safeguards including:

- Secure deployment pipelines with code integrity verification through attestation before CVM activation
- Encrypted persistent storage with proper key management
- CVM access and network lockdown to prevent unauthorized entry points
- Regular security audits and vulnerability assessments

This comprehensive approach enables organizations to collaborate on federated learning while maintaining strong IP protection guarantees.


FLARE Confidential Federated AI Offerings
==========================================

NVIDIA FLARE 2.7.0 introduces Confidential Federated AI capabilities that enable secure, trustworthy federated learning through hardware-backed security. The release includes two deployment options to address different organizational requirements:

On-Premises IP Protection Solution
-----------------------------------

FLARE's on-premises Confidential Federated AI solution provides comprehensive IP protection for organizations that need to protect proprietary models and training code during federated collaboration. This solution leverages confidential virtual machines (CVMs) with:

- **AMD SEV-SNP CPU + NVIDIA GPU** - Confidential VMs running on AMD processors with Secure Encrypted Virtualization, paired with NVIDIA H100 or Blackwell GPUs for GPU-accelerated confidential computing

.. note::

    Intel TDX support will be provided in a future release

- **End-to-End IP Protection** - Model code, weights, and training algorithms are protected throughout the entire lifecycle, from deployment through execution to result storage
- **Attestation-Based Trust** - Hardware-backed attestation verifies the integrity of execution environments before model IP is released to client sites
- **Secure Deployment Pipeline** - Ensures only certified, unmodified training code is deployed to confidential VMs, preventing deployment-time tampering
- **CVM Lockdown** - Comprehensive access control hardening on both server and client CVMs (primarily on client side) including disabled login, blocked SSH access, and restricted network ports to prevent unauthorized access to the protected environment

This solution is ideal for organizations with high-value proprietary models collaborating with partners who may have different security postures or trust levels.


Azure Confidential Computing Cloud Deployment
----------------------------------------------

For organizations seeking cloud-based confidential federated learning **without IP protection requirements**, FLARE supports running Federated learning workload on Azure Confidential Computing infrastructure.
This deployment option provides:

.. note::

    Other CSP supports as well as IP protection on the cloud will be in future releases.

**Trust Establishment Among Participants**

Azure Confidential Computing enables participants to establish explicit trust through:

- **Remote Attestation** - Each participant can verify that the FL server is running in a genuine confidential VM before submitting updates
- **Hardware Root of Trust** - Azure's confidential computing infrastructure provides cryptographic proof of the execution environment's integrity
- **Transparent Security Posture** - All participants can independently verify the security properties of the federated learning environment without relying solely on organizational agreements

This deployment model is suitable for organizations that prioritize data privacy and secure aggregation, while training code and model architectures can be shared among trusted participants.

**Choosing the Right Deployment**

- Use **On-Premises IP Protection** when model IP is highly valuable and must be protected from all participants
- Use **Azure Confidential Computing** when the primary concern is data privacy and secure aggregation among trusted collaborators
- Both options can be combined in hybrid deployments where some sites require IP protection while others focus on secure aggregation



Architecture Design for Confidential Federated AI with IP Protection
=====================================================================

The following documents provide detailed information about FLARE's Confidential Federated AI architecture for IP protection:

- :ref:`cc_architecture` - System architecture and component design
- :ref:`cc_deployment_guide` - Deployment guide for on-premises CVM setup with AMD SEV-SNP and NVIDIA GPU
- :ref:`confidential_computing_attestation` - Attestation mechanisms and trust establishment
- :ref:`hashicorp_vault_trustee_deployment` - Operational HashiCorp key vault deployment with Trustee



FLARE Deployment to Azure Confidential Computing
================================================

- Secure Aggregation on FLARE Server with Azure ACI (Azure Container Instance)
- Client training on CVM node

**documentation to be completed soon**
