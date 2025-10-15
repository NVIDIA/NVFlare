.. _flare_security_overview:

###############################
NVIDIA FLARE Security Overview
###############################

Before diving into the security architecture, it will be helpful to understand FLARE's system architecture: :ref:`flare_system_architecture`

Security in Federated Computing Systems
========================================

Critical Security Concerns in Federated Learning
-------------------------------------------------

**Data Privacy**

- Protect training data from model inversion, membership inference, and property inference attacks
- Prevent gradient leakage during parameter sharing

**System Security**

- Authenticate participants and prevent man-in-the-middle, Sybil, and DoS attacks
- Ensure secure transmission of models and gradients

**Model Security**

- Defend against model poisoning, backdoor attacks, theft, and adversarial manipulation

**Participant Privacy**

- Protect participant identities, participation, and organizational IP

**Computation Integrity**

- Verify correct computation and detect malicious or faulty updates
- Ensure honest execution of the FL protocol

**Access Control**

- Enforce role-based permissions, resource usage limits, and data/model access restrictions

**Regulatory Compliance**

- Comply with data protection regulations (e.g., GDPR, HIPAA)
- Maintain cross-border data governance and audit trails

**Infrastructure Security**

- Secure edge devices, servers, communication channels, and model storage

**Trust Management**

- Establish participant legitimacy and trust using reputation systems

**Aggregation Security**

- Implement secure and Byzantine-robust aggregation protocols
- Protect against colluding participants


NVIDIA FLARE Security Architecture
===================================

NVIDIA FLARE is designed with robust security measures to protect federated learning systems. Its security framework combines built-in application features with each site's existing IT infrastructure.

Security Framework
------------------

FLARE addresses the critical security concerns identified above through multiple security pillars:

- **Access Control** - Role-based access control, resource usage limits, and fine-grained permissions for models and data
- **Regulatory Compliance** - Support for GDPR, HIPAA, and other regulations with comprehensive audit trails
- **Infrastructure Security** - Secure edge devices, servers, communication channels, and model checkpoint storage
- **Trust Management** - Reputation systems and mechanisms for establishing and verifying participant legitimacy
- **Aggregation Security** - Byzantine-robust aggregation protocols that protect against colluding participants

NVIDIA FLARE Security Measures
-------------------------------

FLARE implements security in several key areas:

**Identity Security**

Uses Public Key Infrastructure (PKI) for authentication and authorization of communicating parties. This involves a provisioning tool to create unique "Startup Kits" with security credentials for each participant (Server, Clients, Users), ensuring mutual authentication.

**Secure Filter**

Primary data privacy protection technique that transforms input/output objects to prevent model inversion and data leakage.

- **Per-Job (researcher)** - Defined in job config, applies to a single job
- **Per-Site (org admin)** - Defined in privacy.json, applies to ALL jobs via scope-based policies (e.g., "public"/"private")
- Site filters execute before job filters

**Site Policy Management**

Each participating organization can define and enforce its own policies for resource management, authorization, and privacy protection, offering decentralized control.

**Communication Security**

Employs secure protocols like TLS (both mutual TLS and standard TLS with signed messages) for confidential data communication. It also supports "Bring Your Own Connectivity" (BYOConn) with explicit message authentication for flexible network setups.

**Message Serialization**

Utilizes FOBS (Flare Object Serializer), a secure alternative to Python Pickle, which uses MessagePack for safe serialization and deserialization of data exchanged between the server and clients.

**Data Privacy Protection**

Integrates various privacy-enhancing technologies (PETs) at the organization level to prevent local data leakage.

**Trust-based Security**

Leverages confidential computing's VM-based Trusted Execution Environments (TEEs) to provide an isolated and secure environment for workloads, enhancing data security and privacy during processing. This includes capabilities for building explicit trust among participants, secure aggregation, and model theft prevention.

Key Operational Aspects
-----------------------

- **Federated Authorization** - Instead of a centralized approach, each organization defines and enforces its authorization policy, allowing local control over access to computing resources and FL jobs
- **Site-Specific Authentication** - Allows for custom local authenticators at each site, enabling integration with existing in-house authentication systems
- **No Raw Data Movement** - A fundamental principle is that no raw data ever leaves the participating institutions; only model weights are transmitted


Identity Security
=================

PKI-Based Authentication
------------------------

NVIDIA FLARE's authentication relies on Public Key Infrastructure (PKI) technology. A Project Admin uses a Provisioning Tool to create a Root Certificate Authority (CA) and then issues certificates for all communicating parties (Server, Clients, Users). Each participant receives a "Startup Kit" containing their security credentials (Root CA certificate, identity certificate, and private key) for TLS authentication.

These Startup Kits are distributed to the respective Project Admin (for the FL Server), Org Admin (for each FL Client), and users (for Flare Console). The integrity of each kit is ensured by a signature from the Root CA.

Upon startup, FL clients establish a TLS connection with the FL server using these PKI credentials. The system's security is thus dependent on the secure handling and distribution of these Startup Kits. The provisioning tool uses strong cryptography, with X.509 compliant certificates, 2048-bit private keys, and a 360-day expiry. The NVFlare Dashboard also allows users to download their Startup Kits.

Site-Specific Security and Privacy Policies
--------------------------------------------

NVIDIA FLARE allows organizations to define different scopes within their privacy policies. These scope-specific policies are enforced through FLARE's filter mechanism. For each non-public scope, administrators can define filters to enforce desired security behaviors and prevent accidental data leakage by data scientists.

NVIDIA FLARE provides role-based security mechanisms to control user access. These controls, known as federated policies, are enforced at each site rather than centrally. FLARE's Security documentation details various aspects, including:

- Centralized vs. Federated Authorization
- Policy Configuration
- Roles and Rights
- Controls and Conditions
- Command Categories
- Policy Evaluation
- Command Authorization Process
- Job Submission Authorization Process
- Job Management Commands Authorization

Site-Specific Authentication and Federated Job-Level Authorization
-------------------------------------------------------------------

NVFlare supports site-specific authentication and job-level authorization, enabling each site to inject its own security mechanisms for server/client registration, job deployment, and runtime control.

NVFlare's event-based, pluggable framework allows users to:

- Integrate external authentication systems (e.g. KeyCloak, etc.)
- Use confidential CAs to verify site identity and confidential computing compliance
- Define roles controlling job submission and dataset access

Users can implement custom FLComponents that listen to NVFlare system events and plug in authentication or authorization logic as needed.

**Assumptions and Risks**

Custom security plugins gain access to sensitive data (e.g., IDENTITY_NAME, PUBLIC_KEY, CERTIFICATE), which must remain read-only to prevent compromise. Misconfigured plugins may block job deployment or execution; users must understand when and where to apply checks.

**Integration with External Systems**

Federated environments often involve institutions with distinct in-house authentication systems. NVFlare allows per-site integration, supporting diverse mechanisms (e.g., OLAP, OAuth, KeyCloak). The event-based plugin framework provides a unified way to integrate any external authentication or authorization process.


Communication Security
======================

Connection and Message Security
--------------------------------

FLARE ensures secure communication through TLS and mutual TLS (mTLS). In mTLS mode, the server and clients authenticate each other during connection setup, ensuring that only clients with valid startup kits can connect.

For environments where mTLS is limited by IT policies, FLARE supports BYOConn (Bring Your Own Connectivity), enabling users to integrate their own networking solutions—provided they:

- Allow clients to reach the server (directly or via proxies)
- Maintain confidentiality and integrity of all messages
- Enforce explicit message authentication to prevent unauthorized access

Explicit Message Authentication
--------------------------------

FLARE requires every message to include a valid authentication token and signature:

1. Upon startup, the client logs in to the server using credentials from its startup kit
2. The server issues a token and signature (signed with its private key) binding the client identity to the token
3. The client attaches its name, token, and signature to each message header
4. The server validates these before processing the message

This mechanism works independently of TLS or mTLS and relies on PKI credentials in the startup kits. All sites must protect these kits and never share tokens or signatures.

Message authentication applies to all messages—including those from FL clients, 3rd-party systems, and the server itself.

Connection Security Modes
--------------------------

FLARE's Provision system supports multiple connection security modes, allowing each site to explicitly define how communication is secured.

**TLS (One-way Authentication)**

In TLS, the client authenticates the server using a trusted Root Certificate. The server presents proof signed with its private key, which the client validates using the server's public key. The server does not authenticate the client.

**mTLS (Mutual Authentication)**

In mTLS, both the client and server authenticate each other using PKI credentials from their startup kits. Each side proves its identity by presenting evidence signed with its private key and validating the peer's certificate.

Both parties must use the same mode (TLS or mTLS) for a secure connection.

**Clear Mode**

Clear mode allows unencrypted communication, typically only used when a secure proxy terminates TLS before forwarding traffic to the server.

**Custom Root Certificates**

A custom CA certificate can be used to validate servers in custom network environments, such as those involving intermediate proxies. If not provided, the default root CA generated by the Provision system is used.

If no security mode is specified, mTLS is applied by default.


Secure Message Serialization
============================

In distributed systems, message serialization is a critical security concern. The commonly used Python Pickle mechanism is considered insecure because it can execute arbitrary code during deserialization.

To address this, NVFLARE employs a secure serialization framework called FOBS (FLARE Object Serializer) for all server–client data exchanges.

FOBS Overview
-------------

FOBS is a secure, Pickle replacement built on MessagePack. It ensures that only explicitly supported and registered object types can be serialized, preventing code execution or tampering attacks.

Instead of relying on Python introspection, FOBS uses registered Decomposers to define how each object is safely converted into MessagePack-supported types. This strict registration model enforces type control and eliminates unsafe dynamic behavior.

FOBS automatically supports common data types, enumerations, and dataclasses, and raises errors for unsupported objects—ensuring unregistered or potentially unsafe objects cannot be serialized.

Security Properties
-------------------

- Prevents arbitrary code execution during deserialization
- Enforces type whitelisting through registered decomposers
- Uses MessagePack for compact, cross-language binary encoding
- Ensures data integrity and type safety across federated nodes

FOBS underpins FLARE's secure communication layer by guaranteeing that only validated and structured data objects are exchanged within the federated system.


Trust-Based Security with Confidential Computing
=================================================

Confidential Computing leverages trusted execution environments (TEEs) to protect data and workloads while in use, ensuring confidentiality and integrity even from privileged system software.

Key Capabilities
----------------

- **Trusted Execution Environment (TEE)** - Isolated environment protecting applications and data during processing
- **Virtualization-Based Security** - Supports unmodified workloads via confidential VMs, containers, or Kubernetes pods
- **Secure Transfer** - Hardware-accelerated encryption for data in transit between CPUs and GPUs
- **Hardware Root of Trust** - Attestation and authenticated firmware ensure system integrity and trustworthiness

Technology Stack
----------------

- **Hardware** - AMD SEV-SNP, Intel TDX CPUs; NVIDIA H100/Blackwell GPUs
- **Virtualization** - Confidential VM, Confidential Container, Kata Containers on Kubernetes
- **Key Broker Service (KBS)** - Facilitates remote attestation and secret delivery
- **Attestation Service** - Verifies hardware and system trustworthiness

Confidential Federated AI
--------------------------

Confidential Computing enables trustworthy federated learning by verifying participant integrity and protecting data and models. Key security use cases include:

- **Building Explicit Trust** - Attestation checks ensure participant trust at different stages
- **Secure Aggregation** - FL server operates in a TEE to aggregate client updates securely, preventing model inversion attacks
- **Model Theft Prevention** - TEEs on all nodes protect model IP and training data, preventing unauthorized access or reverse engineering

FLARE Confidential AI Solution offers end-to-end protection: we not only protect the IP (model and code) in use at runtime, but also protect against CVM tampering at deployment.

The solution is able to perform:

- Secure aggregation on the server-side to protect against privacy leaks via model
- Model theft protection on the client-side to safeguard Model IP during collaboration
- Data leak prevention on the client-side with pre-approved, certified code

Frequently Asked Questions
==========================

Where is the data being stored?
--------------------------------

The data is stored in local institutions' storage, either on-premises or in private cloud accounts. FLARE doesn't move the data.

What results are being shared with the central federated site?
---------------------------------------------------------------

The "central federated site" is known as the "FL server" in NVFlare. The location of the final results depends on the aggregator location, which varies by algorithm.

- **For FedAvg-type workflows** - The aggregator is in the FL Server, and the final result is stored in the Job Store, which is a local disk or volume accessible to the FL server
- **For Swarm learning algorithm** - The aggregator is randomly selected at each round among all participating sites, and the final result is stored locally at that site, not shared with the FL Server

In both cases, the aggregator can be a locked-down machine using current security and privacy best practices. Confidential Computing is the latest technology that adds an extra layer of security.

What's the IRB number?
----------------------

The IRB process is outside the scope of NVIDIA FLARE; consortium participants will need to handle this separately.

What do we need to install locally?
------------------------------------

FLARE involves the following steps for local installations:

1. **Provision** - A process to generate software packages and certificates for each participant (called startup kit)
2. **Distribution** - Send the startup kit to each participant
3. **Start** - Participants can pip install nvflare and start the startup kit

FLARE offers two ways to provision:

- **nvflare provision CLI command** - The package is generated locally, the project administrator will then distribute the package manually (sftp, email etc.). This requires the project administrator to know all site information including all the names of the sites
- **FLARE Dashboard** - The web interface allows the project administrator to invite others to join the project and provide site-specific information themselves. The project admin approves the participating client for the sites, and then the startup kit can be downloaded for each participant

For the details of installation instructions, please refer to:

- Installation documentation: https://nvflare.readthedocs.io/en/main/installation.html
- Deployment guide: https://nvflare.readthedocs.io/en/main/user_guide/admin_guide/deployment/index.html

What's the infrastructure on each institution's side?
------------------------------------------------------

FLARE doesn't mandate a specific type of infrastructure, unless you want to leverage confidential computing with IP protection. You can run on CPU or GPU. The minimal requirement is an 8GB CPU with a Linux distribution (such as Ubuntu). For deep learning models, you will need a GPU for faster training.

Do we need to run Docker on our end?
-------------------------------------

No, that's not required. You can use Docker if you like.

Does it run on Red Hat?
-----------------------

Yes.

Who maintains FLARE?
--------------------

NVIDIA FLARE is an open source project, contributed and maintained by NVIDIA and the NVFLARE community.

The software is distributed under the Apache 2.0 License, which is a permissive open-source license. Under this license, the software is provided "as is" without warranties or liabilities. There is no vendor or individual who can be held legally responsible for defects or damages arising from its use. Users are free to use, modify, and distribute the software at their own discretion and risk.

If formal support or indemnification is required, it can be obtained through a third-party service provider that offers commercial support for Apache 2.0–licensed software.

Who owns the data after it leaves the institution?
--------------------------------------------------

No raw data ever leaves any institution. Only the model weights are transmitted.

The model trained can be owned by different participants depending on the collaboration agreement. The model can be:

- Shared by all, or
- Owned by the initial (pre-trained) model owner, or
- Be a shared global model, and personalized model owned by participating clients

It all depends on the business model and contracts. NVIDIA FLARE is not involved in these business decisions.

Is there a data use agreement?
-------------------------------

Since the data never leaves the institution, usually there is no specific data use agreement.

However, collaborators usually need to decide what data to use to jointly train a model. Some kind of agreement needs to be made.

Is the code proxy-aware?
------------------------

Yes, NVIDIA FLARE can operate through network topologies with proxies, such as reverse-proxies or Kubernetes ingress services.

FLARE supports two types of TLS protocols: mutual TLS and standard TLS.

- **For mutual TLS** - The certificate termination point is at the server, not at the proxy. As a result, the proxy must be configured to enable TCP pass-through
- **For standard TLS** - Users can simply use the pre-authorized certificate for TLS handshaking and the FLARE certificate for authentication. In this case, no network configuration is needed

For example, if you use HTTP as a communication protocol, connections through a proxy or Kubernetes ingress service work just like any other HTTPS service. FLARE can switch between gRPC and HTTP via configuration without affecting FLARE applications.

**Additional Resources:**

- Server port consolidation: https://nvflare.readthedocs.io/en/main/user_guide/admin_guide/configurations/server_port_consolidation.html#server-port-consolidation
- Communication security: https://nvflare.readthedocs.io/en/main/user_guide/admin_guide/security/communication_security.html
