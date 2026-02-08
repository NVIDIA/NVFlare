.. _flare_security_overview:
.. _security:

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


NVIDIA FLARE Security Architecture
===================================

NVIDIA FLARE is designed with robust security measures to protect federated learning systems. Its security framework combines built-in application features with each site's existing IT infrastructure.

Security Pillars
----------------

FLARE addresses the critical security concerns through multiple security pillars:

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Security Area
     - Description
     - Learn More
   * - **Identity & Access Control**
     - PKI-based authentication, role-based authorization, per-site policies, and site-specific authentication integration
     - :ref:`Identity Security <identity_security_page>`, :ref:`Site Policy Management <site_policy_management>`, :ref:`Terminologies & Roles <nvflare_roles>`
   * - **Network & Communication**
     - TLS/mTLS encryption, explicit message authentication, BYO connectivity, and proxy support
     - :ref:`Communication Security <communication_security>`
   * - **Data Privacy & Filters**
     - Privacy-preserving filters (differential privacy, homomorphic encryption), per-job and per-site filter chains
     - :ref:`Data Privacy Protection <data_privacy_protection>`
   * - **Secure Serialization**
     - FOBS (FLARE Object Serializer) replaces Python Pickle with type-safe, whitelisted serialization
     - :ref:`Message Serialization <serialization>`
   * - **Audit & Compliance**
     - Comprehensive audit logging for all user commands and job events, with correlated event tracking across sites
     - :ref:`Auditing <auditing>`
   * - **Component Safety**
     - Event-based detection of unsafe job components before execution, preventing data leakage from untrusted code
     - :ref:`Unsafe Component Detection <unsafe_component_detection>`
   * - **Confidential Computing**
     - Hardware-backed TEEs (AMD SEV-SNP, NVIDIA GPU) for secure aggregation, model IP protection, and data leak prevention
     - :ref:`Confidential Computing <confidential_computing>`

Key Principles
--------------

- **No Raw Data Movement** -- Raw data never leaves the participating institutions; only model updates are transmitted
- **Federated Authorization** -- Each organization defines and enforces its own authorization policy (not centralized)
- **Site-Specific Authentication** -- Each site can integrate its own authentication system (e.g., KeyCloak, OAuth)
- **Defense in Depth** -- Multiple layers of security from identity verification to confidential computing


Federated Security: Local Site Control
=======================================

A fundamental principle of NVIDIA FLARE's security architecture is that **each participating site retains full control
over its own security**. Unlike centralized systems where a single authority dictates security policies, FLARE adopts
a federated security model where hospitals, banks, government agencies, and other organizations integrate FLARE
into their existing security infrastructure -- not the other way around.

**Why Federated Security Matters**

In real-world federated learning deployments, each participating institution has its own:

- IT security policies and compliance requirements (HIPAA, SOX, GDPR, etc.)
- Authentication systems (Active Directory, LDAP, KeyCloak, OAuth, etc.)
- Network security infrastructure (firewalls, proxies, VPNs)
- Data governance and access control policies

FLARE is designed to respect and integrate with all of these. No central authority can override a local site's
security decisions.

**How It Works**

- **Local Authentication** -- Each site can plug in its own authentication mechanism. A hospital using KeyCloak and a bank using LDAP can both participate in the same FL project without changing their authentication systems. FLARE's event-based plugin framework allows custom authenticators at each site. See :ref:`Identity Security <identity_security_page>`.

- **Local Authorization** -- Each site defines its own authorization policy that determines what users can and cannot do on that site. A site can restrict which jobs are allowed to run, which resources can be used, and which users have access. The central FL server cannot override a site's authorization decisions. See :ref:`Site Policy Management <site_policy_management>`.

- **Local Privacy Policy** -- Each site defines its own privacy protection rules. For example, a hospital can require that differential privacy filters are applied to all model updates leaving the site, regardless of what the job configuration specifies. Site-level privacy policies always take precedence over job-level settings. See :ref:`Data Privacy Protection <data_privacy_protection>`.

- **Local Resource Control** -- Each site controls its own compute resources (GPUs, memory, storage) and can set limits on what FL jobs are allowed to consume.

- **Local Network Security** -- Sites operate behind their own firewalls and network policies. FLARE supports TLS, mTLS, proxies, and BYO connectivity to work within any network environment. See :ref:`Communication Security <communication_security>`.

This federated approach ensures that no single point of compromise can affect the entire system, and each
institution can meet its own regulatory and compliance requirements independently.


Frequently Asked Questions
==========================

Where is the data being stored?
--------------------------------

The data is stored in local institutions' storage, either on-premises or in private cloud accounts. FLARE doesn't move the data.

What results are being shared with the central federated site?
---------------------------------------------------------------

The "central federated site" is known as the "FL server" in NVFlare. The location of the final results depends on the aggregator location, which varies by algorithm.

- **For FedAvg-type workflows** -- The aggregator is in the FL Server, and the final result is stored in the Job Store, which is a local disk or volume accessible to the FL server
- **For Swarm learning algorithm** -- The aggregator is randomly selected at each round among all participating sites, and the final result is stored locally at that site, not shared with the FL Server

In both cases, the aggregator can be a locked-down machine using current security and privacy best practices. Confidential Computing is the latest technology that adds an extra layer of security.

What's the IRB number?
----------------------

The IRB process is outside the scope of NVIDIA FLARE; consortium participants will need to handle this separately.

What do we need to install locally?
------------------------------------

FLARE involves the following steps for local installations:

1. **Provision** -- A process to generate software packages and certificates for each participant (called startup kit)
2. **Distribution** -- Send the startup kit to each participant
3. **Start** -- Participants can pip install nvflare and start the startup kit

FLARE offers two ways to provision:

- **nvflare provision CLI command** -- The package is generated locally, the project administrator will then distribute the package manually (sftp, email etc.)
- **FLARE Dashboard** -- The web interface allows the project administrator to invite others to join the project and provide site-specific information themselves

For details, see :ref:`installation` and :ref:`deployment_overview`.

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

The software is distributed under the Apache 2.0 License, which is a permissive open-source license. Under this license, the software is provided "as is" without warranties or liabilities.

If formal support or indemnification is required, it can be obtained through a third-party service provider that offers commercial support for Apache 2.0-licensed software.

Who owns the data after it leaves the institution?
--------------------------------------------------

No raw data ever leaves any institution. Only the model weights are transmitted.

The model trained can be owned by different participants depending on the collaboration agreement. NVIDIA FLARE is not involved in these business decisions.

Is there a data use agreement?
-------------------------------

Since the data never leaves the institution, usually there is no specific data use agreement. However, collaborators usually need to decide what data to use to jointly train a model.

Is the code proxy-aware?
------------------------

Yes, NVIDIA FLARE can operate through network topologies with proxies, such as reverse-proxies or Kubernetes ingress services.

FLARE supports two types of TLS protocols: mutual TLS and standard TLS.

- **For mutual TLS** -- The certificate termination point is at the server, not at the proxy. The proxy must be configured to enable TCP pass-through
- **For standard TLS** -- Users can simply use the pre-authorized certificate for TLS handshaking and the FLARE certificate for authentication

For details, see :ref:`Communication Security <communication_security>` and :ref:`Server Port Consolidation <server_port_consolidation>`.
