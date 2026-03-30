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
     - TLS/mTLS encryption, explicit message authentication, BYO connectivity, proxy support, and secure serialization (FOBS)
     - :ref:`Communication Security <communication_security>`
   * - **Data Privacy & Filters**
     - Privacy-preserving filters (differential privacy, homomorphic encryption), per-job and per-site filter chains
     - :ref:`Data Privacy Protection <data_privacy_protection>`
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


For common questions about data storage, infrastructure, and compliance, see :ref:`Security FAQ <security_faq>`.
