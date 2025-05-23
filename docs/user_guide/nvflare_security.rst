.. _nvflare_security:

****************************************
NVIDIA FLARE Security
****************************************

The security framework of NVIDIA FLARE has been reworked for better usability and to improve security.

Security Framework
===================
NVFLARE is an application running in the IT environment of each participating site. The total security of this
application is the combination of the security measures implemented in this application and the security measures of
the site's IT infrastructure.

NVFLARE implements security measures in the following areas (see each section below for details):

    - Identity Security: the authentication and authorization of communicating parties
    - Site Policy Management: the policies for resource management, authorization, and privacy protection defined by each site
    - Communication Security: the confidentiality of data communication messages
    - Message Serialization: techniques for ensuring safe serialization/deserialization process between communicating parties
    - Data Privacy Protection: techniques for preventing local data from being leaked and/or reverse-engineered
    - Auditing: techniques for keeping audit trails to record events (e.g. commands issued by users, learning/training related events that can be analyzed to understand the final results)

.. toctree::
   :maxdepth: 1

   security/terminologies_and_roles
   security/identity_security
   security/site_policy_management
   security/authorization_policy_previewer
   security/communication_security
   security/serialization
   security/data_privacy_protection
   security/auditing
   security/unsafe_component_detection

All other security concerns must be handled by the site's IT security infrastructure. These include, but are not limited to:

    - Physical security
    - Firewall policies
    - Data management policies: storage, retention, cleaning, distribution, access, etc.

Security Trust Boundary and Balance of Risk and Usability
---------------------------------------------------------
The security framework does not operate in vacuum; we assume that physical security is already in place for all
participating server and client machines. TLS provides the authentication mechanism within the trusted environments.


Admin Capabilities Through FLARE Console
----------------------------------------
The NVFLARE system is operated by users using the command line interface provided by the :ref:`FLARE Console <operating_nvflare>`. The following
types of commands are available:

    - Check system operating status
    - View system logs
    - Shutdown, restart server or clients
    - Job management (submit, clone, stop, delete, etc.)
    - Start, stop jobs
    - Clean up job workspaces
 
All commands are subject to authorization policies of the participating sites.

Dynamic Additions and Users and Sites
--------------------------------------
Federated Authorization makes it possible to dynamically add new users and sites without requiring the server to
always keep an up-to-date list of users and sites. This is because the user identity information (name, org, and role)
is included in the certificate of the user; and each site now performs authorization based on its local policies
(instead of the FL Server performing authorization for all sites).
