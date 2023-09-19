.. _nvflare_security:

****************************************
NVIDIA FLARE Security
****************************************

The security framework of NVIDIA FLARE 2.2 has been reworked for better usability and to improve security.

Terminologies
=============
For the ease of discussion, we'll start by defining a few terms.

Project
-------
An FL study with identified participants.

Org 
---
An organization that participates in the study.

Site
----
The computing system that runs NVFLARE application as part of the study.
There are two kinds of sites: Server and Clients.
Each site belongs to an organization.

FL Server
------------
An application running on a Server site responsible for client coordination based on federation workflows. There can be
one or more FL Servers for each project.

FL Client
----------
An application running on a client site that responds to Server's task assignments and performs learning actions based
on its local data.

Overseer
----------
An application responsible for overseeing overall system health and enabling seamless failover of FL servers. This
component is only needed for High Available.

User
-----
A human that participates in the FL project.

.. _nvflare_roles:

Role
------
A role defines a type of users that have certain privileges of system operations. Each user is assigned a role in the
project. There are four defined roles: Project Admin, Org Admin, Lead Researcher, and Member Researcher.

.. _project_admin_role:

Project Admin Role
^^^^^^^^^^^^^^^^^^^^
The Project Admin is responsible for provisioning the participants and coordinating personnel from all sites for the project.
When using the Dashboard UI, the Project Admin is the administrator for the site and is responsible for inputting the
values to set up the project in the beginning and then approving the users and client sites while making edits if necessary.

The Project Admin is also responsible for the management of the FL Server.

There is only one Project Admin for each project.

Org Admin Role
^^^^^^^^^^^^^^^^^^^^
This role is responsible for the management of the sites of his/her organization.

Lead Researcher Role
^^^^^^^^^^^^^^^^^^^^^^^
This role can be configured for increased privileges for an organization for a scientist who works
with other researchers to ensure the success of the project.

Member Researcher Role
^^^^^^^^^^^^^^^^^^^^^^^
This role can be configured for another level of privileges a scientist who works with the Lead Researcher
to make sure his/her site is properly prepared for the project.


FLARE Console (previously called Admin Client)
----------------------------------------------
An console application running on a user's machine that allows the user to perform NVFLARE system operations with a
command line interface.

Provisioning Tool
-----------------
The tool used by Project Admin to provision all participating sites and users of the project. The output of the
Provisioning tool enables all participants (sites and users) to securely communicate with each other.

Security Framework
===================
NVFLARE is an application running in the IT environment of each participating site. The total security of this
application is the combination of the security measures implemented in this application and the security measures of
the site's IT infrastructure.

NVFLARE implements security measures in the following areas:

    - Identity Security:  the authentication and authorization of communicating parties
    - Communication Security: the confidentiality of data communication messages.
    - Message Serialization: techniques for ensuring safe serialization/deserialization process between communicating parties
    - Data Privacy Protection: techniques for preventing local data from being leaked and/or reverse-engineered.
    - Auditing: techniques for keep audit trails of critical events (e.g. commands issued by users, learning/training related events that can be analyzed to understand the final results)

All other security concerns must be handled by the site's IT security infrastructure. These include, but are not limited to:

    - Physical security
    - Firewall policies
    - Data management policies: storage, retention, cleaning, distributions, access, etc.

Security Trust Boundary and Balance of Risk & Usability
---------------------------------------------------------
The security framework does not operate in vacuum, we assume the physical security is already in place for all
participating server and client machines. TLS provides the authentication mechanism within the trusted environments.

Under such circumstances, we trade off some of the security risk with ease of use when transferring data between client
and server in previous releases. The python pickle was used in NVFLARE 2.0. This trade-off caused some concern due to
the use of Pickle. To address such as concern, we replaced python pickle with Flare Object Serializer (FOBS).  See
:ref:`serialization <serialization>` for details.

Identity Security
------------------
This area is concerned with these two trust issues:

    - Authentication: ensures communicating parties have enough confidence about each other's identities – everyone is who they claim to be.
    - Authorization: ensures that the user can only do what he/she is authorized to do.
 
Authentication
^^^^^^^^^^^^^^^
NVFLARE's authentication model is based on Public Key Infrastructure (PKI) technology:

    - For the FL project, the Project Admin uses the Provisioning Tool to create a Root CA with a self-signed root certificate. This Root CA will be used to issue all other certs needed by communicating parties.
    - Identities involved in the study (Server(s), Clients, the Overseer, Users) are provisioned with the Provisioning Tool. Each identity is defined with a unique common name. For each identity, the Provisioning Tool generates a separate password-protected Startup Kit, which includes security credentials for mutual TLS authentication:
        - The certificate of the Root CA
        - The cert of the identity
        - The private key of the identity
    - Startup Kits are distributed to the intended identities:
        - The FL Server's kit is sent to the Project Admin
        - The kit for each FL Client is sent to the Org Admin responsible for the site
        - FLARE Console (previously called Admin Client) kits are sent to the user(s)
    - To ensure the integrity of the Startup Kit, each file in the kit is signed by the Root CA.
    - Each Startup Kit also contains a "start.sh" file, which can be used to properly start the NVFLARE application.
    - Once started, the Client tries to establish a mutually-authenticated TLS connection with the Server, using the PKI credentials in its Startup Kits. This is possible only if the client and the server both have the correct Startup Kits.
    - Similarly, when a user tries to operate the NVFLARE system with the Admin Client app, the admin client tries to establish a mutually-authenticated TLS connection with the Server, using the PKI credentials in its Startup Kits. This is possible only if the admin client and the server both have the correct Startup Kits. The admin user also must enter his/her assigned user name correctly.
 
The security of the system comes from the PKI credentials in the Startup Kits. As you can see, this mechanism involves manual processing and human interactions for Startup Kit distribution, and hence the identity security of the system depends on the trust of the involved people. To minimize security risk, we recommend that people involved follow these best practice guidelines:

    - The Project Admin, who is responsible for the provisioning process of the study, should protect the study's configuration files and store created Startup Kits securely.
    - When distributing Startup Kits, the Project Admin should use trusted communication methods, and never send passwords of the Startup Kits in the same communication. It is preferred to send the Kits and passwords with different communication methods.
    - Org Admin and users must protect their Startup Kits and only use them for intended purposes.
 
.. note::

    The provisioning tool tries to use the strongest cryptography suites possible when generating the PKI credentials. All of the certificates are compliant with the X.509 standard. All private keys are generated with a size of 2048-bits. The backend is openssl 1.1.1f, released on March 31, 2020, with no known CVE.  All certificates expire within 360 days.
 
.. note::

    NVFLARE 2.2 implements a :ref:`website <nvflare_dashboard_ui>` that supports user and site registration. Users will be able to download their Startup Kits (and other artifacts) from the website.

Authorization
^^^^^^^^^^^^^^
See :ref:`Federated Authorization <federated_authorization>`
 
Communication Security
-----------------------
All data communications are through secure channels established with mutually-authenticated TLS connections. The
communication protocol between the FL Server and clients is gRPC. The protocol between FLARE Console instances and the
FL Server is TCP.
 
NVIDIA FLARE uses client-server communication architecture.  The FL Server accepts connection requests from clients.
Clients never need to accept connection requests from anywhere.
 
The IT infrastructure of the FL Server site must allow two ports to be opened: one for the FL Server to communicate with
FL Clients, and one for the FL Server to communicate with FLARE Console instances. Both ports should be unprivileged.
Specifically, we suggest against the use of port 443, the typical port number for HTTPS. This is because gRPC does
not exactly implement HTTPS to the letter, and the firewall of some sites may decide to block it.

The IT infrastructure of FL Client sites must allow the FL application to connect to the address (domain and port)
opened by the FL server.

Enhanced Message Serialization
-------------------------------
Prior to NVFLARE 2.1, messages between the FL server and clients were serialized with Python's pickle facility. Many people
have pointed out the potential security risks due to the flexibility of Pickle.

NVFLARE now uses a more secure mechanism called FOBS (Flare OBject Serializer) for message serialization and
deserialization. See :ref:`serialization <serialization>` for details.

Enhanced Auditing
-------------------
Prior to NVFLARE 2.2, the audit trail only includes user command events (on both server and client sites). NVFLARE 2.2
enhances the audit trail by including critical job events generated by the learning process.

Audit File Location
^^^^^^^^^^^^^^^^^^^^
The audit file audit.txt is located in the root directory of the workspace.

Audit File Format
^^^^^^^^^^^^^^^^^^
The audit file is a text file. Each line in the file is an event. Each event contains headers and an optional message.
Event headers are enclosed in square brackets. The following are some examples of events:

.. code-block::

    [E:b6ac4a2a-eb01-4123-b898-758f20dc028d][T:2022-09-13 13:56:01.280558][U:?][A:_cert_login admin@b.org]
    [E:16392ed4-d6c7-490a-a84b-12685297e912][T:2022-09-1412:59:47.691957][U:trainer@b.org][A:train.deploy]
    [E:636ee230-3534-45a2-9689-d0ec6c90ed45][R:9dbf4179-991b-4d67-be2f-8e4bac1b8eb2][T:2022-09-14 15:08:33.181712][J:c4886aa3-9547-4ba7-902e-eb5e52085bc2][A:train#39027d22-3c70-4438-9c6b-637c380b8669]received task from server

Event Headers
^^^^^^^^^^^^^^^^^^
Event headers specify meta information about the event. Each header is expressed with the header type (one character),
followed by a colon (:) and the value of the header. The following are defined header types and their values.

.. csv-table::
    :header: Checks,Meaning,Value
    :widths: 5, 10, 20

    E,Event ID,A UUID
    T,Timestamp,Time of the event
    U,User,Name of the user
    A,Action,User issued command or job's task name and ID
    J,Job,ID of the job
    R,Reference,Reference to peer's event ID

Most of the headers are self-explanatory, except for the R header. Events can be related. For example, a user command
could cause an event to be recorded on both the server and clients. Similarly, a client's action could cause the server
to act on it (e.g. client submitting task results). The R header records the related event ID on the peer. Reference
event IDs can help to correlate events across the system.

Data Privacy Protection
-------------------------
Federated learning activities are performed with task-based interactions between the server and FL clients: the server
issues tasks to the clients, and clients process tasks and return results back to the server. NVFLARE comes with a
general-purpose data filtering mechanism for processing task data and results:

    - On the Server: before task data is sent to the client, the configured "task_data_filters" defined in the job are executed;
    - On the Client: when the task data is received by the client and before giving it to the executor for processing, NVFLARE framework applies configured "task_data_filters" defined in the job;
    - On the Client: after the execution of the task by the executor and before sending the produced result back to the server, NVFLARE framework applies configured "task_result_filters" to the result before sending to the Server.
    - On the Server: after receiving the task result from the client, the NVFLARE framework applies configured "task_result_filters" before giving it to the Controller for processing.

This mechanism has been used for the purpose of data privacy protection on the client side. For example, differential
privacy filters can be applied to model weights before sending to the server for aggregation.

NVFLARE has implemented some commonly used privacy protection filters: https://github.com/NVIDIA/NVFlare/tree/main/nvflare/app_common/filters

Admin Capabilities
-------------------
The NVFLARE system is operated by users using the command line interface provided by the admin client. The following
types of commands are available:

    - Check system operating status
    - View system logs
    - Shutdown, restart server or clients
    - Job management (submit, clone, stop, delete, etc.)
    - Start, stop jobs
    - Clean up job workspaces
 
All admin commands are subject to authorization policies of the participating sites.

Dynamic Additions and Users and Sites
--------------------------------------
Federated Authorization makes it possible to dynamically add new users and sites without requiring the server to
always keep an up-to-date list of users and sites. This is because the user identity information (name, org, and role)
is included in the certificate of the user; and each site now performs authorization based on its local policies
(instead of the FL Server performing authorization for all sites).

Site Policy Management
------------------------
Prior to NVFLARE 2.2, all policies (resource management, authorization and privacy protection) could only be centrally
controlled by the FL Server. NVFLARE 2.2 made it possible for each site to define end enforce its own policies.

See :ref:`site policy management <site_policy_management>`.
