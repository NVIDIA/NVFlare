***********************
Terminologies and Roles
***********************

Terminologies
=============
For establishing background knowledge, here are a few terms.

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
component is only needed for high availability.

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
