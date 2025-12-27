.. _diff_api_guide:

########################################
How to choose which APIs to use in FLARE
########################################


When getting started with NVIDIA FLARE, one of the first decisions new users face is choosing which APIs to use.
FLARE provides multiple API layers that have evolved over time to support different levels of abstraction—from high-level,
ready-to-use workflows for common federated learning and analytics tasks, to lower-level, highly customizable APIs
for advanced control over execution, orchestration, and security. These APIs are reflected across different FLARE examples,
which can sometimes be confusing for new users when deciding where to start. This guide helps clarify the evolution of
FLARE APIs and provides guidance on selecting the most appropriate API for your use case and development goals.

The newer FLARE APIs—Client API, Job Recipe API, and Collaborative API (released soon) —represent the latest stage in the
evolution of the platform. They are designed primarily for data scientists and researchers, providing a simplified and
intuitive interface that is sufficient for most common federated learning and federated analytics use cases. In contrast,
the Controller/Executor APIs operate at a lower level and are intended for system integration, advanced customization,
and platform-level extensions, where fine-grained control over execution flow, policies, and orchestration is required.


Evolution of FLARE APIs
=======================

You can read the overall evolution of FLARE APIs in :ref:`api_evolution`

**Server-side APIs**

.. image:: ../../resources/server_side_apis.jpg
    :height: 400


**Client-side APIs**

.. image:: ../../resources/client_side_apis.jpg
    :height: 400

**Client-Server Wiring APIs**

.. image:: ../../resources/client_server_wiring_apis.jpg
    :height: 400


Which APIs to use ?
===================

We recommend to use the following APIs depends on your roles

**Applied Data Scientists**
---------------------------

- Client: Client API
- Server: choose a builtin algorithm
- Client-Server wiring: Job Recipe with builtin FL algorithms


**FL Researchers**
---------------------------

- Client: Collab API, Client API
- Server: Collab API
- Client-Server wiring: Job Recipe

**System Integrator**
---------------------------
- Client: Collab API, Executor API
- Server: Collab API, Controller API
- Client-Server wiring: Job Recipe


Deprecated APIs
================

- LearnerExecutor and Learner
- ModelController (once Collab API is released)
- Job Template & CLI Job Template API


API References
================

- Client API: :ref:`client_api`
- Server API: :ref:`collab_api`
- client-server wiring: :ref:`job_recipe`

