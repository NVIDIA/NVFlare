.. _glossary:

########
Glossary
########
Below is a list of terms and concepts in NVIDIA FLARE and their definitions.

Aggregator
==========
The :ref:`aggregator` defines the algorithm used on the server to aggregate the data passed back to the server in the
clients' Shareable object.

Application (app)
=================
An :ref:`Application <application>` is a named directory structure that defines the client and server configuration
and any custom code required to implement the Controller/Worker workflow. Since 2.1.0, :ref:`jobs <job>` have been
introduced to manage the deployment of apps in an experiment.

Controller
==========
The :ref:`Controller <controllers>` is a python object on the FL server side that controls or coordinates Workers
to perform tasks.  The Controller defines the overall collaborative computing workflow.  In its
control logic, the Controller assigns tasks to Workers and processes task results from the workers.

Events
======
:ref:`Events <event_system>` allow for dynamic notifications to be sent to all objects that are a subclass of
:ref:`FLComponent <fl_component>`. Every FLComponent is an event handler. The event mechanism is like a pub-sub mechanism
that enables indirect communication between components for data sharing.

Executor
========
The :ref:`executor` is the component on the FL Client side that executes the task received from the Controller on the FL Server. 
For example, in DL training, the :ref:`executor` would implement the training loop. There can be multiple executors on the
client, designed to execute different tasks (training, validation/evaluation, data preparation, etc.).

Filter
======
:ref:`Filters <filters>` are used to define transformations of the data in the Shareable object when transferred between server
and client and vice versa.  Filters can be applied when the data is sent or received by either the client or server.

FLAdminAPI
==========
:class:`FLAdminAPI<nvflare.fuel.hci.client.fl_admin_api.FLAdminAPI>` is a wrapper for admin commands that can be issued
by an admin client to the FL server. You can use a provisioned admin client's certs and keys to initialize an instance
of FLAdminAPI to programmatically submit commands to the FL server. :ref:`flare_api` is a redesigned version of this introduced
in version 2.3.0.

FLARE API
=========
:ref:`flare_api` is a redesigned version of FLAdminAPI intended to provide a better user experience for
issuing admin commands to the FL server.

FLARE Console (previously referred to as Admin Console or Admin Client)
=======================================================================
The :ref:`FLARE Console <operating_nvflare>` is used to orchestrate the FL study, including starting and stopping the server
and clients and checking their status, deploying applications, and managing FL experiments.

FLComponent
===========
Most component types are subclasses of :ref:`FLComponent <fl_component>`. You can create your own subclass of
FLComponent for various purposes like listening to certain events and handling data.

FLContext
=========
:ref:`FLContext <fl_context>` is one of the key features of NVIDIA FLARE and is available to every method of all :ref:`FLComponent <fl_component>`
types (Controller, Aggregator, Executor, Filter, Widget, ...). An FLContext object contains contextual information
of the FL environment: overall system settings (peer name, job id / run number, workspace location, etc.). FLContext
also contains an important object called Engine, through which you can access important services provided by the
system (e.g. fire events, get all available client names, send aux messages, etc.).

HA
====
:ref:`high_availability` is a feature implemented in NVIDIA FLARE 2.1.0 around FL server failover introducing an Overseer
to coordinate multiple FL servers.

Job
===
:ref:`Jobs <job>` contain all of the apps and the information of which app(s) to deploy to which clients or server, the resource requirements
for the experiment, and everything about the experiment.

Learnable
=========
Learnable is the result of the Federated Learning application maintained by the server.  In DL workflows, the
Learnable is the aspect of the DL model to be learned.  For example, the model weights are commonly the Learnable
feature, not the model geometry.  Depending on the purpose of your study, the Learnable may be any component of interest.
Learnable is an abstract object that is aggregated from the client's Shareable object and is not DL-specific.  It
can be any model, or object.  The Learnable is managed in the Controller workflow.

Learner
=======
:class:`Learner <nvflare.app_common.abstract.learner_spec.Learner>` is a class that focuses just on the training specific tasks, that 
can be used for building a component to use with :class:`LearnerExecutor <nvflare.app_common.executors.learner_executor.LearnerExecutor>`.
This way, the communication constructs, error code handling, etc. that are specific to NVFLARE are handled in LearnerExecutor while 
the training and validation logic can be focused in Learner.

LearnerExecutor
===============
:class:`LearnerExecutor <nvflare.app_common.executors.learner_executor.LearnerExecutor>` is a special type of Executor that abstracts the
execution flow and delegates the actual training work to the :class:`Learner <nvflare.app_common.abstract.learner_spec.Learner>`.

ModelLocator
============
:class:`nvflare.app_common.np.np_model_locator.NPModelLocator` is a component to find the models to be included for cross site
evaluation located on server.

NVIDIA FLARE
============
NVIDIA FLARE stands for NVIDIA Federated Learning Application Runtime Environment, a general-purpose framework designed for
collaborative computing.

Overseer
========
The overseer is a subsystem that monitors the FL servers in :ref:`HA mode <high_availability>` and tells clients which FL
server to connect to. This is only applicable in HA mode.

Peer Context
============
The :ref:`Peer Context <peer_context>` is the contextual information of the message sender that is sent in addition to the
regular payload of the message when the FL parties communicate with each other.

Persistor
=========
A component for saving the state of something. :class:`LearnablePersistor<nvflare.app_common.abstract.learnable_persistor.LearnablePersistor>`
is a method implemented for the FL server to save the state of the Learnable object, for example writing a global model to disk.

POC mode
========
See :ref:`setting_up_poc`.

Project yaml
============
The :ref:`project.yaml <project_yml>` is the file used in the provisioning process that has the Project's specifications
including the FL Server, FL Clients, and Admin Users as well as the :ref:`Builders <bundled_builders>` for assembling the Startup Kits.

Provisioning
============
:ref:`Provisioning <provisioning>` is the process of setting up a secure project with startup kits for the different
participants including the FL Server, FL Clients, and Admin Users.

Roles in NVIDIA FLARE
=====================
The :ref:`user roles <nvflare_roles>` in NVIDIA FLARE include :ref:`Project Admin <project_admin_role>`, Org Admin, Lead researcher,
and Member researcher and can be used to set certain privileges of system operations for different users. See the :ref:`nvflare_security`
page for details.

Scatter and Gather Workflow
===========================
The :ref:`scatter_and_gather_workflow` is an included reference implementation of the default workflow of previous versions
of NVIDIA FLARE with an FL Server aggregating results from FL Clients.

Shareable
=========
:ref:`Shareable <shareable>` is a communication between two peers (server and clients). In the task-based
interaction, the Shareable from server to clients carries the data of the task for the client to execute; and the
Shareable from the client to server carries the result of the task execution.  When this is applied to DL model
training, the task data typically contains model weights for the client to train on; and the task result contains
updated model weights from the client.  The concept of Shareable is very general - it can be whatever that makes
sense for the task.

ShareableGenerator
==================
.. currentmodule:: nvflare.app_common.abstract.shareable_generator.ShareableGenerator

:class:`ShareableGenerator <nvflare.app_common.abstract.shareable_generator.ShareableGenerator>` is a
component that converts between Shareable objects and model objects. The ShareableGenerator implements two methods,
:meth:`learnable_to_shareable` converts a Learnable object to a form of data to be shared to FL clients, and
:meth:`shareable_to_learnable` uses the Shareable data (or aggregated Shareable data) from the FL clients to update the Learnable object.

Startup kit
===========
Startup kits are products of the provisioning process and contain the configuration and certificates necessary to establish
secure connections between the Overseer, FL servers, FL clients, and Admin clients. These files are used to establish identity
and authorization policies between server and clients.  Startup kits are distributed to the Overseer, FL servers, clients,
and Admin clients depending on role.

Task
====
A :ref:`Task <tasks>` is a piece of work (Python code) that is assigned by the :ref:`Controller <controllers>` to
client workers. Depending on how the Task is assigned (broadcast, send, or relay), the task will be performed by one
or more clients.  The logic to be performed in a Task is defined in an :ref:`Executor <executor>`.

TB Analytics Receiver
=====================
The Tensorboard Analytics Receiver is part of the ML Experimental tracking. NVFLARE implemented the
server-side ML Experimental tracking, with Tensorboard as the ML tracking tool. The client side collects the
logs, and the FL server has the Tensorboard Summary Writer to send the logs to Tensorboard. The TB Analytics Receiver is the component
that receives the logging from different clients and then writes to Tensorboard.

Worker
======
A Worker is capable of performing tasks (training, validation/evaluation, data preparation, etc.). Workers run on FL Clients.
