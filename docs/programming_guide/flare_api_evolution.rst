.. _api_evolution:

########################
Evolution of FLARE APIs
########################


Evolution of FLARE Server Side API
==================================

.. image:: ../resources/server_side_apis.jpg
    :height: 400


Controller
----------

In NVIDIA FLARE, the Controller is the server-side component that orchestrates federated learning workflows. It defines
the server logic of the FL algorithm, assigns tasks to clients, and processes returned results to drive the overall
training or evaluation process. Clients pull tasks from the Controller and push results back using a task-based interaction model.

**Controller API**

The Controller API provides the abstractions for implementing federated workflows on the server. It is built around a task abstraction,
where each task represents a unit of client work with an associated payload. Clients periodically request tasks, execute them locally,
and return results to the Controller, enabling flexible and programmable coordination of federated computation.

Model Controller
----------------

In NVIDIA FLARE, the ModelController is a specialized Controller that manages model-centric federated learning workflows.

It uses the FLModel data structure, which wraps the model state, parameters, and metadata. By providing a structured
interface instead of a general dictionary, FLModel makes it easier for data scientists to work with models, reducing errors
and simplifying workflow implementation while still supporting necessary flexibility for federated learning.

This API design reflects an earlier effort to separate FL algorithm logic from underlying communication handling, making it
easier for users to focus on modeling and analytics without worrying about low-level message passing or orchestration details.


Collaborative API (Collab API)
------------------------------

The Collaborative API achieves a complete separation between communication handling and federated learning algorithms.
This design gives users greater freedom from framework constraints, allowing them to implement algorithms and analytics logic without
having to learn FLARE framework concepts. In the Collab API, users decide what kind of message or data structure to pass around.


Evolution of Client Side APIs
=============================

.. image:: ../resources/client_side_apis.jpg
    :height: 400


Executor
--------
The FLARE Executor API is a low-level integration interface that allows users to directly participate in federated workflows
while retaining full control over client-side execution. It is responsible for receiving tasks from the server,
interacting with the FLARE runtime, and returning results. This API offers maximum flexibility and extensibility,
making it suitable for system integration, custom protocols, non-standard workflows, or tight coupling with external systems.
However, it also requires a deep understanding of FLARE-specific concepts such as Executors, task lifecycle, Shareable,
and FLContext, and requires users to structure their logic around FLARE’s execution model.

As a result, the Client-side API Executor is most appropriate for advanced users who need fine-grained control and are
willing to trade simplicity for customization.


Learner Executor & Learner
--------------------------
The LearnerExecutor and Learner pattern is a mid-level client-side abstraction in NVIDIA FLARE that separates federated
orchestration from machine learning logic while preserving FLARE’s Trainer-style execution model. The LearnerExecutor
manages FLARE-specific concerns—task dispatch, execution context, and runtime communication—while delegating all ML
computation to a Learner, which encapsulates framework-specific logic such as training, evaluation, and update handling.
However, this pattern still requires users to learn FLARE-specific concepts such as Shareable and FLContext, and to place
their code within predefined method structures dictated by the Executor lifecycle. As a result, while cleaner than raw
executor implementations, it retains framework constraints and learning overhead compared to higher-level APIs that fully
abstract the execution model.


FLARE Client API
----------------

The Client API is a high-level API built on top of the FLModel abstraction, where all federated communication is performed
through a single, constrained data structure. The FLModel contains only model weights, optimizer parameters, metrics,
and lightweight metadata, and does not expose any low-level communication or execution details. This design makes it easy
for data scientists to understand exactly what information is being passed between client and server.

The Client API is designed to simplify the conversion of existing ML/DL code into federated workloads with minimal code changes.
Unlike the Learner API, users adapt their existing training or analytics code in place, without restructuring it.

As a result, users do not need to learn FLARE-specific framework concepts such as Executors, Controllers, Shareable, or FLContext.
Aside from understanding the FLModel data structure itself, the Client API allows users to focus on ML logic while FLARE
transparently handles federated communication and orchestration.



Client API + Collaborative API (Collab API)
-------------------------------------------

The Collaborative API achieves a complete separation between communication handling and federated learning algorithms,
eliminating the need to use the FLModel structure, which limits some FL algorithm implementations. This design gives
users greater freedom from framework constraints, allowing them to implement algorithms and analytics logic without
having to learn many FLARE framework concepts.

With Client API + Collab API, users are free to continue passing FLModel around or choose any other data structure.


Evolution of Client Server Wiring APIs
======================================

For a federated learning system to function, the client-side Executor and server-side Controller must be properly
connected. In FLARE, this wiring is handled through system- and job-level configurations, which works well for custom
component plugins and system integration developers. However, this approach can be cumbersome for data scientists.
To address this, FLARE has introduced several higher-level APIs and approaches designed to simplify client-server
integration and reduce the setup effort for typical federated learning and analytics tasks.

.. image:: ../resources/client_server_wiring_apis.jpg
    :height: 400


Json Configurations
--------------------

The FLARE Job is defined by three configuration files:

.. code-block:: text

    config_fed_server.json
    config_fed_client.json
    meta.json

There is no need to dive deep into the specific content of the JSON files.
The JSON file format gives the system a way to define dynamic plugin custom components.


Alternative Configurations support: YAML (OmegaConf), pyhocon
-------------------------------------------------------------
We support both YAML and Pyhocon (Python HOCON) configuration formats, each allowing comments and variable substitution:
**Pyhocon** – A JSON variant and HOCON (Human-Optimized Config Object Notation) parser for Python, supporting comments, variable substitution, and inheritance.
**OmegaConf** – A YAML-based hierarchical configuration system, also supporting comments and variable substitution.
Users can work with a single format or combine multiple formats—for example, config_fed_client.conf and config_fed_server.json.


Job Template & Job CLI
----------------------
A FLARE Job Template is a predefined set of job configurations in NVIDIA FLARE. It defines the model, training strategy,
and client/server settings, enabling new federated learning jobs to be copied and modified without rewriting them from scratch.
To simplify working with these templates, we also provide the FLARE Job CLI, which allows users to list available templates,
create jobs from templates, inspect template variables, and submit a job.
This approach represents an early step toward automating job scripting.


Job API
--------

The FLARE Job API is a Python interface that allows users to define components, wire client-server connections, and specify
federated learning workflows, typically expressed in JSON configurations. Users can generate job configurations directly
from the Job API by exporting the workflow definitions to JSON. This provides a precise and programmatic way to describe
and define FL workflows using Python.


Job Recipe & Runtime Environments
---------------------------------

**Job Recipe**

A Job Recipe in NVIDIA FLARE defines the runtime logic and workflow for a federated learning job. It specifies how components such as models, trainers, and aggregators interact, the sequence of operations during training and evaluation, and any special procedures (e.g., validation, early stopping). Essentially, it encodes the behavior of a job, separate from its configuration, so that the same recipe can be reused with different datasets or clients.

**Runtime Environment**

The Runtime Environment describes the execution context for a FLARE job. It includes system-level settings, software dependencies, Python packages, hardware requirements, and communication protocols needed to run the job on clients and servers. By defining a runtime environment, FLARE ensures consistency and reproducibility across heterogeneous devices and platforms.






