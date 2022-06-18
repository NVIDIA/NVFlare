.. _FAQ:

##########################
Frequently Asked Questions
##########################

*******
General
*******

#. What is NVIDIA FLARE?

    NVIDIA FLARE is a general-purpose framework designed for collaborative computing.  In this collaborative
    computing framework, workflows are not limited to aggregation-based federated learning (usually called a Fed-Average workflow),
    and applications are not limited to deep learning.  NVIDIA FLARE is fundamentally a messaging system running in a multithreaded
    environment.

#. What does NVIDIA FLARE stand for?

    NVIDIA Federated Learning Application Runtime Environment.

#. Does NVIDIA FLARE depend on Tensorflow or PyTorch?

    No.  NVIDIA FLARE is a Python library that implements a general collaborative computing framework.  The :ref:`Controllers <controllers>`,
    :ref:`Executors <executor>`, and :ref:`Tasks <tasks>` that one defines to execute the collaborative computing workflow
    are entirely independent.

#. Is NVIDIA FLARE designed for deep learning model training only?

    No.  NVIDIA FLARE implements a communication framework that can support any collaborative computing workflow.  This
    could be deep learning, machine learning, or even simple statistical workflows.

#. Does NVIDIA FLARE require a GPU?

    No.  Hardware requirements are dependent only on what is implemented in the :ref:`Controller <controllers>` workflow and client :ref:`Tasks <tasks>`.
    Client training tasks will typically benefit from GPU acceleration.  Server :ref:`Controller <controllers>` workflows may or may not require a GPU.

#. How does NVIDIA FLARE implement its collaborative computing framework?

    NVIDIA FLARE collaborative computing is achieved through :ref:`Controller/Worker <controllers>` interaction.

#. What is a Controller?

    The :ref:`Controller <controllers>` is a python object that controls or coordinates Workers to perform tasks. The
    Controller is run on the server.  The Controller defines the overall collaborative computing workflow.  In its
    control logic, the Controller assigns tasks to Workers and processes task results from the workers.

#. What is a Worker?

    A Worker is capable of performing tasks (skills). Workers run on Clients.

#. What is a Task?

    A :ref:`Task <tasks>` is a piece of work (Python code) that is assigned by the :ref:`Controller <controllers>` to
    client workers. Depending on how the Task is assigned (broadcast, send, or relay), the task will be performed by one
    or more clients.  The logic to be performed in a Task is defined in an :ref:`Executor <executor>`.

#. What is Learnable?

    Learnable is the result of the Federated Learning application maintained by the server.  In DL workflows, the
    Learnable is the aspect of the DL model to be learned.  For example, the model weights are commonly the Learnable
    feature, not the model geometry.  Depending on the purpose of your study, the Learnable may be any component of interest.
    Learnable is an abstract object that is aggregated from the client's Shareable object and is not DL-specific.  It
    can be any model, or object.  The Learnable is managed in the Controller workflow.

#. What is Shareable?

    :ref:`Shareable <shareable>` is simply a communication between two peers (server and clients). In the task-based
    interaction, the Shareable from server to clients carries the data of the task for the client to execute; and the
    Shareable from the client to server carries the result of the task execution.  When this is applied to DL model
    training, the task data typically contains model weights for the client to train on; and the task result contains
    updated model weights from the client.  The concept of Shareable is very general - it can be whatever that makes
    sense for the task.

#. What is FLContext and what kind of information does it contain?

    :ref:`FLContext <fl_context>` is one of the key features of NVIDIA FLARE and is available to every method of all :ref:`FLComponent <fl_component>`
    types (Controller, Aggregator, Executor, Filter, Widget, ...). An FLContext object contains contextual information
    of the FL environment: overall system settings (peer name, job id / run number, workspace location, etc.). FLContext
    also contains an important object called Engine, through which you can access important services provided by the
    system (e.g. fire events, get all available client names, send aux messages, etc.).

#. What are events and how are they handled?

    :ref:`Events <event_system>` allow for dynamic notifications to be sent to all objects that are a subclass of
    :ref:`FLComponent <fl_component>`. Every FLComponent is an event handler.

    The event mechanism is like a pub-sub mechanism that enables indirect communication between components for data
    sharing. Typically, the data generator fires an event to publish the data, and other components handle the events
    they are subscribed to and consume the data of the event. The fed event mechanism even allows the pub-sub go across
    network boundaries.

#. What additional components may be implemented with NVIDIA FLARE to support the Controller Workflow, and where do they run (server or client):

    LearnablePersistor - Server
        The LearnablePersistor is a method implemented for the server to save the state of the Learnable object, for
        example writing a global model to disk for persistence.
    ShareableGenerator - Server
        The ShareableGenerator is an object that implements two methods: learnable_to_shareable converts a Learnable
        object to a form of data to be shared to the client; shareable_to_learnable uses the shareable data (or
        aggregated shareable data) from the clients to update the learnable object.
    Aggregator - Server
        The aggregator defines the algorithm used on the server to aggregate the data passed back to the server in the
        clients' Shareable object.
    Executor - Client
        The Executor defines the algorithm the clients use to operate on data contained in the Shareable object.  For
        example in DL training, the executor would implement the training loop. There can be multiple executors on the
        client, designed to execute different tasks (training, validation/evaluation, data preparation, etc.).
    Filter - Clients and Server
        :ref:`Filters <filters>` are used to define transformations of the data in the Shareable object when transferred between server
        and client and vice versa.  Filters can be applied when the data is sent or received by either the client or server.
        See the diagram on the :ref:`Filters <filters>` page for details on when "task_data_filters" and "task_result_filters"
        are applied on the client and server.
    Any component of subclass of FLComponent
        All component types discussed above are subclasses of :ref:`FLComponent <fl_component>`. You can create your own subclass of
        FLComponent for various purposes. For example, you can create such a component to listen to certain events and
        handle the data of the events (analysis, dump to disk or DB, etc.).

***********
Operational
***********

#. What is :ref:`Provisioning <provisioning>`?

    NVIDIA FLARE includes an Open Provision API that allows you to generate mutual-trusted system-wide configurations,
    or startup kits, that allow all participants to join the NVIDIA FLARE system from across different locations.  This
    mutual-trust is a mandatory feature of Open Provision API as every participant authenticates others by the
    information inside the configuration.  The configurations usually include, but are not limited to:

        - network discovery, such as domain names, port numbers or IP addresses
        - credentials for authentication, such as certificates of participants and root authority
        - authorization policy, such as roles, rights and rules
        - tamper-proof mechanism, such as signatures
        - convenient commands, such as shell scripts with default command line options to easily start an individual participant

#. What types of startup kits are generated by the Provision tool?

    The Open Provision API allows flexibility in generating startup kits, but typically the provisioning tool is used to
    generate secure startup kits for the Overseer, FL servers, FL clients, and Admin clients.

#. What files does each type of startup kit contain? What are these files used for, and by whom?

    Startup kits contain the configuration and certificates necessary to establish secure connections between the Overseer, FL servers,
    FL clients, and Admin clients.  These files are used to establish identity and authorization policies between server
    and clients.  Startup kits are distributed to the Overseer, FL servers, clients, and Admin clients depending on role.  For the
    purpose of development, startup kits may be generated with limited security to allow simplified connection between
    systems or between processes on a single host.  See the "poc" functionality of the Open Provision API for details.

#. How would you distribute the startup kits to the right people?

    Distribution of startup kits is inherently flexible and can be via email or shared storage.  The API allows the
    addition of builder components to automation distribution.

#. What happens after provisioning?

    After provisioning, the Admin API is used to submit a job to the FL server, and the JobRunner on the server can pick
    it up to deploy and run.

#. What is an Application in NVIDIA FLARE?

    An :ref:`Application <application>` is a named directory structure that defines the client and server configuration
    and any custom code required to implement the Controller/Worker workflow.

#. What is the basic directory structure of an NVIDIA FLARE Application?

    Typically the Application configuration is defined in a ``config/``
    subdirectory and defines paths to Controller and Worker executors.  Custom code can be defined in a ``custom/``
    subdirectory and is subject to rules defined in the Authorization Policy.

#. How do you deploy an application?

    An Application is deployed using the ``submit_job`` admin command. For more configuration, apps can be packaged into
    jobs with deploy_map definitions to specify which sites which apps should be deployed to. The deployment happens
    automatically with the JobRunner on the FL server.

#. Do all FL client have to use the same application configuration?

    No, they do not have to use the same application configuration, even though they can that is frequently done. The
    function of FL clients can be customized by the implementation of Tasks and Executors and the client's
    response to Events.

#. What is the difference between the Admin client and the FL client?

    The :ref:`Admin client <operating_nvflare>` is used to control the state of the server's controller workflow and only interacts with the
    server.  FL clients poll the server and perform tasks based on the state of the server.  The Admin client does not
    interact directly with FL client.

#. Where does the Admin client run?

    The :ref:`Admin client <operating_nvflare>` runs as a standalone process, typically on a researcher's workstation or laptop.

#. What can you do with the Admin client?

    The :ref:`Admin client <operating_nvflare>` is used to orchestrate the FL study, including starting and stopping server
    and clients, deploying applications, and managing FL experiments.

#. Why am I getting an error about my custom files not being found?

    Make sure that BYOC is enabled. BYOC is always enabled in POC mode, but disabled by default in secure mode when
    provisioning.  Either through the UI tool or though yml, make sure the ``enable_byoc`` flag is set for each participant.
    If the ``enable_byoc`` flag is disabled, even if you have custom code in your application folder, it will not be loaded.
    There is also a setting for ``allow_byoc`` through the authorization rule groups. This controls whether or not apps
    containing BYOC code will be allowed to be uploaded and deployed.

********
Security
********

#. What is the scope of security in NVIDIA FLARE?

    Security is multi-faceted and cannot be completely controlled for or provided by the NVIDIA FLARE API.  The Open
    Provision API provides examples of basic communication and identity security using GRPC via shared self-signed
    certificates and authorization policies.  These security measures may be sufficient but can be extended with the
    provided APIs.

#. What about data privacy?

    NVIDIA FLARE comes with a few techniques to help with data privacy during FL: differential privacy and homomorphic encryption
    (see :ref:`Privacy filters<filters_for_privacy>`).

************************
Client related questions
************************

#. What happens if an FL client joins during the FL training?

    An FL client can join the FL training any time. It is up to the workflow logic to manage FL clients.

#. Do federated learning clients need to open any ports for the FL server to reach the FL client?

    No, federated learning training does not require for FL clients to open their network for inbound traffic. The server
    never sends uninvited requests to clients but only responds to client requests.

#. Can a client train with multiple GPUs?

    You do multiple-gpu training by putting your training executor within the a :ref:`MultiProcessExecutor <multi_process_executor>`.

#. How do FL clients get identified?

    The federated learning clients are identified by a dynamically generated FL token issued by the server during runtime.
    When an FL client first joins an FL training, it first needs to send a login request to the FL server. During the login
    process, the FL server and client need to exchange SSL certificates for bi-directional authentication. Once the
    authentication is successful, the FL server sends an FL token to the client. The FL client will use this FL token to
    identify itself for all following requests for the global model and all model updating operations.

#. Can I run multiple FL clients from the same machine?

    Yes. The FL clients are identified by FL token, not machine IP. Each FL client will have its own FL token as well as
    instance name, which is the client name that must be used for issuing specific commands to that client.

#. Can I use the same client package to run multiple instances for the same client?

    Yes, you can start multiple instances of FL clients from the same client packages. Each FL client will be identified
    by its unique instance names, for example: "flclient1", "flclient1_1", "flclient1_2", etc. The instance name must be
    used for issuing specific commands to that client from the admin tool.

#. What happens if a federated learning client crashes?

    Federated learning clients will send a heartbeat call to the FL server once every minute. If an FL client crashes and
    the FL server does not get a heartbeat from that client for 10 minutes (can be set with "heart_beat_timeout" in the
    server's config json), the FL server will remove that client from the training client list.

#. Can FL clients join or quit in the middle of federated learning training?

    Yes, an FL client can join or quit in the middle of the FL training at any time. The client will pick up the global
    model at the current round of the server to participate in the FL training. When quitting, the FL server will
    automatically remove the FL client after it quits and no heartbeat is received for the duration of the
    "heart_beat_timeout" configured on the server. If using an admin tool, it is recommended to use the "abort" and
    "shutdown" commands to gracefully stop the clients.

#. For the Scatter and Gather workflow, what if the number of participating FL clients is below the minimum number of clients required?

    When an FL client passes authentication, it can request the current round of the global model and starts the FL training right away.
    There is no need to wait for other clients. Once the client finishes its own training, it will send the update to the server
    for aggregation. However, if the server does not receive enough updates from other clients, the FL server will not start
    the next round of FL training. The finished FL client will be waiting for the next round's model.

#. For the Scatter and Gather workflow, what happens if more than the minimum numbers of FL clients submit an updated model?

    The FL server begins model aggregation after accepting updates from the minimum number of FL clients required and
    waiting for "wait_after_min_clients" configured on the server. The updates that are received after this will be
    discarded. All the clients will get the next round of the global model to start the next round FL training.

#. How does a client decide to quit federated learning training?

    The FL client always asks the server for the next task to do. See how :ref:`controllers <controllers>` assign tasks to clients.

************************
Server related questions
************************

#. What happens if the FL server crashes?

    See :ref:`high_availability` for the features implemented in NVIDIA FLARE 2.1.0 around FL server failover.

#. Why does my FL server keep crashing after a certain round?

    Check that the amount of memory being consumed is not increasing in a way that it exceeds the available resources.
    If the process consumes too much memory, the operating system may kill it.

#. Does the federated learning server need a GPU?

    No, there is no need to have GPU on the server side for the FL server to deploy. However, certain handlers may require
    GPUs. To disable GPUs on the server, include the following in the shell script that runs the server::

        export CUDA_VISIBLE_DEVICES=

#. What port do I need to open from the firewall on the FL server network?

    Depending on the configuration of :ref:`project.yaml <project_yml>` which controls which port the gRPC is deployed to,
    the FL server network needs to open that port for outside clients to reach the FL server.

#. What if the federated learning server is behind a load balancer?

    Currently, federated learning does not support load balancing between multiple FL servers.

#. Is the Overseer now a new single point of failure?

    Even if the Overseer is out of service for a period of time, the whole system is designed to continue working if no
    FL server outage happens. If an FL server outage happens while the Overseer is also unavailable, the whole system
    will just keep trying to reconnect and restore services when the outage is over. High Availability is not guaranteed
    availability. The design goal of HA was to keep the system operational as much as possible without human intervention.

***************************************
Overall training flow related questions
***************************************

#. How does the federated learning server decide when to stop FL?

    For the Scatter and Gather workflow, the FL server runs from the "start_round" to "num_rounds". The FL server will
    stop the training when the current round meets "num_rounds". For other workflows, the logic within the workflow can
    make that decision.

#. Can I run the FL server on AWS while running the FL client within my institution?

    Yes, use the AWS instance name as the server cn in project.yml file. (e.g.: ec2-3-99-123-456.compute-1.amazonaws.com)

#. How can I deploy different applications for different clients?

    You can edit the application folder for each individual client on your desktop, then upload and deploy to each individual client
    with the admin tool. Each client can run with its own application configuration.

#. What should I do if the admin notices one client's training is behaving erroneously or unexpectedly?

    The admin can issue a command to abort the FL client training for a specified job: ``abort job_id client client_name``. If the command is issued
    without the client_name, then the command will be sent to all the clients. Because of the nature of model training, it
    may take a little time for the FL client to completely stop. Use the "check_status client client_name" command to see
    if the client status is "stopped".

#. Why do the admin commands to the clients have a long delay before getting a response?

    The admin commands to the clients pass through the server. If for some reason the command is delayed by the network, or
    if the client command takes a long time to process, the admin console will experience a delay for the response. The
    default timeout is 10 seconds. You can use the “set_timeout” command to adjust the command timeout. If this timeout
    value is set too low, the admin command may not reach the client to execute the command.

#. Why do commands sometimes fail?

    Sometimes if you are trying to check status of the client and the server is already busy transferring the model and
    does not have extra bandwidth for the command, the command may time out. In that case, please wait and try again.

************
Known issues
************

#. If the IP of the server changes, the admin client may not be able to connect anymore because the admin server remains
   bound to the original host and port. A possible workaround is to restart the FL server manually, and then the host
   will resolve to the updated IP for binding when restarting.
#. Running out of memory can happen at any time, especially if the server and clients are running on same machine.
   This can cause the server to die unexpectedly.
#. After calling ``shutdown client`` for a client running multi GPUs, a process (sub_worker_process) may remain. The
   work around for this is to run ``abort client`` before the ``shutdown`` command.
#. If a snapshot is in a corrupted state, the server may try to restore the job and get stuck. To resolve this, delete
   the snapshot from the location configured in project.yml for the snapshot_persistor storage (by default
   ``/tmp/nvflare/jobs-storage``), and ``abort_job`` should be able to stop the job on the server.
