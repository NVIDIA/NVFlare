**************************
What's New in FLARE v2.7.0
**************************

The new features can be divided into three categories

FLARE Core
==========

Job Recipe
-----------
Introducing new **Flare Job Recipe**: Simple Recipe to capture the code needed to specify the client training and server algorithm. This should greatly
simplify the data scientists code to write for federated learning job. The same Job Recipe can be run in SimEnv, PoCEnv, ProdEnv.

.. note::
    this feature is **technical review**, as we haven't convert all the example and code to Job Recipe.
    But more than half-dozen recipes are provided for you to use.

Here is an example of the FedAvg Job Recipe

.. code-block:: python

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    batch_size = args.batch_size

    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=SimpleNetwork(),
        train_script="client.py",
        train_args=f"--batch_size {batch_size}",
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)
    print()
    print("Result can be found in :", run.get_result())
    print("Job Status is:", run.get_status())
    print()



you can directly experience the recipe with recipe tutorial notebook `Job Recipe Tutorials <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/job_recipe.ipynb>`_
or read the :ref:`job_recipe`

Memory Management Improvements
------------------------------

There are two main issues with sending large messages:
A large memory space is required to serialize the message into bytes before sending it. Once memory is saturated, everything becomes very slow.
A large byte array sent as one single message could cause the network to be saturated, which could also slow down the overall processing.
These issues exists regardless the model is sent directly or via streaming. We have developed a few different ways to address theses issues.

Another issue of LLM streaming is limited by Memory size, the model size must fit into the memory. File-based streaming then not limited by the memory size.

For complete description of the memory management feature, please refer to :ref:`file_streaming` and :ref:`tensor_streaming`

We introduced FileStreamer in previous release, we are now introducing FileDownloader

Push vs. Pull
^^^^^^^^^^^^^

There are two ways to get the file sent from one place to other places: push and pull.
With push, the file owner sends the file to recipient(s). The push process is somewhat strict in that if the file is
sent to multiple recipients, all recipients must process the same chunks at the same time. If any one of them fails,
then the whole sending process fails. Hence, in practice, it is most useful when sending the file to a single recipient.

The “push” method is implemented with the **FileStreamer** class.
With pull, the file owner first prepares the file and gets the Reference ID (RID) for the file. I
t then sends the RID to all recipients in whatever way it wants (e.g. broadcast). Once the RID is received,
each recipient then pulls the file chunk by chunk until the whole file is received.

As you can see, pulling is much more relaxed in that recipients are not synchronized in any way.
Each recipient can pull the file at its own pace. This is very useful when sharing a file with multiple recipients.

The “pull” method is implemented with the **FileDownloader** class.



File Streaming
^^^^^^^^^^^^^^

File streaming is a function that allows a file to be shared with one or more receivers.
The file owner could be the FL Server or any FL Client. File streaming could be a very effective alternative to sending
large amounts of data with messages.

File streaming, on the other hand, sends the big file with many small messages,
each containing a chunk of file data. The big file is never loaded into memory completely.
Since only small messages are sent over the network, it is less likely to completely bog down the network.


FileDownloader
^^^^^^^^^^^^^^
The file downloading process requires three steps:
The data owner prepares the file(s) to be shared with recipients, and obtain one reference id (RID) for each file.
The data owner sends the RID(s) to all recipients. This is usually done with a broadcast message.
Recipients download the files one by one with received RIDs.


Tensor-Downloader
^^^^^^^^^^^^^^^^^^^^^^
in-process


Security Enhancement
--------------------

Fix the following issues:

-- Unsafe Deserialization - torch.jit.load  is replaced with safe-tensor based implementation

-- Unsafe Deserialization - Function Call -- FOB auto-registration is removed. A white listed FOBs are auto-registered.

-- Command Injection via Grep Parameters -- commands are reimplemented to avoid command injections


FLARE Server Port Consolidation
-------------------------------

Historically, Flare’s FL Server requires two communication port numbers to be open to the public.
One port is used for FL Client/Server communication, another is for Admin Client/Server communication.
For customers that port numbers are strictly managed, getting an extra port number could be challenging.

Flare 2.7 consolidates port number requirement to one: the same port number can be used for both types of communication!
For some customers, it may still be desirable to use different port numbers because they can be managed under
different network security policies. To accommodate such customers, the system can still be provisioned to use two different
port numbers for admin/server and client/server communications.

This features can be greatly reduce the dependency for IT support. Not only they only needs single port, the port could be
HTTPS port 443 using HTTP driver and using TLS.

Connection Example Illustration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following diagrams illustrate the two different connection and authentication mechanism
enabled by the single port, TLS, bring your own connection features.

PICTURES TODO


New HTTPS Driver
----------------
Prior to 2.7.0, the HTTP driver is very slow. We rewrite of HTTP driver using aiohttp library to solve the performance issue with the old driver.
The new driver's performance is on par with GRPC driver. The usage is exactly the same.


Pre-Install CLI command
------------------------

In case where custom code /dynamic code is not allowed to deployed, we need to pre-install the application to the
host. Although you can manually deploy these code without using any tool or command, the following pre-install tool
my provide simpler method.

The code pre-installer handles:
- Installation of application code
- Installation of shared libraries
- Site-specific customizations
- Python package dependencies

The tool provides two main commands:
- `prepare`: Package application code for installation
- `install`: Install packaged code to target sites

:ref:`pre_installer`



Confidential Federated AI
=========================

.. note::
    This feature is in **Technical Preview**.
    Reach to NVIDIA FLARE team for CVM build scripts: federatedlearning@nvidia.com

With this release, we offer this first of kind product for end-to-end IP protection solution in federated setup
using confidential computing.

- The solution is for on-premise deployment on bare metal using AMD CPU and NVIDIA GPU with Confidential VM.
- End-To-End Protection: by end-to-end protection, we are stating that it is not only protect the IP (model and code) in use at runtime,but also protect against the CVM tampering at deployment.
- The solution is able to perform
    - **security aggregation** on the server-side to protection privacy leak via model
    - **model theft Protection** on the client-side to safe guard the Model IP during collaboration
    - **data leak prevention** on the client-side with the pre-approved,certified code.

You can read more about the user usage at :ref:`cc_user_guide`

Develop Edge Applications with FLARE
====================================

FLARE 2.7 extends federated learning capabilities to edge devices. Edge device applications present some new challenges.

- **Scalability**: Unlike cross-silo applications where the number of FL clients is relatively small, the number of devices could be in the millions. It’s infeasible to treat the devices as simple FL clients and connect them directly to the FL server.
- **Stability**: Unlike cross-silo applications where the FL clients are stable, edge devices come and go at any time. This requires the training strategy to accommodate this behavior.
- **Compute capability**: Compared to cross-silo applications, edge devices don’t have as much computing power.
- **Platform dependency**: There are multiple edge device platforms (e.g. iOS, Android, etc.), and each has a different application development environment.

To support scalability, we add the following features

- support for hierarchical federated architecture :ref:`flare_hierarchical_architecture`
- asynch federated learning algorithm based on FedBuff: :ref:`_flare_edge`
- model development support for both iOS and Android :ref:`_flare_mobile`

Try FLARE edge development following the `edge examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/edge>`_




