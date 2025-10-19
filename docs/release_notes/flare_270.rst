**************************
What's New in FLARE v2.7.0
**************************

The new features can be divided into following categories:


Confidential Federated AI
=========================

.. sidebar::

   **Confidential Federated AI Applications:**

   - **Collaborative R&D & Analytics**: Joint model training and agent-based analytics inside Confidential Computing enclaves, protecting data and IP.
   - **Model Protection**: Secure use and fine-tuning of proprietary or licensed foundation models.
   - **Regulated & Cross-Border AI**: Enable collaboration across finance, healthcare, defense, and global partners while maintaining privacy, sovereignty, and compliance.
   - **Secure Deployment**: Prevent model or data leakage during inference in untrusted environments.



With this release, we offer this first-of-its-kind product for end-to-end IP protection solution in federated setup
using confidential computing.

- The solution is for on-premise deployment on bare metal using AMD CPU and NVIDIA GPU with Confidential VM.
- End-to-End Protection: By end-to-end protection, we mean that we not only protect the IP (model and code) in use at runtime, but also protect against CVM tampering at deployment.
- The solution is able to perform

    - **Secure aggregation** on the server-side to protect against privacy leaks via model
    - **Model theft protection** on the client-side to safeguard Model IP during collaboration
    - **Data leak prevention** on the client-side with pre-approved, certified code.

.. admonition:: Confidential Federated AI

    This feature is in **Technical Preview**.
    Reach out to the NVIDIA FLARE team for CVM build scripts: federatedlearning@nvidia.com

    You can read more about the user usage at :ref:`confidential_computing`


FLARE Core
==========

Job Recipe
-----------

.. sidebar::

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


Introducing the new Flare Job Recipe: a lightweight way to capture the code needed to specify the client training logic and the server-side algorithm.
The same Job Recipe can run seamlessly in SimEnv, PoCEnv, or ProdEnv—from local experiments to production deployments.

With Flare Job Recipe, we are making the federated learning workflow dramatically simpler for data scientists.
In most cases, constructing a complete federated learning job requires only about 6+ lines of Python code.
When combined with the Client API (typically 4+ lines), building and running federated learning experiments becomes almost effortless.


.. admonition:: Job Recipe

    This feature is in **technical preview**. Not all examples and code have been converted to use Job Recipe yet.
    However, you can directly experience the recipe with recipe tutorial notebook `Job Recipe Tutorials <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/job_recipe.ipynb>`_
    or read the :ref:`job_recipe`, more than half a dozen ready-to-use recipes are provided: :ref:`quickstart`


Enhanced Communication: Port Consolidation and new HTTP Driver
--------------------------------------------------------------

.. sidebar::

    **Port Consolidation**
    Previously, FLARE’s server required two separate ports: one for FL client/server communication and another for
    Admin client/server communication. In 2.7, these are merged into a single configurable port, reducing network configuration complexity.
    Dual-port mode remains available for environments with stricter network policies.

    **New HTTPS Driver**
    The HTTP driver has been rewritten using aiohttp to address prior performance limitations. It now matches gRPC performance,
    while maintaining the same API, TLS support, and backward compatibility with existing deployments.


- **Consolidated Port**: Reduced from two ports to a single port, simplifying deployment.
- **Standard Port Compatibility**: Use standard HTTPS port 443 - no need for IT to open additional ports
- **High Performance**: New HTTP driver matches gRPC in speed and reliability.

.. admonition:: Why it matters

    **Faster Deployment**: Eliminates network configuration delays and IT approvals for opening custom ports.
    FLARE 2.7.0 simplifies network requirements and allows fully secure deployments on standard infrastructure.
    :ref:`Check out FL server port consolidation details <server_port_consolidation>`.

Security Enhancement
--------------------

Fixed the following issues:

- Unsafe Deserialization - torch.jit.load is replaced with safe-tensor based implementation
- Unsafe Deserialization - Function Call - FOB auto-registration is removed. A whitelist of FOBs is auto-registered.
- Command Injection via Grep Parameters - commands are reimplemented to avoid command injections


.. admonition:: Security Enhancements

    Many similar issues are also fixed



Develop Edge Applications with FLARE
====================================

.. sidebar::

   .. image:: resources/hierarchical_fl.png
        :height: 150px

   .. image:: resources/edge_cross_device_fl.png
        :height: 150px

   .. image:: resources/edge_simplify_device_programming.png
        :height: 150px



FLARE 2.7 extends federated learning to edge devices with features that directly address the unique challenges of edge
environments:

**Scalability**: **Hierarchical federated architecture** :ref:`flare_hierarchical_architecture` allows millions of edge devices
to participate efficiently without connecting each directly to the server.

**Intermittent Device Participation**: **Asynchronous FL** based on FedBuff :ref:`flare_edge` handles devices that may join,
leave, or fail to return local training results due to network or power interruptions.

**Cross-Platform & No Device Programming Required**: Data scientists can deploy models to iOS and Android :ref:`flare_mobile`
without writing Swift, Objective-C, Java, or Kotlin. FLARE handles PyTorch → Executorch conversion and device training code automatically.

**Simulation Tools**: device simulator for large scale testing


.. admonition:: FLARE Edge

    Try FLARE edge development following the `edge examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/edge>`_



Self-Paced-Training Tutorials
==============================

Welcome to the five-part course on Federated Learning with NVIDIA FLARE!
This course covers everything from the fundamentals to advanced applications, system deployment, privacy, security,
and real-world industry use cases.

.. admonition:: Federated Learning with NVIDIA FLARE

    This tutorial has **100+ notebooks** and **80 videos**.
    See details in :ref:`self_paced_training`


Extra Features
==============

There are additional new features released in version 2.7.0, including memory management improvements with FileDownloader for large model streaming and a pre-install CLI command for environments where dynamic code deployment is restricted. You can find more details in :ref:`extra_270`.






