**************************
What's New in FLARE v2.7.0
**************************

The new features can be divided into three categories


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

You can read more about the user usage at :ref:`confidential_computing`

.. sidebar::

    **Confidential Federated AI Applications:**
    -------------------------------------------

    Where would you use this ?

    - **Pharmaceutical and Biotech R&D:** Multiple organizations can jointly train or validate models while ensuring that each company’s proprietary model and data remain protected.

    - **Licensed or Proprietary Foundation Models:** Models can be used as starting points without risking leakage or violating license agreements.

    - **AI Sovereignty Across Borders:** Organizations can collaborate globally while ensuring models remain under sovereign control and do not cross restricted boundaries.

    - **Secure Inference Serving:** Safeguard models from being exposed to infrastructure providers, cloud platforms, or third-party serving environments.

    - **Financial Services:** Add an extra layer of protection for highly regulated sectors such as banking, insurance, and trading.

    - **Healthcare Collaborations:** Enable hospitals and research institutes to co-train models without risking sensitive patient data or revealing proprietary clinical models.

    - **Defense and National Security:** Allow cross-agency AI projects while preserving strict confidentiality of algorithms and data.

    - **Cross-Industry Consortia:** Support joint innovation (e.g., supply chain, energy, automotive) without risking leakage of competitive IP.

FLARE Core
==========

Job Recipe
-----------

Introducing the new Flare Job Recipe: a lightweight way to capture the code needed to specify the client training logic and the server-side algorithm.
The same Job Recipe can run seamlessly in SimEnv, PoCEnv, or ProdEnv—from local experiments to production deployments.

With Flare Job Recipe, we are making the federated learning workflow dramatically simpler for data scientists.
In most cases, constructing a complete federated learning job requires only about 6+ lines of Python code.
When combined with the Client API (typically 4+ lines), building and running federated learning experiments becomes almost effortless.


.. note::
    This feature is in technical review. Not all examples and code have been converted to use Job Recipe yet.
    However, more than half a dozen ready-to-use recipes are provided: :ref:quickstart


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


Simplified Deployment: Single-Port Server & Efficient HTTPS Driver
------------------------------------------------------------------

Deploying a Federated Learning system often requires IT support to open new ports, which can be time-consuming due to
additional security reviews and approvals. FLARE 2.7.0 addresses this challenge by consolidating the port requirements to
a single port, utilizing TLS, and introducing a new HTTPS driver that performs on par with gRPC. This allows the use of
the standard HTTPS port 443, significantly reducing the dependency on IT support.

**Port Consolidation**

Historically, FLARE’s FL Server required two communication ports: one for FL Client/Server communication and another for
Admin Client/Server communication. This posed challenges for customers with strict port management policies. With FLARE 2.7,
the requirement is consolidated to a single port for both communication types. However, for those who prefer separate ports
due to different network security policies, the system can still be configured to use two distinct ports.
reference :ref:`server_port_consolidation` for details

**New HTTPS Driver**

Prior to version 2.7.0, the HTTP driver was slow. The new driver, rewritten using the `aiohttp` library, resolves these
performance issues, matching the efficiency of the gRPC driver. The usage remains unchanged, ensuring a seamless transition.

**Connection Example Illustration**

The following diagrams illustrate the two different connection and authentication mechanisms
enabled by the single port, TLS, bring your own connection features.

.. image:: resources/flare_byocc.png
    :height: 300px

See :ref:`check out FL server port consolidation details <server_port_consolidation>`.


Security Enhancement
--------------------

Fix the following issues:

-- Unsafe Deserialization - torch.jit.load  is replaced with safe-tensor based implementation

-- Unsafe Deserialization - Function Call -- FOB auto-registration is removed. A white listed FOBs are auto-registered.

-- Command Injection via Grep Parameters -- commands are reimplemented to avoid command injections



Develop Edge Applications with FLARE
====================================

FLARE 2.7 extends federated learning capabilities to edge devices. Edge device applications present some new challenges.

- **Scalability**: Unlike cross-silo applications where the number of FL clients is relatively small, the number of devices could be in the millions. It’s infeasible to treat the devices as simple FL clients and connect them directly to the FL server.
- **Stability**: Unlike cross-silo applications where the FL clients are stable, edge devices come and go at any time. This requires the training strategy to accommodate this behavior.
- **Compute capability**: Compared to cross-silo applications, edge devices don’t have as much computing power.
- **Platform dependency**: There are multiple edge device platforms (e.g. iOS, Android, etc.), and each has a different application development environment.

To support scalability, we add the following features

- support for hierarchical federated architecture :ref:`flare_hierarchical_architecture`
- asynch federated learning algorithm based on FedBuff: :ref:`flare_edge`
- model development support for both iOS and Android :ref:`flare_mobile`

Try FLARE edge development following the `edge examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/edge>`_


Self-Paced-Training Tutorials
==============================

Federated Learning with NVIDIA FLARE: Notebooks and videos
----------------------------------------------------------
Welcome to the five-part course on Federated Learning with NVIDIA FLARE!
This course covers everything from the fundamentals to advanced applications, system deployment, privacy, security, and real-world industry use cases.
it has **100+** notebooks, **80** videos

see details in :ref:`self_paced_training`



Extra Features
==============
There are additional new features released in version 2.7.0. You can find more details in `./extra_270.rst`.




