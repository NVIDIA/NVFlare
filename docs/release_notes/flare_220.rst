What's New in FLARE v2.2
========================

With FLARE v2.2, the primary goals were to:
 - Accelerate the federated learning workflow
 - Simplify deploying a federated learning project in the real-world
 - Support federated data science
 - Enable integration with other platforms

To accomplish these goals, a set of key new tools and features were developed, including:
 - FL Simulator
 - FLARE Dashboard
 - Site-policy management
 - Federated XGboost <https://github.com/NVIDIA/NVFlare/tree/main/examples/xgboost>
 - Federated Statistics <https://github.com/NVIDIA/NVFlare/tree/main/examples/federated_statistics>
 - MONAI Integration <https://github.com/NVIDIA/NVFlare/tree/main/integration/monai>

The sections below provide an overview of these features.  For more detailed documentation and usage information, refer to the :ref:`User Guide <user_guide>` and :ref:`Programming Guide <programming_guide>`.

FL Simulator
------------
The :ref:`FL Simulator <fl_simulator>` is a lightweight tool that allows you to build, debug, and run a FLARE
application locally without explicitly deploying a provisioned FL system.  The FL Simulator provides both a CLI for
interactive use and an API for developing workflows programmatically. Clients are implemented using threads for each
client. If running in an environment with limited resources, multiple clients can be run sequentially using single
threads (or GPUs). This allows for testing the scalability of an application even with limited resources.

Users can run the FL Simulator in a python environment to debug FLARE application code directly. Jobs can be submitted
directly to the simulator without debugging, just as in a production FLARE deployment.  This allows you to build, debug,
and test in an interactive environment, and then deploy the same application in production without modification.

POC mode upgrade
----------------
For researchers who prefer to use :ref:`POC (proof of concept) <poc_command>` mode, the usage has been improved for
provisioning and starting a server and clients locally.

FLARE Dashboard and Provisioning
--------------------------------
The :ref:`FLARE Dashboard <nvflare_dashboard_ui>` provides a web UI that allows a project administrator to configure a
project and distribute client startup kits without the need to gather client information up-front, or manually configure
the project using the usual ``project.yml`` configuration.  Once the details of the project have been configured,
:ref:`provisioning <provisioning>` of client systems and FLARE Console users, is done on the fly. The web UI allows users to
register, and once approved, download project startup kits on-demand.  For those who wish to provision manually, the
provisioning CLI is still included in the main nvflare CLI:

.. code-block:: shell

  nvflare provision -h

The CLI method of provisioning has also been enhanced to allow for :ref:`dynamic provisioning <dynamic_provisioning>`,
allowing the addition of new sites or users without the need to re-provision existing sites.

In addition to these enhancements to the provisioning workflow, we provide some new tools to simplify local deployment
and troubleshoot client connectivity.  First is a ``docker-compose`` :ref:`utility <docker_compose>` that allows the
administrator to provision a set of local startup kits, and issue ``docker-compose up`` to start the server and connect
all clients.

We also provide a new :ref:`pre-flight check <preflight_check>` to help remote sites troubleshoot potential environment
and connectivity issues before attempting to connect to the FL Server.

.. code-block:: shell

  nvflare preflight-check -h

This command will examine all available provisioned packages (server, admin, clients, overseers) to check connections
between the different components (server, clients, overseers), ports, dns, storage access, etc., and provide suggestions
for how to fix any potential issues.

Federated Data Science
----------------------

Federated XGBoost
"""""""""""""""""

XGBoost is a popular machine learning method used by applied data scientists in a wide variety of applications. In FLARE v2.2,
we introcuce federated XGBoost integration, with a controller and executor that run distributed XGBoost training among a group
of clients.  See the `hello-xgboost example <https://github.com/NVIDIA/NVFlare/tree/main/examples/xgboost>`_ to get started.

Federated Statistics
""""""""""""""""""""
Before implementing a federated training application, a data scientist often performs a process of data exploration,
analysis, and feature engineering. One method of data exploration is to explore the statistical distribution of a dataset.
With FLARE v2.2, we indroduce federated statistics operators - a server controller and client executor.  With these
pre-defined operators, users define the statistics to be calculated locally on each client dataset, and the workflow
controller generates an output json file that contains global as well as individual site statistics.  This data can be
visualized to allow site-to-site and feature-to-feature comparison of metrics and histograms across the set of clients.

Site Policy Management and Security
-----------------------------------

Although the concept of client authorization and security policies are not new in FLARE, version 2.2 has shifted to
federated :ref:`site policy management <site_policy_management>`. In the past, authorization policies were defined by the
project administrator at time of provisioning, or in the job specification.  The shift to federated site policy allows
individual sites to control:

 - Site security policy
 - Resource management
 - Data privacy

With these new federated controls, the individual site has full control over authorization policies, what resources are
available to the client workflow, and what security filters are applied to incoming and outgoing traffic.

There is a new :ref:`project.yml template <project_yml>` for FLARE v2.2, and previous startup kits from previous versions (which contain the old TLS certificates)
will need to be re-provisioned.

In addition to the federated site policy, FLARE v2.2 also introduces secure logging and security auditing.  Secure
logging, when enabled, limits client output to only file and line numbers in the event of an error, rather than a full
traceback, preventing unintentionally disclosing site-specific information to the project administrator.  Secure
auditing keeps a site-specific log of all access and commands performed by the project admin.
