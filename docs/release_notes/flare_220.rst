************************
What's New in FLARE v2.2
************************

Goals of v2.2 and New Features
==============================
With FLARE v2.2, the primary goals were to:
 - Accelerate the federated learning workflow
 - Simplify deploying a federated learning project in the real-world
 - Support federated data science
 - Enable integration with other platforms

To accomplish these goals, a set of key new tools and features were developed, including:
 - FL Simulator
 - FLARE Dashboard
 - :ref:`dynamic_provisioning`
 - Improved :ref:`POC (proof of concept) command <poc_command>`
 - :ref:`docker_compose`
 - :ref:`preflight_check`
 - Site-policy management
 - Federated XGboost <https://github.com/NVIDIA/NVFlare/tree/main/examples/xgboost>
 - Federated Statistics <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/federated-statistics>
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
we introduce federated XGBoost integration, with a controller and executor that run distributed XGBoost training among a group
of clients.  See the :github_nvflare_link:`hello-xgboost example <examples/xgboost>` to get started.

Federated Statistics
""""""""""""""""""""
Before implementing a federated training application, a data scientist often performs a process of data exploration,
analysis, and feature engineering. One method of data exploration is to explore the statistical distribution of a dataset.
With FLARE v2.2, we introduce federated statistics operators - a server controller and client executor.  With these
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

Migration to 2.2.1: Notes and Tips
==================================

Stop using Pickle in favor of using FOBS to serialize/deserialize data between Client and Server
------------------------------------------------------------------------------------------------
Prior to NVFLARE 2.1.4, NVFLARE used python's `pickle <https://docs.python.org/3/library/pickle.html>`_ to transfer data between the FL clients and server.
NVFLARE now uses the FLARE Object Serializer (FOBS). You might experience failures if your code is still using Pickle. 
To migrate the code or if you experience errors due to this, please refer to :github_nvflare_link:`Flare Object Serializer (FOBS) <nvflare/fuel/utils/fobs/README.rst>`.

Another type of failure is due to data types that are not supported by FOBS. By default FOBS supports some data types, if the data type (Custom Class or Class from 3rd parties)
is not part of supported FOBS data type, then you need to follow the instructions at
:github_nvflare_link:`Flare Object Serializer (FOBS) <nvflare/fuel/utils/fobs/README.rst>`.

Essentially, to address this type of issue, you need to do the following steps:
  - Create a FobDecomposer class for the targeted data type
  - Register the newly created FobDecomposer before the data type is transmitted between client and server.

The following examples are directly copied from :github_nvflare_link:`Flare Object Serializer (FOBS) <nvflare/fuel/utils/fobs/README.rst>`.

.. code-block:: python

    from nvflare.fuel.utils import fobs

    class Simple:

        def __init__(self, num: int, name: str, timestamp: datetime):
            self.num = num
            self.name = name
            self.timestamp = timestamp


    class SimpleDecomposer(fobs.Decomposer):

        @staticmethod
        def supported_type() -> Type[Any]:
            return Simple

        def decompose(self, obj) -> Any:
            return [obj.num, obj.name, obj.timestamp]

        def recompose(self, data: Any) -> Simple:
            return Simple(data[0], data[1], data[2])

Register the data type in FOBS before the data type is used, then you can register the newly created FOBDecomposer

.. code-block:: python

    fobs.register(SimpleDecomposer)

.. note::

  The decomposers must be registered in both server and client code before FOBS is used.
  A good place for registration is the constructors for the controllers and executors. It can also be done in the START_RUN event handler.

Use FOBS to serialize data before you use sharable
""""""""""""""""""""""""""""""""""""""""""""""""""
A custom object cannot be put in shareable directly, it must be serialized using FOBS first.
Assuming custom_data contains custom type, this is how data can be stored in shareable:

.. code-block:: python

    shareable[CUSTOM_DATA] = fobs.dumps(custom_data)

On the receiving end:

.. code-block:: python

    custom_data = fobs.loads(shareable[CUSTOM_DATA])


.. note::

  This does not work:

  .. code-block:: python
  
    shareable[CUSTOM_DATA] = custom_data

Replace TLS certificates
------------------------
With 2.2.1, the authorization model has been changed so previous startup kits (which contain the old TLS certificates) will no longer work. You will need to clean up
the old setartup kits and re-provision your project.

Use new Project.yml template
----------------------------
With 2.2.1, federated site policies require the new project.yml template. Please refer to :ref:`project_yml`.

New local directory
-------------------
With 2.2.1, the provision command will produce not only the ``startup`` directory, but a ``local`` directory. 
The resource allocation that used to be in ``project.yml`` is now expected in a ``resources.json`` file in this new ``local`` directory, and each
sites/clients needs to manage this separately for each location.
You need to place/modify your own site's ``authorization.json`` and ``privacy.json`` files in the ``local`` directory as well if you want to
change the default policies.

The default configurations are provided in each site's local directory:

.. code-block::

    local
    ├── authorization.json.default
    ├── log.config.default
    ├── privacy.json.sample
    └── resources.json.default

These defaults can be overridden by removing the default suffix and modifying the configuration as needed for the specific site.
