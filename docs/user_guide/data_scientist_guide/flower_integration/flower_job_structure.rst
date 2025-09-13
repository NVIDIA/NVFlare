********************
Flower Job Structure
********************
Even though Flower Programming is out of the scope of FLARE/Flower integration, you need to have a good
understanding of the Flower Job Structure when submitting to FLARE.

A Flower job is a regular FLARE job with special requirements for the ``custom`` directory, as shown below.

.. code-block:: none

    ├── flwr_pt
    │   ├── client.py   # <-- contains `ClientApp`
    │   ├── __init__.py # <-- to register the python module
    │   ├── server.py   # <-- contains `ServerApp`
    │   └── task.py     # <-- task-specific code (model, data)
    └── pyproject.toml  # <-- Flower project file

Project Folder
==============
All Flower app code must be placed in a subfolder in the ``custom`` directory of the job. This subfolder is called
the project folder of the app. In this example, the project folder is named ``flwr_pt``. Typically, this folder
contains ``server.py``, ``client.py``, and the ``__init__.py``. Though you could organize them differently (see discussion
below), we recommend always including the ``__init__.py`` so that the project folder is guaranteed to be a valid Python
package, regardless of Python versions.

Pyproject.toml
==============
The ``pyproject.toml`` file exists in the job's ``custom`` folder. It is an important file that contains server and
client app definition and configuration information. Such information is used by the Flower system to find the
server app and the client app, and to pass app-specific configuration to the apps.

Here is an example of ``pyproject.toml``, taken from :github_nvflare_link:`this example <examples/hello-world/hello-flower/jobs/hello-flwr-pt/app/custom/pyproject.toml>`.

.. code-block:: toml

    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"

    [project]
    name = "flwr_pt"
    version = "1.0.0"
    description = ""
    license = "Apache-2.0"
    dependencies = [
        "flwr[simulation]>=1.11.0,<2.0",
        "nvflare~=2.5.0rc",
        "torch==2.2.1",
        "torchvision==0.17.1",
    ]

    [tool.hatch.build.targets.wheel]
    packages = ["."]

    [tool.flwr.app]
    publisher = "nvidia"

    [tool.flwr.app.components]
    serverapp = "flwr_pt.server:app"
    clientapp = "flwr_pt.client:app"

    [tool.flwr.app.config]
    num-server-rounds = 3

    [tool.flwr.federations]
    default = "local-simulation"

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 2


.. note:: Note that the information defined in pyproject.toml must match the code in the project folder!

Project Name
------------
The project name should match the name of the project folder, though not a requirement. In this example, it is ``flwr_pt``. 
Server App Specification

This value is specified following this format:

.. code-block:: toml

    <server_app_module>:<server_app_var_name>

where:

    - The <server_app_module> is the module that contains the server app code. This module is usually defined as ``server.py`` in the project folder (flwr_pt in this example). 
    - The <server_app_var_name> is the name of the variable that holds the ServerApp object in the <server_app_module>. This variable is usually defined as ``app``:

.. code-block:: python

    app = ServerApp(server_fn=server_fn)


Client App Specification
------------------------
This value is specified following this format:

.. code-block:: toml

	<client_app_module>:<client_app_var_name>

where:

	- The <client_app_module> is the module that contains the client app code. This module is usually defined as ``client.py`` in the project folder (flwr_pt in this example). 
	- The <client_app_var_name> is the name of the variable that holds the ClientApp object in the <client_app_module>. This variable is usually defined as ``app``:

.. code-block:: python

    app = ClientApp(client_fn=client_fn)


App Configuration
-----------------
The pyproject.toml file can contain app config information, in the ``[tool.flwr.app.config]`` section. In this example,
it defines the number of rounds:

.. code-block:: toml

    [tool.flwr.app.config]
    num-server-rounds = 3

The content of this section is specific to the server app code. The ``server.py`` in the example shows how this is used:

.. code-block:: python

    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]

        # Define config
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)

Supernode Count
---------------
If you run the Flower job with its simulation (not as a FLARE job), you need to specify how many clients (supernodes) to use
for the simulation in the ``[tool.flwr.federations.local-simulation]`` section, like this:

.. code-block:: toml

    options.num-supernodes = 2

But this does not apply when submitting it as a FLARE job.
