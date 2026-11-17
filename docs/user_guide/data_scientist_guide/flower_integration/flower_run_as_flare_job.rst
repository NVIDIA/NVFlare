***********************************
Run Flower Application as FLARE Job
***********************************

Install NVFlare and ``flwr>=1.43`` in the server and client environments.
Use the same Flower minor version throughout a job. The integration supports
HTTP Fleet communication on POSIX hosts; it no longer supports the gRPC adapter.

.. code-block:: shell

    pip install 'flwr>=1.43'

NVFlare writes a job-scoped ``$FLWR_HOME/config.toml`` and uses it to connect
``flwr run``, ``flwr list``, and ``flwr stop`` to the job's local SuperLink.
Do not configure a global SuperLink address. See :ref:`flower_http_design` for
payload limits, timeouts, security boundaries, and migration instructions.

Prefer :class:`FlowerRecipe<nvflare.app_opt.flower.recipe.FlowerRecipe>` for new
jobs. The following JSON configurations describe the underlying components.

To run a Flower application as a job in FLARE, follow these steps:

    - Copy all Flower application code (python code) into the job's "custom" folder. Note that all training functions are implemented in Flower, not in FLARE!
    - Create the ``config_fed_server.json`` and ``config_fed_client.json``
    - Submit the created job to FLARE system for execution

For a full example, see:
:github_nvflare_link:`Hello Flower <examples/hello-world/hello-flower>`

Server Config: config_fed_server.json
=====================================
A typical server configuration looks like this:

.. code-block:: json

    {
        "format_version": 2,
        "task_data_filters": [],
        "task_result_filters": [],
        "components": [
        ],
        "workflows": [
            {
                "id": "ctl",
                "path": "nvflare.app_opt.flower.controller.FlowerController",
                "args": {}
            }
        ]
    }

The :class:`FlowerController<nvflare.app_opt.flower.controller.FlowerController>`
configures the local SuperLink. ``superlink_ready_timeout`` defaults to 10 seconds
and waits for the HTTP health endpoint. ``run_config`` overrides Flower app
configuration. ``database`` selects Flower's database; ``num_rounds`` is retained
for TIE configuration, while Flower's app configuration controls training rounds.

``max_message_bytes`` must match the client configuration. FlowerRecipe sets
both automatically. Remove the obsolete ``int_client_grpc_options`` argument
from older exported jobs. See the API reference for other lifecycle settings.

Client Config: config_fed_client.json
-------------------------------------
A typical client configuration looks like this:

.. code-block:: json

    {
        "format_version": 2,
        "executors": [
            {
                "tasks": ["*"],
                "executor": {
                    "path": "nvflare.app_opt.flower.executor.FlowerExecutor",
                    "args": {}
                }
            }
        ],
        "task_result_filters": [],
        "task_data_filters": [],
        "components": []
    }

The FlowerExecutor has additional args that can be set to finetune its behavior, as shown below:

.. code-block:: python

    class FlowerExecutor(TieExecutor):
        def __init__(
            self,
            start_task_name=Constant.START_TASK_NAME,
            configure_task_name=Constant.CONFIG_TASK_NAME,
            per_msg_timeout=2.0,
            tx_timeout=25.0,
            client_shutdown_timeout=5.0,
        ):

The ``per_msg_timeout`` and ``tx_timeout`` configure :class:`ReliableMessage<nvflare.apis.utils.reliable_message.ReliableMessage>`,
which is used to send requests to the server. HTTP deadline caps are described in
:ref:`flower_http_design`.

The ``client_shutdown_timeout`` specifies how long to wait in seconds for graceful shutdown of the Flower's client-app process when
stopping the FL client. If the client-app process does not shut down within this time, it will be killed by Flare.
