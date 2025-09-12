***********************************
Run Flower Application as FLARE Job
***********************************

Before running Flower applications with FLARE, you must have both FLARE and Flower frameworks
installed in your Python environment. As of this writing, the Flower version to be used is 1.11.0rc0.

.. code-block:: shell

    pip install flwr==1.11.0rc0

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

The :class:`FlowerController<nvflare.app_opt.flower.controller.FlowerController>` has additional args that can be
set to finetune its behavior, as shown below:

.. code-block:: python

    class FlowerController(TieController):
        def __init__(
            self,
            num_rounds=1,
            database: str = "",
            server_app_args: list = None,
            superlink_ready_timeout: float = 10.0,
            configure_task_name=TieConstant.CONFIG_TASK_NAME,
            configure_task_timeout=TieConstant.CONFIG_TASK_TIMEOUT,
            start_task_name=TieConstant.START_TASK_NAME,
            start_task_timeout=TieConstant.START_TASK_TIMEOUT,
            job_status_check_interval: float = TieConstant.JOB_STATUS_CHECK_INTERVAL,
            max_client_op_interval: float = TieConstant.MAX_CLIENT_OP_INTERVAL,
            progress_timeout: float = TieConstant.WORKFLOW_PROGRESS_TIMEOUT,
            int_client_grpc_options=None,
        ):
            """Constructor of FlowerController

            Args:
                num_rounds: number of rounds. Not used in this version.
                database: database name
                server_app_args: additional server app CLI args
                superlink_ready_timeout: how long to wait for the superlink to become ready before starting server app
                configure_task_name: name of the config task
                configure_task_timeout: max time allowed for config task to complete
                start_task_name: name of the start task
                start_task_timeout: max time allowed for start task to complete
                job_status_check_interval: how often to check job status
                max_client_op_interval: max time allowed for missing client requests
                progress_timeout: max time allowed for missing overall progress
                int_client_grpc_options: internal grpc client options
            """

The args ``num_rounds``, ``database``, and ``server_app_args`` are not currently used. 

Default values for most args should be good enough. You may need to adjust the following args in some special cases.

``Superlink_ready_timeout`` - superlink process is started first and must become ready before starting the server-app process.
It may take some time for the superlink to become ready (port is open and ready for the server-app). The default value is
10 seconds, which should be enough for most cases. If not, you may need to increase it.


Rest of the args are for job lifecycle management. Their meanings are the same as those used for
:ref:`XGBoost controller<secure_xgboost_controller>`.


Client Config: config_fed_client.json
=====================================
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
            per_msg_timeout=10.0,
            tx_timeout=100.0,
            client_shutdown_timeout=5.0,
        ):

The ``per_msg_timeout`` and ``tx_timeout`` configure :class:`ReliableMessage<nvflare.apis.utils.reliable_message.ReliableMessage>`,
which is used to send requests to the server.

The ``client_shutdown_timeout`` specifies how long to wait in seconds for graceful shutdown of the Flower's client-app process when
stopping the FL client. If the client-app process does not shut down within this time, it will be killed by Flare.
