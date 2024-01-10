class Constant:

    CONFIG_TASK_NAME = "config"
    START_TASK_NAME = "start"

    KEY_MAX_RUN_TIME = "max_run_time"
    KEY_EXIT_CODE = "exit_code"
    KEY_CLIENT_RANKS = "client_ranks"

    CONFIG_TASK_TIMEOUT = 10.0
    START_TASK_TIMEOUT = 10.0
    XGB_SERVER_READY_TIMEOUT = 10.0

    TASK_CHECK_INTERVAL = 0.5
    JOB_STATUS_CHECK_INTERVAL = 2.0
    PER_CLIENT_STATUS_REPORT_TIMEOUT = 90.0
    WORKFLOW_PROGRESS_TIMEOUT = 3600.0

    TOPIC_XGB_REQUEST = "xgb.request"
    TOPIC_CLIENT_DONE = "xgb.client_done"

    KEY_XGB_OP = "xgb.op"
    KEY_XGB_MSG = "xgb.msg"

    OP_ALL_GATHER = "all_gather"
    OP_ALL_GATHER_V = "all_gather_v"
    OP_ALL_REDUCE = "all_reduce"
    OP_BROADCAST = "broadcast"
