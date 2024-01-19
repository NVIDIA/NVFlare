class Constant:

    CONFIG_TASK_NAME = "config"
    START_TASK_NAME = "start"

    CONF_KEY_MAX_RUN_TIME = "max_run_time"
    CONF_KEY_EXIT_CODE = "exit_code"
    CONF_KEY_CLIENT_RANKS = "client_ranks"
    CONF_KEY_RANK = "rank"
    CONF_KEY_WORLD_SIZE = "world_size"
    CONF_KEY_NUM_ROUNDS = "num_rounds"

    CONFIG_TASK_TIMEOUT = 10.0
    START_TASK_TIMEOUT = 10.0
    XGB_SERVER_READY_TIMEOUT = 10.0

    TASK_CHECK_INTERVAL = 0.5
    JOB_STATUS_CHECK_INTERVAL = 2.0
    PER_CLIENT_STATUS_REPORT_TIMEOUT = 90.0
    WORKFLOW_PROGRESS_TIMEOUT = 3600.0

    TOPIC_XGB_REQUEST = "xgb.request"
    TOPIC_XGB_REQUEST_CHECK = "xgb.req_check"
    TOPIC_CLIENT_DONE = "xgb.client_done"

    KEY_XGB_OP = "xgb.op"
    KEY_XGB_REQ_ID = "xgb.req_id"
    KEY_XGB_REQ_TRY_NUM = "xgb.req_try_num"
    KEY_XGB_REQ_RECEIVED = "xgb.req_received"
    KEY_XGB_RANK = "xgb.rank"
    KEY_XGB_SEQ = "xgb.seq"
    KEY_XGB_SEND_BUF = "xgb.send_buf"
    KEY_XGB_DATA_TYPE = "xgb.data_type"
    KEY_XGB_REDUCE_OP = "xgb.reduce_op"
    KEY_XGB_ROOT = "xgb.root"
    KEY_XGB_RCV_BUF = "xgb.rcv_buf"

    OP_ALL_GATHER = "all_gather"
    OP_ALL_GATHER_V = "all_gather_v"
    OP_ALL_REDUCE = "all_reduce"
    OP_BROADCAST = "broadcast"

    OPCODE_NONE = 0
    OPCODE_ALL_GATHER = 1
    OPCODE_ALL_GATHER_V = 2
    OPCODE_ALL_REDUCE = 3
    OPCODE_BROADCAST = 4
    OPCODE_DONE = 99

    ERR_OP_MISMATCH = -1
    ERR_INVALID_RANK = -2
    ERR_NO_CLIENT_FOR_RANK = -3
    ERR_CLIENT_ERROR = -4

    PARAM_KEY_SEQ = "seq"
    PARAM_KEY_SEND_BUF = "send_buf"
    PARAM_KEY_DATA_TYPE = "data_type"
    PARAM_KEY_REDUCE_OP = "reduce_op"
    PARAM_KEY_ROOT = "root"
