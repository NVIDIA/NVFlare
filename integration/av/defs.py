CHANNEL = "flare_agent"

TOPIC_GET_TASK = "av_get_task"
TOPIC_SUBMIT_RESULT = "av_submit_result"
TOPIC_HEARTBEAT = "heartbeat"
TOPIC_HELLO = "hello"
TOPIC_BYE = "bye"
TOPIC_ABORT = "abort"

JOB_META_KEY_AGENT_ID = "agent_id"


class RC:
    OK = "OK"
    BAD_TASK_DATA = "BAD_TASK_DATA"
    EXECUTION_EXCEPTION = "EXECUTION_EXCEPTION"


class MsgHeader:

    TASK_ID = "task_id"
    TASK_NAME = "task_name"
    RC = "rc"


class PayloadKey:
    MODEL = "model"
    MODEL_META = "model_meta"


class ModelMetaKey:
    CURRENT_ROUND = "current_round"
    TOTAL_ROUND = "total_round"
    DATA_KIND = "data_kind"
    NUM_STEPS_CURRENT_ROUND = "NUM_STEPS_CURRENT_ROUND"
    PROCESSED_ALGORITHM = "PROCESSED_ALGORITHM"
    PROCESSED_KEYS = "PROCESSED_KEYS"
    INITIAL_METRICS = "initial_metrics"
    FILTER_HISTORY = "filter_history"
