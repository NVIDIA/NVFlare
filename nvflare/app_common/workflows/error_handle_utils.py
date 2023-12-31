from nvflare.apis.fl_constant import ReturnCode

ABORT_WHEN_IN_ERROR = {
    ReturnCode.EXECUTION_EXCEPTION: True,
    ReturnCode.TASK_UNKNOWN: True,
    ReturnCode.EXECUTION_RESULT_ERROR: False,
    ReturnCode.TASK_DATA_FILTER_ERROR: True,
    ReturnCode.TASK_RESULT_FILTER_ERROR: True,
}

