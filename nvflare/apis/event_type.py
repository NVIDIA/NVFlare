# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class EventType(object):
    """Built-in system events."""

    SYSTEM_START = "_system_start"
    SYSTEM_END = "_system_end"
    ABOUT_TO_START_RUN = "_about_to_start_run"
    START_RUN = "_start_run"
    ABOUT_TO_END_RUN = "_about_to_end_run"
    END_RUN = "_end_run"
    SWAP_IN = "_swap_in"
    SWAP_OUT = "_swap_out"
    START_WORKFLOW = "_start_workflow"
    END_WORKFLOW = "_end_workflow"
    ABORT_TASK = "_abort_task"
    FATAL_SYSTEM_ERROR = "_fatal_system_error"
    FATAL_TASK_ERROR = "_fatal_task_error"
    JOB_DEPLOYED = "_job_deployed"
    JOB_STARTED = "_job_started"
    JOB_COMPLETED = "_job_completed"
    JOB_ABORTED = "_job_aborted"
    JOB_CANCELLED = "_job_cancelled"

    BEFORE_PULL_TASK = "_before_pull_task"
    AFTER_PULL_TASK = "_after_pull_task"
    BEFORE_PROCESS_SUBMISSION = "_before_process_submission"
    AFTER_PROCESS_SUBMISSION = "_after_process_submission"

    BEFORE_TASK_DATA_FILTER = "_before_task_data_filter"
    AFTER_TASK_DATA_FILTER = "_after_task_data_filter"
    BEFORE_TASK_RESULT_FILTER = "_before_task_result_filter"
    AFTER_TASK_RESULT_FILTER = "_after_task_result_filter"
    BEFORE_TASK_EXECUTION = "_before_task_execution"
    AFTER_TASK_EXECUTION = "_after_task_execution"
    BEFORE_SEND_TASK_RESULT = "_before_send_task_result"
    AFTER_SEND_TASK_RESULT = "_after_send_task_result"

    CRITICAL_LOG_AVAILABLE = "_critical_log_available"
    ERROR_LOG_AVAILABLE = "_error_log_available"
    EXCEPTION_LOG_AVAILABLE = "_exception_log_available"
    WARNING_LOG_AVAILABLE = "_warning_log_available"
    INFO_LOG_AVAILABLE = "_info_log_available"
    DEBUG_LOG_AVAILABLE = "_debug_log_available"

    PRE_RUN_RESULT_AVAILABLE = "_pre_run_result_available"

    # event types for job scheduling - server side
    BEFORE_CHECK_CLIENT_RESOURCES = "_before_check_client_resources"

    # event types for job scheduling - client side
    BEFORE_CHECK_RESOURCE_MANAGER = "_before_check_resource_manager"

    BEFORE_BUILD_COMPONENT = "_before_build_component"
