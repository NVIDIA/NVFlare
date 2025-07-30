# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


class EdgeContextKey:
    REQUEST_FROM_EDGE = "__request_from_edge__"
    REPLY_TO_EDGE = "__reply_to_edge__"


class EdgeEventType:
    EDGE_TASK_REQUEST_RECEIVED = "_edge_task_request_received"
    EDGE_RESULT_REPORT_RECEIVED = "_edge_result_report_received"
    EDGE_SELECTION_REQUEST_RECEIVED = "_edge_selection_request_received"
    EDGE_JOB_REQUEST_RECEIVED = "_edge_job_request_received"


class EdgeTaskHeaderKey:
    TASK_SEQ = "task_seq"
    HAS_UPDATE_DATA = "has_update_data"
    UPDATE_INTERVAL = "update_interval"


class EdgeMsgTopic:
    TASK_REQUEST = "task_request"
    SELECTION_REQUEST = "selection_request"
    RESULT_REPORT = "result_report"


class MsgKey:
    PAYLOAD = "payload"
    RESULT = "result"
    MODE = "mode"
    WEIGHTS = "weights"
    NUM_DEVICES = "num_devices"
    TASK_ID = "task_id"
    TASK_DONE = "task_done"
    MODEL_VERSION = "model_version"


class EdgeApiStatus:
    OK = "OK"
    RETRY = "RETRY"
    DONE = "DONE"
    ERROR = "ERROR"
    NO_JOB = "NO_JOB"
    NO_TASK = "NO_TASK"
    INVALID_REQUEST = "INVALID_REQUEST"


class CookieKey:
    MODEL_VERSION = "model_version"
    DEVICE_SELECTION_ID = "device_selection_id"


class HttpHeaderKey:
    DEVICE_ID = "X-Flare-Device-ID"
    DEVICE_INFO = "X-Flare-Device-Info"
    USER_INFO = "X-Flare-User-Info"


class EdgeConfigFile:
    DEVICE_CONFIG = "device_config.json"


class SpecialDeviceId:
    DUMMY = "dummy"
    MAX_INDICATOR = "?"
    NUM_INDICATOR = "#"


class JobDataKey:
    CONFIG = "config"
