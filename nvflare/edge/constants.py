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
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode


class Status(CellReturnCode):
    NO_TASK = "no_task"
    NO_JOB = "no_job"


class EdgeProtoKey:
    STATUS = "status"
    DATA = "data"


class EdgeContextKey:
    JOB_ID = "__edge_job_id__"
    EDGE_CAPABILITIES = "__edge_capabilities__"
    REQUEST_FROM_EDGE = "__request_from_edge__"
    REPLY_TO_EDGE = "__reply_to_edge__"


class EventType:
    EDGE_REQUEST_RECEIVED = "_edge_request_received"
    EDGE_JOB_REQUEST_RECEIVED = "_edge_job_request_received"
