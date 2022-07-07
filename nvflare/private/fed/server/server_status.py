# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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


class ServerStatus(object):
    NOT_STARTED = 0
    STARTING = 1
    STARTED = 2
    STOPPED = 3
    SHUTDOWN = 4

    status_messages = {
        NOT_STARTED: "app server not started",
        STARTING: "app server starting",
        STARTED: "app server started",
        STOPPED: "app server stopped",
        SHUTDOWN: "FL server shutdown",
    }


def get_status_message(status):
    return ServerStatus.status_messages.get(status)
