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

from enum import Enum


class APIStatus(str, Enum):
    """Constants for the valid status options for the status of FLAdminAPIResponse."""

    SUCCESS = "SUCCESS"  # command issues successfully
    ERROR_PROTOCOL = (
        "ERROR_PROTOCOL"  # the payload/data is not following the correct format/protocol expected by the server
    )
    ERROR_CERT = "ERROR_CERT"  # key or certs are incorrect
    ERROR_AUTHENTICATION = "ERROR_AUTHENTICATION"  # authentication failed, need to log in
    ERROR_AUTHORIZATION = "ERROR_AUTHORIZATION"  # authorization failed, permissions
    ERROR_SYNTAX = "ERROR_SYNTAX"  # command syntax incorrect
    ERROR_RUNTIME = "ERROR_RUNTIME"  # various errors at runtime depending on the command
    ERROR_INVALID_CLIENT = "ERROR_INVALID_CLIENT"  # wrong/invalid client names exists in command
    ERROR_INACTIVE_SESSION = "ERROR_INACTIVE_SESSION"  # admin client session is inactive
    ERROR_SERVER_CONNECTION = "ERROR_SERVER_CONNECTION"  # server connection error
