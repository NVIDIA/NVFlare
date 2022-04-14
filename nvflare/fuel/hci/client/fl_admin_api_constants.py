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

from enum import Enum


class FLDetailKey(str, Enum):
    """Constants for FL details that can be returned in the FLAdminAPI."""

    APP_NAME = "app_name"
    REGISTERED_CLIENTS = "registered_clients"
    CONNECTED_CLIENTS = "connected_clients"
    SERVER_ENGINE_STATUS = "server_engine_status"
    SERVER_LOG = "server_log"
    CLIENT_LOG = "client_log"
    STATUS_TABLE = "status_table"
    RESPONSES = "responses"
    SUBMITTED_MODELS = "submitted_models"
