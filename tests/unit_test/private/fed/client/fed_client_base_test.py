# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock

from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.client.fed_client_base import FederatedClientBase


def test_client_registration_log_omits_credentials():
    client = FederatedClientBase.__new__(FederatedClientBase)
    client.client_name = "site-1"
    client.token = None
    client.token_signature = None
    client.ssid = None
    client.communicator = MagicMock()
    client.communicator.client_registration.return_value = ("auth-token", "token-signature", "session-id")
    client.fl_ctx = FLContext()
    client.logger = MagicMock()

    client.client_register("project-1", FLContext())

    client.logger.info.assert_called_once_with("Successfully registered client site-1 for project project-1.")
