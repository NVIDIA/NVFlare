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

from nvflare.private.fed.server.server_engine import ServerEngine


class _FakeClientManager:
    def __init__(self):
        self.disabled = set()

    def is_client_disabled(self, client_name):
        return client_name in self.disabled

    def disable_client(self, client_name):
        self.disabled.add(client_name)
        return []


class _FakeServer:
    def __init__(self):
        self.client_manager = _FakeClientManager()
        self.admin_server = None
        self.removed_tokens = []

    def remove_client_data(self, token):
        self.removed_tokens.append(token)


def test_disable_clients_reports_already_disabled_state():
    server = _FakeServer()
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    first = engine.disable_clients(["site-1"])
    second = engine.disable_clients(["site-1"])

    assert first["clients"][0]["already_disabled"] is False
    assert second["clients"][0]["already_disabled"] is True
