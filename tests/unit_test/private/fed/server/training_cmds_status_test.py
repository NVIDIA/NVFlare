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

from types import SimpleNamespace

from nvflare.apis.fl_constant import MachineStatus
from nvflare.private.fed.server.server_status import ServerStatus
from nvflare.private.fed.server.training_cmds import _server_status_value


def test_server_status_value_uses_server_lifecycle_started():
    engine = SimpleNamespace(server=SimpleNamespace(status=ServerStatus.STARTED))
    assert _server_status_value(engine) == MachineStatus.STARTED.value


def test_server_status_value_uses_server_lifecycle_starting():
    engine = SimpleNamespace(server=SimpleNamespace(status=ServerStatus.STARTING))
    assert _server_status_value(engine) == MachineStatus.STARTING.value


def test_server_status_value_defaults_to_stopped():
    engine = SimpleNamespace(server=SimpleNamespace(status=ServerStatus.STOPPED))
    assert _server_status_value(engine) == MachineStatus.STOPPED.value
