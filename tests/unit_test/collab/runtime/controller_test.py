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

from nvflare.collab.runtime import controller as controller_module
from nvflare.collab.runtime.controller import CollabController
from nvflare.collab.runtime.defs import SyncKey


def test_default_sync_timeout_allows_client_startup():
    assert CollabController().sync_task_timeout == 60


def test_fractional_sync_timeout_rounds_up(monkeypatch):
    task_class = MagicMock()
    monkeypatch.setattr(controller_module, "Task", task_class)
    monkeypatch.setattr(controller_module, "prepare_for_remote_call", MagicMock())
    monkeypatch.setattr(controller_module, "run_server", MagicMock())

    engine = MagicMock()
    engine.get_clients.return_value = []
    engine.get_cell.return_value.get_fqcn.return_value = "server"
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    fl_ctx.get_identity_name.return_value = "server"
    fl_ctx.get_job_id.return_value = "job"
    fl_ctx.get_prop.return_value = None
    fl_ctx.get_peer_context.return_value = None
    server_app = MagicMock()
    server_app.get_collab_interface.return_value = {}

    controller = CollabController(sync_task_timeout=0.5)
    controller.server_app = server_app
    controller.broadcast_and_wait = MagicMock()
    controller.control_flow(abort_signal=MagicMock(), fl_ctx=fl_ctx)

    assert task_class.call_args.kwargs["timeout"] == 1
    assert task_class.call_args.kwargs["data"][SyncKey.SERVER_FQCN] == "server"
