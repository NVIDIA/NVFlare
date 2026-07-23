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

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.collab.runtime import executor as executor_module
from nvflare.collab.runtime.defs import SYNC_TASK_NAME
from nvflare.collab.runtime.executor import CollabExecutor


def test_failed_start_run_does_not_publish_partially_configured_app(monkeypatch):
    monkeypatch.setattr(executor_module, "optional_import", lambda **kwargs: (None, False))

    executor = CollabExecutor(client_obj_id="client")
    executor.process_config = MagicMock(return_value="invalid config")
    executor.system_panic = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value.get_component.return_value = object()
    fl_ctx.get_identity_name.return_value = "site-1"

    try:
        executor._handle_start_run("start_run", fl_ctx)
        assert executor.client_app is None
        executor.system_panic.assert_called_once_with("invalid config", fl_ctx)
    finally:
        executor.thread_executor.shutdown()


def test_execute_returns_error_when_start_run_did_not_create_client_app():
    executor = CollabExecutor(client_obj_id="client")

    try:
        reply = executor.execute(SYNC_TASK_NAME, Shareable(), FLContext(), MagicMock())
        assert reply.get_return_code() == ReturnCode.ERROR
    finally:
        executor.thread_executor.shutdown()
