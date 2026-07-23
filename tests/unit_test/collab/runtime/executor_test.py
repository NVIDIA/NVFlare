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

import threading
from unittest.mock import MagicMock

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.shareable import Shareable
from nvflare.collab import collab
from nvflare.collab.runtime import executor as executor_module
from nvflare.collab.runtime.defs import SETUP_TASK_NAME, SYNC_TASK_NAME, SyncKey
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


def test_end_run_installs_client_context_on_event_thread():
    finalized_at = []

    class Client:
        @collab.final
        def finalize(self):
            finalized_at.append(collab.site_name)

    executor = CollabExecutor(client_obj_id="client")
    executor.client_app = executor_module.ClientApp(Client())
    executor.client_app.name = "site-1"
    executor.client_ctx = executor.client_app.new_context("site-1", "site-1", set_call_ctx=False)
    thread = threading.Thread(target=executor._handle_end_run, args=("end_run", FLContext()))
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert finalized_at == ["site-1"]


def test_setup_uses_each_remote_clients_reported_interface(monkeypatch):
    first_client = MagicMock()
    first_client.name = "site-1"
    second_client = MagicMock()
    second_client.name = "site-2"
    monkeypatch.setattr(executor_module, "from_dict", lambda client: client)
    monkeypatch.setattr(executor_module, "prepare_for_remote_call", MagicMock())

    executor = CollabExecutor(client_obj_id="client")
    executor.client_app = MagicMock()
    executor.client_app.name = "site-1"
    executor.client_app.get_collab_interface.return_value = {"": {"local": []}}
    executor.client_app.new_context.return_value = MagicMock()
    executor.log_info = MagicMock()
    executor._prepare_server_proxy = MagicMock(return_value=MagicMock())
    executor._prepare_client_proxy = MagicMock(side_effect=[MagicMock(), MagicMock()])

    engine = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    fl_ctx.get_identity_name.return_value = "site-1"
    fl_ctx.get_prop.side_effect = lambda key, default=None: {
        FLContextKey.JOB_META: {JobMetaKey.JOB_CLIENTS: [first_client, second_client]}
    }.get(key, default)
    shareable = Shareable(
        {
            SyncKey.COLLAB_INTERFACE: {"": {"server": []}},
            SyncKey.CLIENT_INTERFACES: {
                "site-1": {"": {"first": []}},
                "site-2": {"": {"second": []}},
            },
            SyncKey.SERVER_FQCN: "server/job",
        }
    )

    try:
        reply = executor.execute(SETUP_TASK_NAME, shareable, fl_ctx, MagicMock())
    finally:
        executor.thread_executor.shutdown()

    assert reply.get_return_code() == ReturnCode.OK
    calls = executor._prepare_client_proxy.call_args_list
    assert calls[0].args[4] == {"": {"first": []}}
    assert calls[1].args[4] == {"": {"second": []}}
