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

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.shareable import Shareable
from nvflare.collab import collab
from nvflare.collab.runtime import executor as executor_module
from nvflare.collab.runtime.defs import SETUP_TASK_NAME, SYNC_TASK_NAME, SyncKey
from nvflare.collab.runtime.executor import CollabExecutor
from nvflare.job_config.fed_job_config import FedJobConfig


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
        executor._shutdown_call_executors()


def test_execute_returns_error_when_start_run_did_not_create_client_app():
    executor = CollabExecutor(client_obj_id="client")

    try:
        reply = executor.execute(SYNC_TASK_NAME, Shareable(), FLContext(), MagicMock())
        assert reply.get_return_code() == ReturnCode.ERROR
    finally:
        executor._shutdown_call_executors()


def test_inbound_and_outbound_calls_use_separate_executors():
    executor = CollabExecutor(
        client_obj_id="client",
        max_call_threads=1,
        max_inbound_call_threads=2,
        max_outbound_call_threads=3,
    )

    try:
        assert executor.inbound_executor is not executor.outbound_executor
        assert executor.inbound_executor._max_workers == 2
        assert executor.outbound_executor._max_workers == 3
    finally:
        executor._shutdown_call_executors()


def test_executor_serializes_independent_pool_sizes_and_collab_objects(tmp_path):
    executor = CollabExecutor(
        client_obj_id="client",
        collab_obj_ids=["extra"],
        max_inbound_call_threads=2,
        max_outbound_call_threads=3,
    )

    try:
        args = FedJobConfig("job", 1)._get_args(executor, str(tmp_path))
    finally:
        executor._shutdown_call_executors()

    assert args["collab_obj_ids"] == ["extra"]
    assert args["max_inbound_call_threads"] == 2
    assert args["max_outbound_call_threads"] == 3


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


def test_end_run_shuts_down_call_executors_when_finalize_raises():
    executor = CollabExecutor(client_obj_id="client")
    executor.client_app = MagicMock()
    executor.client_app.name = "site-1"
    executor.client_app.finalize.side_effect = RuntimeError("finalize failed")
    executor.client_ctx = MagicMock()
    executor.inbound_executor.shutdown = MagicMock(wraps=executor.inbound_executor.shutdown)
    executor.outbound_executor.shutdown = MagicMock(wraps=executor.outbound_executor.shutdown)

    with pytest.raises(RuntimeError, match="finalize failed"):
        executor._handle_end_run("end_run", FLContext())

    executor.inbound_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
    executor.outbound_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


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
        executor._shutdown_call_executors()

    assert reply.get_return_code() == ReturnCode.OK
    calls = executor._prepare_client_proxy.call_args_list
    assert calls[0].args[4] == {"": {"first": []}}
    assert calls[1].args[4] == {"": {"second": []}}


def test_client_proxy_builds_children_from_remote_interface():
    executor = CollabExecutor(client_obj_id="client")
    executor.client_app = MagicMock()
    executor.client_app.name = "site-1"
    client = MagicMock()
    client.name = "site-2"
    client.get_fqcn.return_value = "site-2"
    client.get_fqsn.return_value = "site-2"
    remote_interface = {
        "": {"train": []},
        "remote_only": {"evaluate": ["model"]},
    }

    try:
        proxy = executor._prepare_client_proxy(
            "job",
            MagicMock(),
            client,
            MagicMock(),
            remote_interface,
            MagicMock(),
        )
    finally:
        executor._shutdown_call_executors()

    assert set(proxy.children) == {"remote_only"}
    assert proxy.remote_only.target_interface.to_dict() == {"evaluate": ["model"]}
    assert proxy.backend.thread_executor is executor.outbound_executor


def test_external_setup_starts_distributed_session_without_initializing_parent_app(monkeypatch):
    client = MagicMock()
    client.name = "site-1"
    client.get_fqcn.return_value = "site-1"
    client.get_fqsn.return_value = "site-1"
    session = MagicMock()
    session_type = MagicMock(return_value=session)
    monkeypatch.setattr(executor_module, "from_dict", lambda value: value)
    monkeypatch.setattr(executor_module, "DistributedClientSession", session_type)
    prepare = MagicMock()
    monkeypatch.setattr(executor_module, "prepare_for_distributed_call", prepare)

    executor = CollabExecutor(
        client_obj_id="client",
        collab_obj_ids=["extra"],
        props={"learning_rate": 0.1},
        launch_external_process=True,
        command="python3 -m torch.distributed.run --nproc_per_node=2",
    )
    executor.client_app = MagicMock()
    executor.client_app.name = "site-1"
    executor.client_app.get_collab_interface.return_value = {"": {"train": []}}
    executor.log_info = MagicMock()

    engine = MagicMock()
    cell = engine.get_cell.return_value
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    fl_ctx.get_identity_name.return_value = "site-1"
    fl_ctx.get_job_id.return_value = "job"
    fl_ctx.get_prop.side_effect = lambda key, default=None: {
        FLContextKey.JOB_META: {JobMetaKey.JOB_CLIENTS: [client]}
    }.get(key, default)
    shareable = Shareable(
        {
            SyncKey.COLLAB_INTERFACE: {"": {"run": []}},
            SyncKey.CLIENT_INTERFACES: {"site-1": {"": {"train": []}}},
            SyncKey.SERVER_FQCN: "server/job",
        }
    )

    try:
        reply = executor.execute(SETUP_TASK_NAME, shareable, fl_ctx, MagicMock())
    finally:
        executor._shutdown_call_executors()

    assert reply.get_return_code() == ReturnCode.OK
    session.start.assert_called_once()
    assert session.start.call_args.kwargs["client_obj_id"] == "client"
    assert session.start.call_args.kwargs["collab_obj_ids"] == ["extra"]
    assert session.start.call_args.kwargs["server_spec"]["target"] == "server/job"
    assert session.start.call_args.kwargs["fl_ctx"] is fl_ctx
    executor.client_app.setup.assert_not_called()
    executor.client_app.initialize.assert_not_called()
    prepare.assert_called_once_with(
        cell=cell,
        session=session,
        logger=executor.logger,
        executor=executor.inbound_executor,
    )


def test_end_run_finalizes_distributed_session_instead_of_parent_app():
    executor = CollabExecutor(client_obj_id="client", launch_external_process=True)
    executor.client_app = MagicMock()
    executor.client_app.name = "site-1"
    executor.distributed_session = MagicMock()

    executor._handle_end_run("end_run", FLContext())

    executor.distributed_session.stop.assert_called_once_with(finalize=True)
    executor.client_app.finalize.assert_not_called()


def test_end_run_shuts_down_call_executors_when_distributed_finalize_fails():
    executor = CollabExecutor(client_obj_id="client", launch_external_process=True)
    executor.client_app = MagicMock()
    executor.client_app.name = "site-1"
    executor.distributed_session = MagicMock()
    executor.distributed_session.stop.side_effect = RuntimeError("rank 1 finalization failed")
    executor.inbound_executor.shutdown = MagicMock(wraps=executor.inbound_executor.shutdown)
    executor.outbound_executor.shutdown = MagicMock(wraps=executor.outbound_executor.shutdown)

    with pytest.raises(RuntimeError, match="rank 1 finalization failed"):
        executor._handle_end_run("end_run", FLContext())

    executor.inbound_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
    executor.outbound_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


@pytest.mark.parametrize("failing_method", ["setup", "initialize"])
def test_setup_returns_error_without_publishing_context_when_client_setup_fails(monkeypatch, failing_method):
    client = MagicMock()
    client.name = "site-1"
    monkeypatch.setattr(executor_module, "from_dict", lambda client: client)
    monkeypatch.setattr(executor_module, "prepare_for_remote_call", MagicMock())

    executor = CollabExecutor(client_obj_id="client")
    executor.client_app = MagicMock()
    executor.client_app.name = "site-1"
    executor.client_app.get_collab_interface.return_value = {"": {"local": []}}
    getattr(executor.client_app, failing_method).side_effect = RuntimeError(f"{failing_method} failed")
    executor.log_info = MagicMock()
    executor.log_exception = MagicMock()
    executor._prepare_server_proxy = MagicMock(return_value=MagicMock())
    executor._prepare_client_proxy = MagicMock(return_value=MagicMock())

    fl_ctx = MagicMock()
    fl_ctx.get_prop.side_effect = lambda key, default=None: {
        FLContextKey.JOB_META: {JobMetaKey.JOB_CLIENTS: [client]}
    }.get(key, default)
    shareable = Shareable(
        {
            SyncKey.COLLAB_INTERFACE: {"": {"server": []}},
            SyncKey.CLIENT_INTERFACES: {"site-1": {"": {"local": []}}},
            SyncKey.SERVER_FQCN: "server/job",
        }
    )

    try:
        reply = executor.execute(SETUP_TASK_NAME, shareable, fl_ctx, MagicMock())
    finally:
        executor._shutdown_call_executors()

    assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert executor.client_ctx is None
    executor.log_exception.assert_called_once()
