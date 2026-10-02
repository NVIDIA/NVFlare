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

from unittest.mock import Mock

import pytest

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE, AnalyticsDataType
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import EventScope, FLContextKey, FLMetaKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.executors.client_api import script_utils
from nvflare.app_common.executors.client_api.backend_spec import CLIENT_API_BACKEND_FACTORY
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.widgets.streaming import AnalyticsSender
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.private.fed.client import task_worker_client_api as client_api
from nvflare.private.fed.client.task_worker_client_api import TaskClientAPI, TaskClientAPIBackend
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity
from nvflare.private.fed.task_worker.runtime import TaskRuntime


def test_client_binding_supplies_only_client_context_and_backend_services():
    engine = Mock()
    fl_ctx, store, analytics = FLContextManager(engine=engine).new_context(), Mock(), []
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    client_api.bind_client_task_context(fl_ctx, identity, store, analytics)
    assert fl_ctx.get_prop(FLContextKey.CLIENT_NAME) == identity.site_name
    assert fl_ctx.get_process_type() == "client_task_worker"
    factory = fl_ctx.get_prop(CLIENT_API_BACKEND_FACTORY)
    backend = factory()
    assert isinstance(backend, TaskClientAPIBackend)
    assert backend._store is store
    assert backend._identity is identity
    assert backend._analytics is analytics
    assert client_api.CLIENT_TASK_CONTEXT_KEYS == {
        FLContextKey.CLIENT_NAME,
        FLContextKey.PROCESS_TYPE,
        CLIENT_API_BACKEND_FACTORY,
    }
    assert fl_ctx.get_engine() is engine
    engine.add_event_observer.assert_called_once()


def test_task_client_api_requires_exactly_one_durable_send():
    store = Mock()
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    api = TaskClientAPI({}, store, identity)
    with pytest.raises(RuntimeError, match="without sending"):
        api.get_result()
    api._publish_result(Shareable())
    assert api.get_result() is store.read_script_result.return_value
    store.read_script_result.assert_called_once_with(identity, store.stage_script_result.return_value)
    assert not api.is_running()
    assert api.receive() is None
    with pytest.raises(RuntimeError, match="exactly one"):
        api._publish_result(Shareable())
    store.stage_script_result.assert_called_once()
    api.close()


def test_received_model_includes_job_and_site_metadata_and_starts_result_clock():
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    store = Mock()
    api = TaskClientAPI({FLMetaKey.JOB_ID: "job", FLMetaKey.SITE_NAME: "site"}, store, identity)
    api.init()
    try:
        api.stage_input(FLModelUtils.to_shareable(FLModel(params={"weight": 1})))
        store.mark_result_wait_started.assert_not_called()
        model = api.receive()
        assert model.meta[FLMetaKey.JOB_ID] == "job"
        assert model.meta[FLMetaKey.SITE_NAME] == "site"
        api.receive()
        store.mark_result_wait_started.assert_called_once_with(identity)
    finally:
        api.close()


def test_ordinary_executor_analytics_use_the_client_durable_event_binding():
    runtime = TaskRuntime(None, "site", "job")
    sender = AnalyticsSender()
    runtime.set_compute_graph({"sender": sender}, FLComponent())
    fl_ctx = runtime.new_context()
    records = []
    client_api.bind_client_task_context(
        fl_ctx, TaskAttemptIdentity("job", "site", "task", "train", "attempt"), Mock(), records
    )
    runtime.fire_event(EventType.ABOUT_TO_START_RUN, fl_ctx)
    sender.add("loss", 0.5, AnalyticsDataType.SCALAR, global_step=1)
    assert len(records) == 1
    assert records[0]["event_type"] == ANALYTIC_EVENT_TYPE
    assert records[0]["federated"] is False
    fl_ctx.set_prop(FLContextKey.EVENT_SCOPE, EventScope.FEDERATION, private=True, sticky=False)
    fl_ctx.set_prop(
        FLContextKey.EVENT_DATA,
        create_analytic_dxo("loss", 0.4, AnalyticsDataType.SCALAR).to_shareable(),
        private=True,
        sticky=False,
    )
    runtime.fire_event(ANALYTIC_EVENT_TYPE, fl_ctx)
    assert records[-1]["federated"] is True


def test_task_adapter_rejects_custom_client_api_subclass():
    class CustomExecutor(ClientAPIExecutor):
        pass

    executor = CustomExecutor(execution_mode="in_process", task_script_path="train.py")
    backend = TaskClientAPIBackend(None, None, [])
    with pytest.raises(RuntimeError, match="subclasses"):
        backend.initialize(executor._build_backend_context(), None)


def test_task_adapter_rejects_external_process_mode():
    executor = ClientAPIExecutor(execution_mode="external_process", command=["python", "train.py"])
    backend = TaskClientAPIBackend(None, None, [])
    with pytest.raises(RuntimeError, match="in_process scripts only"):
        backend.initialize(executor._build_backend_context(), None)


@pytest.fixture
def task_backend(monkeypatch):
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    api = Mock()
    api.get_result.return_value = Shareable({"result": 1})
    monkeypatch.setattr(client_api, "TaskClientAPI", lambda *_args: api)
    runner = Mock()
    monkeypatch.setattr(script_utils, "TaskScriptRunner", lambda **_kwargs: runner)
    backend = TaskClientAPIBackend(Mock(), identity, [])
    executor = ClientAPIExecutor(execution_mode="in_process", task_script_path="train.py")
    fl_ctx = Mock()
    bus = DataBus()
    previous_api = bus.get_data(CLIENT_API_KEY)
    try:
        yield backend, executor._build_backend_context(), fl_ctx, api, runner
    finally:
        backend.finalize(fl_ctx)
        bus.put_data(CLIENT_API_KEY, previous_api)


def test_task_backend_uses_lifecycle_and_preserves_other_bus_owner(task_backend):
    backend, context, fl_ctx, api, runner = task_backend
    backend.initialize(context, fl_ctx)
    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    assert result == Shareable({"result": 1})
    assert DataBus().get_data(CLIENT_API_KEY) is api
    runner.run.assert_called_once()
    backend.abort(fl_ctx)
    assert api.stop is True
    other_api = object()
    DataBus().put_data(CLIENT_API_KEY, other_api)
    backend.finalize(fl_ctx)
    backend.finalize(fl_ctx)
    api.close.assert_called_once()
    assert DataBus().get_data(CLIENT_API_KEY) is other_api


@pytest.mark.parametrize("setup_failure", ["api", "runner"])
def test_task_backend_initialization_unwinds_before_propagating_failure(task_backend, monkeypatch, setup_failure):
    backend, context, fl_ctx, api, _runner = task_backend
    if setup_failure == "api":
        api.init.side_effect = RuntimeError("setup failed")
    else:
        monkeypatch.setattr(script_utils, "TaskScriptRunner", Mock(side_effect=RuntimeError("setup failed")))
    with pytest.raises(RuntimeError, match="setup failed"):
        backend.initialize(context, fl_ctx)
    api.close.assert_called_once()
    assert backend._api is None


@pytest.mark.parametrize("failure", ["uninitialized", "wrong_task", "aborted", "repeated"])
def test_task_backend_rejects_invalid_assignment(task_backend, failure):
    backend, context, fl_ctx, _api, runner = task_backend
    signal = Signal()
    task_name = "train"
    if failure != "uninitialized":
        backend.initialize(context, fl_ctx)
    if failure == "wrong_task":
        task_name = "validate"
    elif failure == "aborted":
        signal.trigger(True)
    elif failure == "repeated":
        backend.execute(task_name, Shareable(), fl_ctx, signal)
    with pytest.raises(RuntimeError):
        backend.execute(task_name, Shareable(), fl_ctx, signal)
    assert runner.run.call_count == (1 if failure == "repeated" else 0)


@pytest.mark.parametrize("exit_code", [None, 0, 2])
def test_task_backend_handles_only_clean_script_system_exit(task_backend, exit_code):
    backend, context, fl_ctx, _api, runner = task_backend
    backend.initialize(context, fl_ctx)
    runner.run.side_effect = SystemExit(exit_code)
    if exit_code in (None, 0):
        assert backend.execute("train", Shareable(), fl_ctx, Signal())["result"] == 1
    else:
        with pytest.raises(SystemExit):
            backend.execute("train", Shareable(), fl_ctx, Signal())


def test_task_backend_finalization_failure_clears_owned_bus_entry(task_backend):
    backend, context, fl_ctx, api, _runner = task_backend
    backend.initialize(context, fl_ctx)
    backend.execute("train", Shareable(), fl_ctx, Signal())
    api.close.side_effect = RuntimeError("finalize failed")
    with pytest.raises(RuntimeError, match="finalize failed"):
        backend.finalize(fl_ctx)
    assert backend._api is None
    assert DataBus().get_data(CLIENT_API_KEY) is None
