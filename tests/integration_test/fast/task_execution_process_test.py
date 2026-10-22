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


import gc
import importlib
import json
import os
import sys
import textwrap
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReturnCode, TaskResultReceipt
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskExecutionStatus, TaskLauncherSpec
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.widgets.decomposer_reg import DecomposerRegister
from nvflare.client.in_process.adapters import TaskInProcessAdapter
from nvflare.job_config.task_execution import prepare_task_execution
from nvflare.private.fed.client.task_worker_executor import TASK_ATTEMPT_IDENTITY, TaskWorkerExecutor
from nvflare.private.fed.task_worker.artifacts import ArtifactReference, FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import read_bootstrap
from tests.task_execution_test_utils import SCRIPT, _adapter, _config, _statistics_config
from tests.task_execution_test_utils import native_task as _native_task

# Register the shared fixture in this collected module only.
native_task = _native_task


def test_script_owned_dataloader_is_recreated_for_each_assignment(native_task):
    from nvflare.app_common.utils.fl_model_utils import FLModelUtils

    _, custom, context, policy = native_task
    script = """
import uuid
import nvflare.client as flare

class DataLoader:
    def __init__(self):
        self.instance = str(uuid.uuid4())
        self.calls = 0

    def load(self):
        self.calls += 1
        return self.calls

flare.init()
loader = DataLoader()
model = flare.receive()
flare.send(flare.FLModel(params=model.params,
    metrics={"calls": loader.load(), "instance": loader.instance}, current_round=model.current_round))
"""
    adapter, data = _adapter(custom, context, policy, script=script)
    results = []
    try:
        for attempt in (1, 2):
            context.set_prop(FLContextKey.TASK_ID, f"task-{attempt}", private=True, sticky=True)
            context.set_prop(FLContextKey.TASK_ATTEMPT_ID, f"attempt-{attempt}", private=True, sticky=True)
            results.append(FLModelUtils.from_shareable(adapter.execute("train", data, context, Signal())).metrics)
        assert [result["calls"] for result in results] == [1, 1]
        assert results[0]["instance"] != results[1]["instance"]
    finally:
        adapter.handle_event(EventType.END_RUN, context)


@pytest.mark.parametrize("transfer", ["FULL", "DIFF"])
def test_real_task_script_runs_once_on_main_thread_with_current_round(native_task, transfer):
    root, custom, context, policy = native_task
    adapter, data = _adapter(
        custom,
        context,
        policy,
        params_transfer_type=transfer,
        task_script_args=["--label", "two words", "${secret:TASK_TEST_SECRET}"],
    )
    result = adapter.execute("train", data, context, Signal())
    dxo = from_shareable(result)
    np.testing.assert_array_equal(dxo.data["w"], [2, 3] if transfer == "FULL" else [1, 1])
    assert result.get_header(AppConstants.CURRENT_ROUND) == 3
    # Client API stores metrics in DXO metadata.
    from nvflare.app_common.utils.fl_model_utils import FLModelUtils

    metrics = FLModelUtils.from_shareable(result).metrics
    assert metrics["pid"] != os.getpid()
    assert metrics["main_thread"] is True
    assert metrics["argv"] == ["--label", "two words", "local-placeholder"]
    assert (metrics["site"], metrics["job"]) == ("site-1", "job-1")
    assert metrics["metadata"][FLMetaKey.JOB_ID] == "job-1"
    assert metrics["metadata"][FLMetaKey.SITE_NAME] == "site-1"
    assert context.get_engine().fire_event.call_count > 0
    identity = context.get_prop(TASK_ATTEMPT_IDENTITY)
    payload = root / "job-1" / ".nvflare" / "task-execution" / "attempts"
    files = list(payload.rglob("result.fobs"))
    assert len(files) == 1
    context.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
    context.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED, private=True, sticky=False)
    adapter.handle_event(EventType.AFTER_SEND_TASK_RESULT, context)
    assert not files[0].exists()
    adapter.handle_event(EventType.END_RUN, context)
    assert identity.attempt_id == "attempt-1"


def test_task_send_writes_to_disk_without_copying_or_retaining_the_model(native_task):
    from nvflare.private.fed.task_worker import TaskAttemptIdentity

    root, _custom, _context, _policy = native_task
    store = FileTaskArtifactStore(str(root / "artifacts"))
    identity = TaskAttemptIdentity("job-1", "site-1", "task-1", "train", "disk-send")
    directory = Path(store.create_attempt(identity))
    adapter = TaskInProcessAdapter(on_result=lambda data: store.stage_result(identity, data))
    adapter.bind(SimpleNamespace(_set_received_shareable=Mock()), None, None)
    incoming = Shareable()
    incoming.set_header(AppConstants.CURRENT_ROUND, 3)
    adapter.stage_input(incoming)
    weights = np.ones(1024 * 1024, dtype=np.float32)
    data = DXO(DataKind.WEIGHTS, {"w": weights}).to_shareable()
    data.__deepcopy__ = Mock(side_effect=AssertionError("send must serialize instead of copying"))
    weak_weights = weakref.ref(weights)
    adapter.publish_result(data)
    reference = adapter.get_result()
    assert isinstance(reference, ArtifactReference)
    assert (directory / "result.fobs").is_file()
    assert not (directory / "completion.json").exists()
    weights[0] = 99
    with pytest.raises(RuntimeError, match="exactly one result"):
        adapter.publish_result(data)
    del weights, data
    gc.collect()
    assert weak_weights() is None
    store.commit_staged_result(identity, reference)
    restored, _completion = store.read_result(identity)
    assert from_shareable(restored).data["w"][0] == 1
    assert restored.get_header(AppConstants.CURRENT_ROUND) == 3


def test_script_can_observe_durable_send_before_exit_and_cannot_mutate_it(native_task):
    root, custom, context, policy = native_task
    directory = root / "job-1" / ".nvflare" / "task-execution" / "attempts" / "attempt-1"
    script = f"""
from pathlib import Path
import nvflare.client as flare
flare.init()
model = flare.receive()
result = flare.FLModel(params=model.params)
flare.send(result, clear_cache=False)
directory = Path({str(directory)!r})
assert (directory / "result.fobs").is_file(), "send did not persist the result"
assert not (directory / "completion.json").exists(), "send completed the attempt prematurely"
result.params["w"][0] = 99
"""
    adapter, data = _adapter(custom, context, policy, script=script)
    try:
        result = adapter.execute("train", data, context, Signal())
        np.testing.assert_array_equal(from_shareable(result).data["w"], [1, 2])
    finally:
        adapter.handle_event(EventType.END_RUN, context)


@pytest.mark.parametrize(
    "script",
    [
        "import nvflare.client as flare\nflare.init()\nflare.receive()\n",
        SCRIPT + "\nraise RuntimeError('failure after send')\n",
    ],
)
def test_script_failure_cannot_publish_completion(native_task, script):
    root, custom, context, policy = native_task
    adapter, data = _adapter(custom, context, policy, script=script)
    with pytest.raises(RuntimeError, match="task worker .* failed:"):
        adapter.execute("train", data, context, Signal())
    adapter.handle_event(EventType.END_RUN, context)
    assert not list((root / "job-1" / ".nvflare").rglob("completion.json"))
    assert list((root / "job-1" / ".nvflare").rglob("failure.json"))
    assert bool(list((root / "job-1" / ".nvflare").rglob("result.fobs"))) == ("failure after send" in script)


def test_result_wait_timeout_cancels_the_real_worker(native_task):
    root, custom, context, policy = native_task
    script = "import time\nimport nvflare.client as flare\nflare.init()\nflare.receive()\ntime.sleep(10)\n"
    adapter, data = _adapter(custom, context, policy, script=script, result_wait_timeout=0.1)
    with pytest.raises(TimeoutError, match="timed out"):
        adapter.execute("train", data, context, Signal())
    adapter.handle_event(EventType.END_RUN, context)
    assert not adapter.supervisor.settlement_unconfirmed
    assert not list((root / "job-1" / ".nvflare").rglob("completion.json"))


def test_result_wait_timeout_includes_script_finalization(native_task):
    _, custom, context, policy = native_task
    adapter, data = _adapter(
        custom, context, policy, script=SCRIPT + "\nimport time\ntime.sleep(0.4)\n", result_wait_timeout=0.2
    )
    with pytest.raises(TimeoutError, match="timed out"):
        adapter.execute("train", data, context, Signal())
    adapter.handle_event(EventType.END_RUN, context)


def test_default_post_send_bound_stops_a_script_even_without_result_wait_timeout(native_task):
    _, custom, context, policy = native_task
    assert policy.finalization_timeout == 30
    policy = replace(policy, finalization_timeout=0.1)
    adapter, data = _adapter(custom, context, policy, script=SCRIPT + "\nimport time\ntime.sleep(10)\n")
    with pytest.raises(TimeoutError, match="timed out"):
        adapter.execute("train", data, context, Signal())
    adapter.handle_event(EventType.END_RUN, context)
    assert not adapter.supervisor.settlement_unconfirmed


def test_slow_result_and_checkpoint_staging_do_not_consume_application_deadlines(native_task):
    _, custom, context, policy = native_task
    policy = replace(policy, finalization_timeout=0.2)
    script = (
        """
import time
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
original_result = FileTaskArtifactStore.stage_result
original_checkpoint = FileTaskArtifactStore.stage_checkpoint
def slow_result(self, *args, **kwargs):
    time.sleep(0.6)
    return original_result(self, *args, **kwargs)
def slow_checkpoint(self, *args, **kwargs):
    time.sleep(0.6)
    return original_checkpoint(self, *args, **kwargs)
FileTaskArtifactStore.stage_result = slow_result
FileTaskArtifactStore.stage_checkpoint = slow_checkpoint
"""
        + SCRIPT
    )
    adapter, data = _adapter(custom, context, policy, script=script, result_wait_timeout=0.3)
    try:
        result = adapter.execute("train", data, context, Signal())
        np.testing.assert_array_equal(from_shareable(result).data["w"], [2, 3])
    finally:
        adapter.handle_event(EventType.END_RUN, context)


def test_server_resends_saved_results_without_relaunch_replay_or_checkpoint_rollback(native_task):
    from nvflare.app_common.utils.fl_model_utils import FLModelUtils

    root, custom, context, policy = native_task
    script = """
import os
from pathlib import Path
import nvflare.client as flare
flare.init()
model = flare.receive()
paths = flare.system_info()["task_checkpoint"]
previous = int(Path(paths["input"]).read_text()) if paths["input"] else 0
Path(paths["output"]).write_text(str(previous + 1))
flare.send(flare.FLModel(params=model.params, metrics={"previous": previous, "pid": os.getpid()}))
"""
    adapter, data = _adapter(custom, context, policy, script=script)
    replay = adapter._replay_analytics = Mock(wraps=adapter._replay_analytics)
    launch = policy.launcher.launch_task = Mock(wraps=policy.launcher.launch_task)

    def execute(number):
        context.set_prop(FLContextKey.TASK_ID, f"task-{number}", private=True, sticky=True)
        context.set_prop(FLContextKey.TASK_ATTEMPT_ID, f"attempt-{number}", private=True, sticky=True)
        return FLModelUtils.from_shareable(adapter.execute("train", data, context, Signal()))

    try:
        first = execute(1)
        checkpoint = next(root.rglob("*.checkpoint"))
        before = checkpoint.read_bytes()
        assert execute(1).metrics == first.metrics
        assert checkpoint.read_bytes() == before
        assert (launch.call_count, replay.call_count) == (1, 1)
        second = execute(2)
        assert second.metrics["previous"] == 1
        before = checkpoint.read_bytes()
        assert execute(1).metrics == first.metrics
        assert checkpoint.read_bytes() == before
        assert execute(2).metrics == second.metrics
        assert (launch.call_count, replay.call_count) == (2, 2)
        assert execute(3).metrics["previous"] == 2
        assert (launch.call_count, replay.call_count) == (3, 3)
    finally:
        adapter.handle_event(EventType.END_RUN, context)


@pytest.mark.parametrize("event", [EventType.END_RUN, EventType.ABORT_TASK])
def test_shutdown_or_abort_during_result_decoding_keeps_cancellation_without_false_panic(
    native_task, monkeypatch, event
):
    _, _, context, policy = native_task

    class SettledLauncher(TaskLauncherSpec):
        def launch_task(self, request):
            bootstrap = read_bootstrap(request.argv[4])
            FileTaskArtifactStore(bootstrap.artifact_root).commit_result(bootstrap.identity, Shareable({"done": True}))
            status = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=True)
            return SimpleNamespace(
                request=request,
                execution_id="settled-probe",
                poll=lambda: status,
                wait_for_settlement=lambda **_: status,
                cancel=Mock(return_value=status),
            )

    reading, proceed = threading.Event(), threading.Event()
    original = FileTaskArtifactStore.read_result

    def blocked_read(store, identity):
        reading.set()
        assert proceed.wait(5)
        return original(store, identity)

    monkeypatch.setattr(FileTaskArtifactStore, "read_result", blocked_read)
    adapter = TaskWorkerExecutor(
        {"path": "unused.Executor"}, (), replace(policy, launcher=SettledLauncher(), settlement_timeout=0.01)
    )
    adapter.system_panic = Mock()
    adapter.handle_event(EventType.START_RUN, context)
    with ThreadPoolExecutor() as pool:
        running = pool.submit(adapter.execute, "train", Shareable(), context, Signal())
        try:
            assert reading.wait(5)
            assert not adapter.supervisor.owns_execution
            adapter.handle_event(event, context)
            adapter.system_panic.assert_not_called()
        finally:
            proceed.set()
        assert running.result(timeout=5).get_return_code() == ReturnCode.TASK_ABORTED


def test_worker_log_is_replayed_through_cj_logging(native_task):
    root, custom, context, policy = native_task
    (root / "local" / "log_config.json").write_text(
        json.dumps(
            {
                "version": 1,
                "handlers": {"file": {"class": "logging.FileHandler", "filename": "log.txt"}},
                "root": {"level": "INFO", "handlers": ["file"]},
            }
        )
    )
    adapter, data = _adapter(
        custom, context, policy, script="import logging\nlogging.warning('trainer probe log')\n" + SCRIPT
    )
    adapter.log_info = Mock()
    adapter.execute("train", data, context, Signal())
    assert any("trainer probe log" in str(call.args) for call in adapter.log_info.call_args_list)
    adapter.handle_event(EventType.END_RUN, context)


def test_registered_decomposer_is_available_before_worker_input_decode(native_task, monkeypatch):
    _, custom, context, policy = native_task
    (custom / "task_codec.py").write_text(
        """
from nvflare.fuel.utils.fobs import Decomposer

class Counter:
    def __init__(self, value):
        self.value = value

class CounterDecomposer(Decomposer):
    def supported_type(self):
        return Counter

    def decompose(self, obj, manager):
        return obj.value

    def recompose(self, data, manager):
        return Counter(data)
"""
    )
    monkeypatch.syspath_prepend(str(custom))
    codec = importlib.import_module("task_codec")
    monkeypatch.setitem(sys.modules, "task_codec", codec)
    # This is the same registration the resident CJ component performs.
    DecomposerRegister(["task_codec.CounterDecomposer"])
    (custom / "train.py").write_text(
        """
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel
flare.init()
model = flare.receive()
flare.send(FLModel(metrics={"decoded": model.meta["custom"].value}))
"""
    )
    config = _config(params_exchange_format="numpy", server_expected_format="numpy")
    config["components"] = [
        {
            "id": "codec",
            "path": "nvflare.app_common.widgets.decomposer_reg.DecomposerRegister",
            "args": {"classes": ["task_codec.CounterDecomposer"]},
        }
    ]
    plan = prepare_task_execution(config).executors[0]
    adapter = TaskWorkerExecutor(plan.executor, plan.components, policy)
    adapter.handle_event(EventType.START_RUN, context)
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.app_common.utils.fl_model_utils import FLModelUtils

    data = FLModelUtils.to_shareable(FLModel(params={"w": np.ones(2)}, meta={"custom": codec.Counter(41)}))
    try:
        result = adapter.execute("train", data, context, Signal())
        assert FLModelUtils.from_shareable(result).metrics["decoded"] == 41
    finally:
        adapter.handle_event(EventType.END_RUN, context)


@pytest.mark.parametrize("write_failure", [None, "result", "analytics"])
def test_client_api_worker_runs_directly_without_event_dispatch(native_task, monkeypatch, write_failure):
    import nvflare.client.api as client_api
    from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
    from nvflare.private.fed.client.client_api_task_runner import bind_client_task_context
    from nvflare.private.fed.task_worker import (
        FileTaskArtifactStore,
        TaskAttemptIdentity,
        WorkerBootstrap,
        worker,
        write_bootstrap,
    )

    root, custom, context, _policy = native_task
    monkeypatch.setattr(client_api, "context_dict", {})
    monkeypatch.setattr(client_api, "default_context", None)
    # A script may catch an I/O error, but it cannot make failed persistence
    # look like a successfully completed assignment.
    (custom / "train.py").write_text("try:\n" + textwrap.indent(SCRIPT, "    ") + "\nexcept OSError:\n    pass\n")
    executor = ClientAPIExecutor(
        execution_mode="in_process",
        task_script_path="train.py",
        params_exchange_format="numpy",
        server_expected_format="numpy",
    )
    handler = Mock(side_effect=AssertionError("executor events must not run in a task worker"))
    monkeypatch.setattr(executor, "handle_event", handler)
    monkeypatch.setattr("nvflare.apis.utils.event.fire_event_to_components", handler)

    def build_executor(_config):
        return executor

    monkeypatch.setattr(worker, "_build_executor", build_executor)
    identity = TaskAttemptIdentity("job-1", "site-1", "task-1", "train", "direct-api")
    store = FileTaskArtifactStore(str(root / "artifacts"))
    store.create_attempt(identity)
    data = DXO(DataKind.WEIGHTS, {"w": np.array([1.0])}).to_shareable()
    data.set_header(AppConstants.CURRENT_ROUND, 3)
    store.write_input(identity, data)
    path = store.bootstrap_path(identity)
    write_bootstrap(path, WorkerBootstrap(identity, store.root_dir, str(root), _config()["executors"][0]["executor"]))
    if write_failure is not None:
        method = "stage_result" if write_failure == "result" else "write_analytics_record"
        monkeypatch.setattr(FileTaskArtifactStore, method, Mock(side_effect=OSError("cannot persist task output")))
        with pytest.raises(OSError, match="cannot persist task output"):
            worker.run_worker(path, context_binding=bind_client_task_context)
        assert not Path(store.attempt_dir(identity), "completion.json").exists()
        assert store.read_failure(identity)["error_type"] == "OSError"
        return

    stage_result = FileTaskArtifactStore.stage_result
    staged = []

    def stage_once(artifact_store, attempt, data):
        assert Path(artifact_store.attempt_dir(attempt), "result_staging_started.json").is_file()
        assert not Path(artifact_store.attempt_dir(attempt), "script_result_sent.json").exists()
        reference = stage_result(artifact_store, attempt, data)
        staged.append(reference)
        return reference

    read_payload = FileTaskArtifactStore._read_payload

    def read_input_only(artifact_store, attempt, reference):
        assert reference.kind == "input", "the worker must not reload its saved result or analytics"
        return read_payload(artifact_store, attempt, reference)

    monkeypatch.setattr(FileTaskArtifactStore, "stage_result", stage_once)
    monkeypatch.setattr(FileTaskArtifactStore, "_read_payload", read_input_only)
    worker.run_worker(path, context_binding=bind_client_task_context)
    assert len(staged) == 1
    assert Path(store.attempt_dir(identity), "script_result_sent.json").is_file()
    assert Path(store.attempt_dir(identity), "script_finalized.json").is_file()
    monkeypatch.setattr(FileTaskArtifactStore, "_read_payload", read_payload)
    result, completion = store.read_result(identity)
    np.testing.assert_array_equal(from_shareable(result).data["w"], [2.0])
    assert "analytics" in completion.diagnostics
    records = store.read_analytics(identity, completion)
    assert len(records) == 1
    assert from_shareable(records[0]["data"]).data["track_value"] == 0.75
    handler.assert_not_called()


def test_task_runner_keeps_initialization_error_when_cleanup_also_fails(native_task, monkeypatch):
    from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
    from nvflare.client.in_process.api import InProcessClientAPI
    from nvflare.private.fed.client import client_api_task_runner

    _root, _custom, context, _policy = native_task
    executor = ClientAPIExecutor(
        execution_mode="in_process",
        task_script_path="train.py",
        params_exchange_format="numpy",
        server_expected_format="numpy",
    )
    monkeypatch.setattr(
        client_api_task_runner, "TaskScriptRunner", Mock(side_effect=ValueError("primary setup failure"))
    )
    close = InProcessClientAPI.close

    def close_then_fail(api):
        close(api)
        raise OSError("secondary cleanup failure")

    monkeypatch.setattr(InProcessClientAPI, "close", close_then_fail)
    runner = client_api_task_runner._ClientAPITaskRunner(executor)
    with pytest.raises(ValueError, match="primary setup failure"):
        runner.initialize(context)
    assert runner._api is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_statistics_generator_has_fresh_direct_lifecycle_for_each_assignment(native_task, wrapped):
    from nvflare.app_common.abstract.statistics_spec import StatisticConfig
    from nvflare.app_common.app_constant import StatisticsConstants as StC
    from nvflare.app_common.statistics.statisitcs_objects_decomposer import fobs_registration
    from nvflare.fuel.utils import fobs

    root, custom, context, policy = native_task
    marker = root / "stats_lifecycle.jsonl"
    (custom / "stats_probe.py").write_text(
        """
import json, os
from nvflare.app_common.abstract.statistics_spec import Statistics, Feature, DataType

class ProbeStatistics(Statistics):
    def __init__(self, marker):
        super().__init__()
        self.marker = marker
        self.calls = 0

    def record(self, event):
        with open(self.marker, "a") as out:
            out.write(json.dumps([event, os.getpid(), self.calls]) + "\\n")

    def initialize(self, fl_ctx):
        self.record("initialize")

    def features(self):
        return {"train": [Feature("x", DataType.INT)]}

    def count(self, dataset_name, feature_name):
        self.calls += 1
        return 8

    def finalize(self, fl_ctx):
        self.record("finalize")
"""
    )
    config = _statistics_config()
    config["components"][0]["args"] = {"marker": str(marker)}
    if wrapped:
        config["components"][0]["args"] = {"class_attributes": config["components"][0]["args"]}
        executor = config["executors"][0]["executor"]
        executor["args"] = {"class_attributes": executor["args"]}
    plan = prepare_task_execution(config).executors[0]
    adapter = TaskWorkerExecutor(plan.executor, plan.components, policy)
    adapter.handle_event(EventType.START_RUN, context)
    fobs_registration()
    data = Shareable(
        {
            StC.STATISTICS_TASK_KEY: StC.STATS_1st_STATISTICS,
            StC.STATS_TARGET_STATISTICS: fobs.dumps([StatisticConfig("count", {})]),
        }
    )
    try:
        for attempt in (1, 2):
            context.set_prop(FLContextKey.TASK_ID, f"stats-task-{attempt}", private=True, sticky=True)
            context.set_prop(FLContextKey.TASK_ATTEMPT_ID, f"stats-attempt-{attempt}", private=True, sticky=True)
            result = adapter.execute(StC.FED_STATS_TASK, data, context, Signal())
            assert result.get_return_code() == ReturnCode.OK
            dxo = from_shareable(result)
            assert dxo.data_kind == DataKind.STATISTICS
            stats = fobs.loads(dxo.data[StC.STATS_1st_STATISTICS])
            assert stats["count"] == {"train": {"x": 8}}
            assert stats["failure_count"] == {"train": {"x": 0}}
            assert fobs.loads(dxo.data[StC.STATS_FEATURES])["train"][0].feature_name == "x"
    finally:
        adapter.handle_event(EventType.END_RUN, context)
    events = [json.loads(line) for line in marker.read_text().splitlines()]
    assert [entry[0] for entry in events] == ["initialize", "finalize", "initialize", "finalize"]
    assert [entry[2] for entry in events] == [0, 1, 0, 1]
    assert events[0][1] == events[1][1]
    assert events[2][1] == events[3][1]
    assert events[0][1] != events[2][1]
    assert all(entry[1] != os.getpid() for entry in events)


@pytest.mark.parametrize("failure", ["script", "finalizer", "replay"])
def test_failed_worker_checkpoint_is_not_promoted_and_retry_uses_previous_state(native_task, failure):
    from nvflare.app_common.utils.fl_model_utils import FLModelUtils

    root, custom, context, policy = native_task
    script = """
from pathlib import Path
import nvflare.client as flare
flare.init()
model = flare.receive()
paths = flare.system_info()["task_checkpoint"]
previous = int(Path(paths["input"]).read_text()) if paths["input"] else 0
Path(paths["output"]).write_text(str(previous + 1))
if model.meta.get("fail") == "script":
    raise RuntimeError("checkpoint script failure")
if model.meta.get("fail") == "finalizer":
    from nvflare.private.fed.client.client_api_task_runner import _ClientAPITaskRunner
    original = _ClientAPITaskRunner.finalize
    def fail_finalize(self, fl_ctx):
        original(self, fl_ctx)
        raise RuntimeError("checkpoint finalizer failure")
    _ClientAPITaskRunner.finalize = fail_finalize
flare.send(flare.FLModel(params=model.params, metrics={"previous": previous, "next": previous + 1}))
"""
    adapter, _ = _adapter(custom, context, policy, script=script)

    def execute(attempt, failed=False):
        context.set_prop(FLContextKey.TASK_ID, f"task-{attempt}", private=True, sticky=True)
        context.set_prop(FLContextKey.TASK_ATTEMPT_ID, f"attempt-{attempt}", private=True, sticky=True)
        data = DXO(DataKind.WEIGHTS, {"w": np.array([1.0])}, meta={"fail": failure} if failed else {}).to_shareable()
        replay = adapter._replay_worker_log
        if failed and failure == "replay":
            adapter._replay_worker_log = Mock(side_effect=RuntimeError("checkpoint replay failure"))
        try:
            return FLModelUtils.from_shareable(adapter.execute("train", data, context, Signal()))
        finally:
            adapter._replay_worker_log = replay

    try:
        assert execute(1).metrics == {"previous": 0, "next": 1}
        checkpoint = next(root.rglob("*.checkpoint"))
        previous = checkpoint.read_bytes()
        with pytest.raises(RuntimeError, match=f"checkpoint {failure} failure"):
            execute(2, failed=True)
        assert checkpoint.read_bytes() == previous
        completion_exists = (
            root / "job-1" / ".nvflare" / "task-execution" / "attempts" / "attempt-2" / "completion.json"
        ).exists()
        assert completion_exists == (failure == "replay")
        assert execute(3).metrics == {"previous": 1, "next": 2}
        assert len(list(root.rglob("*.checkpoint"))) == 1
    finally:
        adapter.handle_event(EventType.END_RUN, context)


def test_checkpoint_directory_sync_failure_returns_matching_result_and_next_worker_resumes(native_task, monkeypatch):
    from nvflare.app_common.utils.fl_model_utils import FLModelUtils
    from nvflare.private.fed.task_worker import artifacts

    root, custom, context, policy = native_task
    script = """
from pathlib import Path
import nvflare.client as flare
flare.init()
model = flare.receive()
paths = flare.system_info()["task_checkpoint"]
previous = int(Path(paths["input"]).read_text()) if paths["input"] else 0
Path(paths["output"]).write_text(str(previous + 1))
flare.send(flare.FLModel(params=model.params, metrics={"previous": previous, "next": previous + 1}))
"""
    adapter, _ = _adapter(custom, context, policy, script=script)

    def execute(attempt):
        context.set_prop(FLContextKey.TASK_ID, f"task-{attempt}", private=True, sticky=True)
        context.set_prop(FLContextKey.TASK_ATTEMPT_ID, f"attempt-{attempt}", private=True, sticky=True)
        data = DXO(DataKind.WEIGHTS, {"w": np.array([1.0])}).to_shareable()
        return FLModelUtils.from_shareable(adapter.execute("train", data, context, Signal()))

    replace_file, sync = os.replace, os.fsync
    published = failed = False

    def publish(*args, **kwargs):
        nonlocal published
        replace_file(*args, **kwargs)
        published = True

    def fail_directory_sync(fd):
        nonlocal failed
        if published:
            failed = True
            raise OSError("checkpoint directory sync failed after replacement")
        return sync(fd)

    try:
        assert execute(1).metrics == {"previous": 0, "next": 1}
        with monkeypatch.context() as patch:
            patch.setattr(artifacts.os, "replace", publish)
            patch.setattr(artifacts.os, "fsync", fail_directory_sync)
            assert execute(2).metrics == {"previous": 1, "next": 2}
        assert failed
        assert execute(3).metrics == {"previous": 2, "next": 3}
        assert len(list(root.rglob("*.checkpoint"))) == 1
    finally:
        adapter.handle_event(EventType.END_RUN, context)


@pytest.mark.parametrize("wire_layout", ["model_meta", "collection"])
def test_hand_authored_scaffold_task_is_rejected_before_script_runs(native_task, wire_layout):
    from nvflare.app_common.app_constant import AlgorithmConstants

    _, custom, context, policy = native_task
    adapter, _ = _adapter(custom, context, policy, script="raise AssertionError('must not train')")
    controls = {"w": np.array([0.0])}
    if wire_layout == "model_meta":
        data = DXO(
            DataKind.WEIGHTS, {"w": np.array([1.0])}, meta={AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: controls}
        ).to_shareable()
    else:
        data = DXO(
            DataKind.COLLECTION,
            {
                AppConstants.MODEL_WEIGHTS: DXO(DataKind.WEIGHTS, {"w": np.array([1.0])}),
                AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: DXO(DataKind.WEIGHT_DIFF, controls),
            },
        ).to_shareable()
    try:
        with pytest.raises(RuntimeError, match="SCAFFOLD client control variates"):
            adapter.execute("train", data, context, Signal())
    finally:
        adapter.handle_event(EventType.END_RUN, context)


def test_hand_authored_local_model_submission_is_rejected_before_script_runs(native_task):
    _, custom, context, policy = native_task
    adapter, data = _adapter(custom, context, policy, script="raise AssertionError('must not run')")
    try:
        with pytest.raises(RuntimeError, match="local-model submission"):
            adapter.execute("submit_model", data, context, Signal())
    finally:
        adapter.handle_event(EventType.END_RUN, context)


def test_worker_releases_decoded_input_after_sending_result(native_task):
    _, custom, context, policy = native_task
    script = """
import gc
import weakref
import nvflare.client as flare
flare.init()
model = flare.receive()
reference = weakref.ref(model.params["w"])
flare.send(flare.FLModel(params={"w": model.params["w"] + 1}))
gc.collect()
assert reference() is None, "worker retains decoded input after send"
"""
    adapter, data = _adapter(custom, context, policy, script=script)
    try:
        assert from_shareable(adapter.execute("train", data, context, Signal())).data["w"][0] == 2.0
    finally:
        adapter.handle_event(EventType.END_RUN, context)


pytestmark = pytest.mark.timeout(120)
