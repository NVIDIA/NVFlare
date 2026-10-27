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

import argparse
import gc
import importlib
import json
import os
import sys
import textwrap
import threading
import uuid
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
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReservedKey, ReturnCode, TaskResultReceipt
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskExecutionStatus, TaskLauncherSpec
from nvflare.apis.workspace import Workspace
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.widgets.decomposer_reg import DecomposerRegister
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.client.in_process.adapters import TaskInProcessAdapter
from nvflare.job_config.task_execution import (
    CLIENT_API_EXECUTOR_PATH,
    STATISTICS_EXECUTOR_PATH,
    prepare_task_execution,
    validate_task_resources,
)
from nvflare.private.fed.client import task_execution_policy as policy_module
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
from nvflare.private.fed.client.task_worker_executor import TASK_ATTEMPT_IDENTITY, TaskWorkerExecutor
from nvflare.private.fed.task_worker.artifacts import ArtifactReference, FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import read_bootstrap
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize


def _config(**args):
    return {
        "format_version": 3,
        "execution_lifetime": "task",
        "executors": [
            {
                "tasks": ["*"],
                "executor": {
                    "path": CLIENT_API_EXECUTOR_PATH,
                    "args": {"execution_mode": "in_process", "task_script_path": "train.py", **args},
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "mode,module_name,class_name,gpu_field",
    [
        ("k8s", "nvflare.app_opt.task_launcher.k8s_launcher", "K8sTaskLauncher", "num_of_gpus"),
        ("slurm", "nvflare.app_opt.job_launcher.slurm.task_launcher", "SlurmTaskLauncher", "gpus_per_node"),
    ],
)
def test_shared_native_selection_is_exact_lazy_and_delegates_site_validation(
    tmp_path, monkeypatch, mode, module_name, class_name, gpu_field
):
    class NativeLauncher(TaskLauncherSpec):
        launch_mode = mode

        def launch_task(self, request):
            raise NotImplementedError()

        def validate_site_config(self, site_config):
            validate(site_config)

    validate = Mock()
    launcher = NativeLauncher()
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(**{class_name: NativeLauncher}))
    monkeypatch.setattr(policy_module._SiteBuilder, "build_component", lambda *_: launcher)
    imported = []
    real_import = importlib.import_module

    def selected_import(name):
        imported.append(name)
        return real_import(name)

    monkeypatch.setattr(policy_module.importlib, "import_module", selected_import)
    config = {"launcher": {"path": module_name + "." + class_name}}
    data = {"task_execution": config, "components": [{"path": "site.CPJobLauncher"}]}
    path = tmp_path / "resources.json"
    path.write_text(json.dumps(data))
    policy = TaskExecutionPolicy.from_workspace(SimpleNamespace(get_resources_file_path=lambda: str(path)))
    validate.assert_called_once_with(data)
    assert set(imported) == {module_name}
    assert policy.worker_environment() == {}
    assert policy.task_resources({"launcher_spec": {"site": {mode: {gpu_field: 1}}}}, "site").gpu_count == 0
    with pytest.raises(ValueError, match="portable GPUs reserve"):
        policy.task_resources({"resource_spec": {"site": {"num_of_gpus": 1}}}, "site")
    with pytest.raises(ValueError, match="trusted backend"):
        TaskExecutionPolicy.from_config({**config, "environment_variables": ["APP_DATA"]})

    class Impersonator(NativeLauncher):
        pass

    monkeypatch.setattr(policy_module._SiteBuilder, "build_component", lambda *_: Impersonator())
    with pytest.raises(ValueError, match="supported"):
        TaskExecutionPolicy.from_config(config)


@pytest.mark.parametrize("mode", ["docker", "k8s", "slurm"])
@pytest.mark.parametrize("missing", ["backend", "parent", "dependency"])
def test_native_selection_reports_missing_backend_without_hiding_missing_dependency(monkeypatch, mode, missing):
    class NativeLauncher(TaskLauncherSpec):
        launch_mode = mode

        def launch_task(self, request):
            raise NotImplementedError()

    module_name, class_name = policy_module._NATIVE_TASK_LAUNCHERS[mode]
    missing_name = {
        "backend": module_name,
        "parent": module_name.rsplit(".", 1)[0],
        "dependency": "missing_backend_dependency",
    }[missing]
    monkeypatch.setattr(policy_module._SiteBuilder, "build_component", lambda *_: NativeLauncher())

    def unavailable(name):
        assert name == module_name
        raise ModuleNotFoundError(name=missing_name)

    monkeypatch.setattr(policy_module.importlib, "import_module", unavailable)
    config = {"launcher": {"path": module_name + "." + class_name}}
    expected = ModuleNotFoundError if missing == "dependency" else ValueError
    with pytest.raises(expected) as error:
        TaskExecutionPolicy.from_config(config)
    if missing == "dependency":
        assert error.value.name == missing_name
    else:
        assert "requires a supported" in str(error.value)


def test_alias_timeout_and_specs_are_inert_snapshots():
    config = _config(result_wait_timeout=12)
    spec = config["executors"][0]["executor"]
    spec["name"] = "ClientAPIExecutor"
    del spec["path"]
    plan = prepare_task_execution(config)
    assert plan.executors[0].result_wait_timeout == 12
    spec["args"]["result_wait_timeout"] = 99
    assert plan.executors[0].result_wait_timeout == 12


@pytest.mark.parametrize("mode", ["external_process", "attach"])
def test_task_script_rejects_other_modes(mode):
    with pytest.raises(ValueError, match="in_process"):
        prepare_task_execution(_config(execution_mode=mode))


@pytest.mark.parametrize("field", ["task_execution", "task_launcher", "task_state"])
def test_jobs_cannot_select_site_policy_or_state(field):
    config = _config()
    config[field] = {}
    with pytest.raises(ValueError):
        prepare_task_execution(config)


def test_application_components_and_filter_dependencies_stay_resident():
    config = _config(task_script_args=["--model", "train"])
    config["components"] = [
        {"id": "train", "path": "unused.Model", "args": {"loader_id": "loader"}},
        {"id": "loader", "path": "unused.Loader"},
        {"id": "filter_helper", "path": "unused.Helper"},
    ]
    config["task_data_filters"] = [
        {"tasks": ["train"], "filters": [{"path": "unused.Filter", "args": {"helper_id": "filter_helper"}}]}
    ]
    plan = prepare_task_execution(config)
    assert plan.executors[0].components == ()
    assert [c["id"] for c in config["components"]] == ["train", "loader", "filter_helper"]
    assert plan.executors[0].executor["args"]["task_script_args"] == ["--model", "train"]


@pytest.mark.parametrize("field", ["component_dependencies", "execution_scope"])
def test_task_profile_rejects_compute_placement_flags(field):
    config = _config()
    config["executors"][0]["executor"][field] = ["receiver"] if field == "component_dependencies" else "job"
    config["components"] = [{"id": "receiver", "path": "unused.Receiver"}]
    with pytest.raises(ValueError, match=field + ".*not supported"):
        prepare_task_execution(config)


def test_collab_rpc_service_requires_job_lifetime():
    config = _config()
    config["executors"][0]["executor"] = {"path": "nvflare.collab.runtime.executor.CollabExecutor"}
    with pytest.raises(ValueError, match="not supported.*use job lifetime"):
        prepare_task_execution(config)


@pytest.mark.parametrize(
    "path",
    [
        "nvflare.app_common.ccwf.client_ctl.ClientControllerExecutor",
        "nvflare.app_opt.flower.executors.flower_executor.FlowerExecutor",
        "nvflare.app_opt.xgboost.tree_based.executor.XGBTreeExecutor",
        "nvflare.app_common.psi.dh_psi.dh_psi_executor.DhPSIExecutor",
        "custom.UnqualifiedExecutor",
    ],
)
def test_initial_task_profile_rejects_unqualified_executors_without_importing(path):
    config = _config()
    config["executors"][0]["executor"] = {"path": path}
    with pytest.raises(ValueError, match="not supported.*use job lifetime"):
        prepare_task_execution(config)


def test_task_config_requires_format_that_older_clients_reject():
    config = _config()
    config["format_version"] = 2
    with pytest.raises(ValueError, match="older clients reject"):
        prepare_task_execution(config)


@pytest.mark.parametrize("lifetime", ["job", "task"])
def test_jobs_cannot_select_the_internal_adapter(lifetime):
    config = _config()
    config["execution_lifetime"] = lifetime
    config["executors"][0]["executor"] = {"path": "nvflare.private.fed.client.task_worker_executor.TaskWorkerExecutor"}
    with pytest.raises(ValueError, match="framework-managed"):
        prepare_task_execution(config)


def test_cpu_validation_only_considers_receiving_site():
    meta = {"resource_spec": {"site-2": {"num_of_gpus": 1}}}
    validate_task_resources(meta, "site-1")
    with pytest.raises(ValueError, match="GPUs"):
        validate_task_resources(meta, "site-2")
    meta = {"launcher_spec": {"default": {"process": {"num_of_gpus": 1}}, "site-1": {"process": {"num_of_gpus": 0}}}}
    validate_task_resources(meta, "site-1")
    with pytest.raises(ValueError, match="GPUs"):
        validate_task_resources(meta, "site-2")


@pytest.mark.parametrize(
    "name",
    [
        *JobProcessEnv.ALL,
        "CUDA_VISIBLE_DEVICES",
        "NVFLARE_CLIENT_API_CONFIG",
        "NVFLARE_CLIENT_API_BOOTSTRAP",
        "CLIENT_API_TYPE",
    ],
)
def test_site_policy_never_forwards_credential_or_gpu_environment(name):
    with pytest.raises(ValueError, match="cannot forward"):
        TaskExecutionPolicy.from_config({"environment_variables": [name]})


def test_site_environment_is_explicit_and_uses_canonical_resource_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TASK_TEST_SECRET", "local-placeholder")
    monkeypatch.setenv("UNRELATED_SECRET", "not-forwarded")
    path = tmp_path / "resources.json"
    path.write_text(json.dumps({"task_execution": {"environment_variables": ["TASK_TEST_SECRET"]}}))
    workspace = SimpleNamespace(get_resources_file_path=lambda: str(path))
    policy = TaskExecutionPolicy.from_workspace(workspace)
    assert policy.worker_environment()["TASK_TEST_SECRET"] == "local-placeholder"
    assert "UNRELATED_SECRET" not in policy.worker_environment()
    assert policy.worker_environment()["CUDA_VISIBLE_DEVICES"] == ""
    TaskExecutionPolicy.from_workspace(SimpleNamespace(get_resources_file_path=lambda: None))


def _configure(tmp_path, config, authorizer=None):
    config_file = tmp_path / "config_fed_client.json"
    config_file.write_text(json.dumps(config))
    args = argparse.Namespace(
        sp_scheme="grpc",
        sp_target="localhost:8002",
        client_name="site-1",
        parent_url=None,
        job_id="job-1",
        workspace=str(tmp_path),
    )
    workspace = SimpleNamespace(
        get_app_custom_dir=lambda _: str(tmp_path / "custom"),
        get_app_config_dir=lambda _: str(tmp_path / "config"),
        get_resources_file_path=lambda: None,
    )
    configurator = ClientJsonConfigurator(workspace, str(config_file), args, str(tmp_path))
    if authorizer:
        configurator.set_component_build_authorizer(authorizer)
    configurator.configure()
    return configurator


def test_client_config_does_not_import_or_construct_task_executor(tmp_path, monkeypatch):
    from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor

    monkeypatch.setattr(ClientAPIExecutor, "__init__", Mock(side_effect=AssertionError("must construct in worker")))
    config = _config(default_task_fetch_interval=-10)
    config["components"] = [
        {
            "id": "converter",
            "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
            "args": {"events_to_convert": ["analytix_log_stats"]},
        },
    ]
    configurator = _configure(tmp_path, config)
    assert isinstance(configurator.executors[0].executor, TaskWorkerExecutor)
    assert set(configurator.components) == {"converter"}
    assert configurator.executors[0].executor.component_specs == ()
    assert configurator.runner_config.default_task_fetch_interval == 0.5


@pytest.mark.parametrize(
    "nested",
    [
        {"helper": {"path": "never_imported.Forbidden"}},
        {"helpers": [{"path": "never_imported.Forbidden"}]},
        {"path": "/data", "helper": {"path": "never_imported.Forbidden"}},
    ],
)
def test_task_profile_rejects_nested_compute_before_application_import(tmp_path, nested):
    config = _config(**nested)
    with pytest.raises(ValueError, match="nested components are not supported"):
        _configure(tmp_path, config)
    assert "never_imported" not in sys.modules


@pytest.fixture
def native_task(tmp_path, monkeypatch, isolated_fobs):
    if not hasattr(os, "waitid"):
        pytest.skip("CPU Process task execution requires waitid/WNOWAIT")
    from nvflare.fuel.f3.cellnet.cell import Cell
    from nvflare.fuel.f3.cellnet.core_cell import CoreCell
    from nvflare.fuel.f3.stats_pool import StatsPoolManager

    nvflare_fobs_initialize()
    register_framework_decomposers("numpy", "numpy")
    root = tmp_path / "workspace"
    (root / "startup").mkdir(parents=True)
    (root / "local").mkdir()
    custom = root / "job-1" / "app_site-1" / "custom"
    custom.mkdir(parents=True)
    (custom.parent / "config").mkdir()
    (root / "job-1" / "meta.json").write_text(json.dumps({"byoc": True}))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[5]))
    monkeypatch.setenv("TASK_TEST_SECRET", "local-placeholder")
    workspace = Workspace(str(root), site_name="site-1")
    # Workers join the Client Job cell, so the executor needs a real one; it connects nowhere itself.
    # A unique name keeps it clear of cells other tests in this process may leave behind, and the
    # process-wide tables that stop() leaves alone are restored afterwards.
    original_cells, original_pools = dict(CoreCell.ALL_CELLS), dict(StatsPoolManager.pools)
    cell = Cell(f"site-1.job-{uuid.uuid4().hex}", "tcp://127.0.0.1:0", secure=False, credentials={})
    cell.start()
    try:
        context = FLContext()
        context.set_prop(
            ReservedKey.ENGINE,
            SimpleNamespace(get_workspace=lambda: workspace, get_cell=lambda: cell, fire_event=Mock()),
            private=True,
            sticky=True,
        )
        for key, value in (
            (ReservedKey.RUN_NUM, "job-1"),
            (ReservedKey.IDENTITY_NAME, "site-1"),
            (FLContextKey.WORKSPACE_OBJECT, workspace),
            (FLContextKey.TASK_ID, "task-1"),
            (FLContextKey.TASK_NAME, "train"),
            (FLContextKey.TASK_ATTEMPT_ID, "attempt-1"),
        ):
            context.set_prop(key, value, private=True, sticky=True)
        policy = TaskExecutionPolicy.from_config({"worker_timeout": 30, "environment_variables": ["TASK_TEST_SECRET"]})
        yield root, custom, context, policy
    finally:
        cell.stop()
        CoreCell.ALL_CELLS.clear(), CoreCell.ALL_CELLS.update(original_cells)
        StatsPoolManager.pools.clear(), StatsPoolManager.pools.update(original_pools)


SCRIPT = """
import os, sys, threading
import numpy as np
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.apis.analytix import AnalyticsDataType
flare.init()
while flare.is_running():
    model = flare.receive()
    flare.log("accuracy", 0.75, AnalyticsDataType.SCALAR, global_step=model.current_round)
    # Larger than the download-reference threshold: must travel inline over the session.
    flare.log("image", np.zeros((1024, 1024), dtype=np.float32), AnalyticsDataType.IMAGE, global_step=0)
    result = FLModel(params={"w": model.params["w"] + 1},
        metrics={"pid": os.getpid(), "main_thread": threading.current_thread() is threading.main_thread(),
                 "argv": sys.argv[1:], "site": flare.get_site_name(), "job": flare.get_job_id(),
                 "metadata": model.meta},
        current_round=model.current_round)
    flare.send(result)
"""


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


def _adapter(custom, context, policy, script=SCRIPT, **args):
    (custom / "train.py").write_text(script)
    spec = _config(params_exchange_format="numpy", server_expected_format="numpy", **args)["executors"][0]["executor"]
    plan = prepare_task_execution({"execution_lifetime": "task", "executors": [{"executor": spec}]})
    adapter = TaskWorkerExecutor(spec, (), policy, result_wait_timeout=plan.executors[0].result_wait_timeout)
    adapter.handle_event(EventType.START_RUN, context)
    data = DXO(DataKind.WEIGHTS, {"w": np.array([1.0, 2.0])}).to_shareable()
    data.set_header(AppConstants.CURRENT_ROUND, 3)
    return adapter, data


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
    fire_event = context.get_engine().fire_event
    seen_during_attempt = []
    fire_event.side_effect = lambda *_args: seen_during_attempt.append(adapter.supervisor.owns_execution)
    result = adapter.execute("train", data, context, Signal())
    assert seen_during_attempt == [True, True], "metrics reach the job over the cell while the attempt is still owned"
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
    identity = context.get_prop(TASK_ATTEMPT_IDENTITY)
    payload = root / "job-1" / ".nvflare" / "task-execution" / "attempts"
    files = list(payload.rglob("result.fobs"))
    assert len(files) == 1
    assert not list(payload.rglob("analytics.fobs")), "metrics are messages, never files"
    worker_fqcn = read_bootstrap(payload / identity.attempt_id / "bootstrap.json").cell_session["worker_fqcn"]
    assert worker_fqcn.startswith(context.get_engine().get_cell().get_fqcn() + ".task_worker_")
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


def test_result_deadline_pauses_only_during_framework_staging(tmp_path, monkeypatch):
    from nvflare.private.fed.client import client_api_task_runner as runner
    from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity

    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    store.create_attempt(identity)
    clock = [100.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    runner._mark_progress(store, identity, "result_wait_started.json")
    clock[0] = 102
    runner._mark_progress(store, identity, "result_staging_started.json")
    clock[0] = 200
    assert runner.result_wait_deadline(store, identity, 10, 30) is None
    runner._mark_progress(store, identity, "script_result_sent.json")
    assert runner.result_wait_deadline(store, identity, 10, 30) == 208
    clock[0] = 209
    assert runner.result_wait_deadline(store, identity, 10, 30) < clock[0]
    runner._mark_progress(store, identity, "script_finalized.json")
    assert runner.result_wait_deadline(store, identity, 10, 30) is None


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
        assert launch.call_count == 1
        second = execute(2)
        assert second.metrics["previous"] == 1
        before = checkpoint.read_bytes()
        assert execute(1).metrics == first.metrics
        assert checkpoint.read_bytes() == before
        assert execute(2).metrics == second.metrics
        assert launch.call_count == 2
        assert execute(3).metrics["previous"] == 2
        assert launch.call_count == 3
    finally:
        adapter.handle_event(EventType.END_RUN, context)


def test_gpu_worker_is_settled_before_saved_result_retry(native_task, tmp_path):
    from nvflare.apis.task_launcher_spec import TaskLaunchRequest, TaskResourceRequest
    from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher

    # Static visibility regression; CUDA execution is qualified separately.
    _, custom, context, policy = native_task
    devices = ["GPU-00000000-0000-0000-0000-000000000001"]
    launcher = ProcessTaskLauncher(gpu_devices=devices)
    policy = replace(policy, launcher=launcher)
    context.set_prop(FLContextKey.JOB_META, {"launcher_spec": {"site-1": {"process": {"num_of_gpus": 1}}}})
    adapter, data = _adapter(custom, context, policy)
    launch = launcher.launch_task = Mock(wraps=launcher.launch_task)
    try:
        first = adapter.execute("train", data, context, Signal())
        # The prior CJ result has no server receipt and is still retained.
        # A different CJ can nevertheless run on the exact same GPU now.
        second_launcher = ProcessTaskLauncher(gpu_devices=devices)
        second = second_launcher.launch_task(
            TaskLaunchRequest(
                "other-job",
                "site-1",
                "task",
                "attempt",
                (sys.executable, "-c", "pass"),
                resources=TaskResourceRequest(gpu_count=1),
            )
        )
        try:
            assert second.wait_for_settlement(10).succeeded
        finally:
            second.cancel()
        repeated = adapter.execute("train", data, context, Signal())
        np.testing.assert_array_equal(from_shareable(repeated).data["w"], from_shareable(first).data["w"])
        assert launch.call_count == 1
    finally:
        adapter.handle_event(EventType.END_RUN, context)


def test_client_api_oom_failure_clears_execution_and_accepts_a_fresh_task(native_task):
    root, custom, context, policy = native_task
    script = """
import torch
import nvflare.client as flare
flare.init()
flare.receive()
raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected without GPU allocation)")
"""
    adapter, data = _adapter(custom, context, policy, script=script)
    try:
        with pytest.raises(RuntimeError, match="OutOfMemoryError.*CUDA out of memory"):
            adapter.execute("train", data, context, Signal())
        assert adapter._done.is_set()
        assert not adapter._gate.locked()
        assert not adapter.supervisor.owns_execution
        assert not adapter.supervisor.settlement_unconfirmed
        artifacts = root / "job-1" / ".nvflare" / "task-execution" / "attempts" / "attempt-1"
        assert json.loads((artifacts / "failure.json").read_text())["error_type"] == "OutOfMemoryError"
        assert not (artifacts / "completion.json").exists()
        from nvflare.private.fed.task_worker.cell_session import _sessions

        failed_session = read_bootstrap(artifacts / "bootstrap.json").cell_session["worker_fqcn"]
        assert failed_session not in _sessions
        (custom / "train.py").write_text(SCRIPT)
        context.set_prop(FLContextKey.TASK_ID, "task-after-oom", private=True, sticky=True)
        context.set_prop(FLContextKey.TASK_ATTEMPT_ID, "attempt-after-oom", private=True, sticky=True)
        result = adapter.execute("train", data, context, Signal())
        np.testing.assert_array_equal(from_shareable(result).data["w"], [2, 3])
        healthy_artifacts = artifacts.parent / "attempt-after-oom"
        healthy_session = read_bootstrap(healthy_artifacts / "bootstrap.json").cell_session["worker_fqcn"]
        assert healthy_session != failed_session
        assert healthy_session not in _sessions
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


def test_a_receiver_failure_is_logged_as_in_job_lifetime_and_only_unfirable_metrics_are_rejected():
    """Event dispatch records receiver exceptions instead of raising, for task and job lifetime alike."""
    from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE, AnalyticsDataType
    from nvflare.apis.fl_component import FLComponent
    from nvflare.apis.utils.analytix_utils import create_analytic_dxo
    from nvflare.apis.utils.event import fire_event_to_components

    class FailingReceiver(FLComponent):
        def handle_event(self, event_type, fl_ctx):
            if event_type == ANALYTIC_EVENT_TYPE:
                raise RuntimeError("tracking backend is down")

    receiver = FailingReceiver()
    adapter = TaskWorkerExecutor({}, (), TaskExecutionPolicy.from_config({}))
    context = FLContext()
    engine = SimpleNamespace(fire_event=lambda event, ctx: fire_event_to_components(event, [receiver], ctx))
    context.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    metric = create_analytic_dxo(tag="loss", value=0.5, data_type=AnalyticsDataType.SCALAR).to_shareable()

    adapter._fire_analytics(context, metric)
    assert isinstance(context.get_prop(FLContextKey.EXCEPTIONS)[receiver.name], RuntimeError)
    with pytest.raises(ValueError, match="DXO Shareable"):
        adapter._fire_analytics(context, {"not": "a shareable"})


def test_receipts_ignore_other_identity_and_retain_filtered_failure():
    from nvflare.apis.fl_constant import ReturnCode
    from nvflare.apis.shareable import make_reply
    from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity

    adapter = TaskWorkerExecutor({"path": "unused.Executor"}, (), TaskExecutionPolicy.from_config({}))
    adapter.supervisor = Mock()
    context = FLContext()
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    for key, value in (
        (ReservedKey.RUN_NUM, "job"),
        (ReservedKey.IDENTITY_NAME, "site"),
        (FLContextKey.TASK_ID, "task"),
        (FLContextKey.TASK_NAME, "evaluate"),
        (FLContextKey.TASK_ATTEMPT_ID, "attempt"),
        (TASK_ATTEMPT_IDENTITY, identity),
        (FLContextKey.TASK_RESULT, make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR)),
        (FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED),
    ):
        context.set_prop(key, value, private=True, sticky=False)
    adapter._record_receipt(context)
    adapter.supervisor.record_receipt.assert_not_called()
    context.set_prop(FLContextKey.TASK_NAME, "train", private=True, sticky=False)
    adapter._record_receipt(context)
    adapter.supervisor.retain_attempt.assert_called_once_with(identity)
    adapter.supervisor.record_receipt.assert_called_once_with(identity, TaskResultReceipt.RECEIVED)


@pytest.mark.parametrize("write_failure", [None, "result", "analytics"])
def test_client_api_worker_runs_directly_without_event_dispatch(native_task, monkeypatch, request, write_failure):
    import nvflare.client.api as client_api
    from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
    from nvflare.private.fed.client.client_api_task_runner import bind_client_task_context
    from nvflare.private.fed.task_worker import (
        FileTaskArtifactStore,
        TaskAttemptIdentity,
        WorkerBootstrap,
        cell_session,
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
    # The worker runs in this process and joins the fixture's Client Job cell.
    records = []
    on_analytics = records.append
    if write_failure == "analytics":
        on_analytics = Mock(side_effect=ValueError("the Client Job rejected the metric"))
    cj_cell = context.get_engine().get_cell()
    cell_session.listen(cj_cell, "localhost")
    session = cell_session.CellWorkerSession(cj_cell, on_analytics)
    request.addfinalizer(session.close)
    spec = _config()["executors"][0]["executor"]
    write_bootstrap(
        path, WorkerBootstrap(identity, store.root_dir, str(root), spec, cell_session=session.bootstrap_fields())
    )
    if write_failure is not None:
        error, match = OSError, "cannot persist task output"
        if write_failure == "result":
            monkeypatch.setattr(FileTaskArtifactStore, "stage_result", Mock(side_effect=OSError(match)))
        else:
            error, match = RuntimeError, "the Client Job rejected the metric"
        with pytest.raises(error, match=match):
            worker.run_worker(path, context_binding=bind_client_task_context)
        assert not Path(store.attempt_dir(identity), "completion.json").exists()
        assert store.read_failure(identity)["error_type"] == error.__name__
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
        assert reference.kind == "input", "the worker must not reload its saved result"
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
    assert "analytics" not in completion.diagnostics, "metrics are messages, never files"
    scalar, image = (from_shareable(record).data["track_value"] for record in records)
    assert scalar == 0.75
    assert isinstance(image, np.ndarray) and image.shape == (1024, 1024) and not image.any()
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


def test_task_runner_finalizes_api_even_if_script_cleanup_fails():
    from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
    from nvflare.private.fed.client.client_api_task_runner import _ClientAPITaskRunner

    executor = ClientAPIExecutor(execution_mode="in_process", task_script_path="train.py")
    runner = _ClientAPITaskRunner(executor)
    script = Mock(release_runtime=Mock(side_effect=ValueError("script cleanup failure")))
    api = Mock(close=Mock(side_effect=OSError("API cleanup failure")))
    runner._runner, runner._api = script, api
    with pytest.raises(ValueError, match="script cleanup failure"):
        runner.finalize(FLContext())
    api.close.assert_called_once()
    # Cleanup consumes ownership even if a resource reports failure.
    runner.finalize(FLContext())
    script.release_runtime.assert_called_once()
    api.close.assert_called_once()


def test_script_arguments_do_not_relocate_resident_services(tmp_path):
    config = _config(task_script_args=["--metric_writer", "client_receiver"])
    config["components"] = [
        {
            "id": "client_receiver",
            "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
            "args": {"events_to_convert": ["analytix_log_stats"]},
        }
    ]
    configurator = _configure(tmp_path, config)
    assert set(configurator.components) == {"client_receiver"}
    adapter = configurator.executors[0].executor
    assert adapter.component_specs == ()
    assert adapter.executor_spec["args"]["task_script_args"] == ["--metric_writer", "client_receiver"]


@pytest.mark.parametrize("kind", ["fifo", "symlink"])
def test_worker_log_replay_rejects_special_files_without_blocking(tmp_path, kind):
    from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity

    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    path = tmp_path / "task_attempt_log.txt"
    if kind == "fifo":
        os.mkfifo(path)
    else:
        target = tmp_path / "actual.txt"
        target.write_text("must not replay\n")
        path.symlink_to(target)
    adapter = TaskWorkerExecutor({}, (), TaskExecutionPolicy.from_config({}))
    adapter.log_warning = Mock()
    adapter.log_info = Mock()
    workspace = SimpleNamespace(get_log_root=lambda _: str(tmp_path))
    with ThreadPoolExecutor() as pool:
        replay = pool.submit(adapter._replay_worker_log, workspace, identity, FLContext())
        try:
            replay.result(timeout=1)
        except TimeoutError:
            # Release the old blocking reader before reporting the regression.
            with path.open("w") as writer:
                writer.write("release reader\n")
            replay.result(timeout=5)
            pytest.fail("worker log replay blocked on a FIFO")
    adapter.log_warning.assert_called_once()
    adapter.log_info.assert_not_called()


def test_worker_log_replay_has_a_finite_size_and_line_bound(tmp_path):
    from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity

    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    (tmp_path / "task_attempt_log.txt").write_text("log line\n" * 4097 + "x" * (1024 * 1024))
    adapter = TaskWorkerExecutor({}, (), TaskExecutionPolicy.from_config({}))
    adapter.log_warning = Mock()
    adapter.log_info = Mock()
    adapter._replay_worker_log(SimpleNamespace(get_log_root=lambda _: str(tmp_path)), identity, FLContext())
    assert adapter.log_info.call_count <= 4096
    adapter.log_warning.assert_called_once()


def test_execute_releases_owner_even_if_log_replay_fails(tmp_path, monkeypatch):
    from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity
    from nvflare.private.fed.task_worker.supervisor import TaskSupervisorOutcome

    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    adapter = TaskWorkerExecutor({}, (), TaskExecutionPolicy.from_config({}))
    adapter._started = True
    adapter._identity = Mock(return_value=identity)
    result = Shareable({"done": True})
    adapter.supervisor = SimpleNamespace(
        run=Mock(return_value=TaskSupervisorOutcome(identity, result=result)),
        owns_execution=False,
        settlement_unconfirmed=False,
        take_issues=lambda: [],
    )
    run_dir = tmp_path / "job"
    run_dir.mkdir()
    workspace = SimpleNamespace(get_run_dir=lambda _: str(run_dir), get_root_dir=lambda: str(tmp_path))
    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=False)
    monkeypatch.setattr(adapter, "_replay_worker_log", Mock(side_effect=RuntimeError("logging failed")))
    with pytest.raises(RuntimeError, match="logging failed"):
        adapter.execute("train", Shareable(), context, Signal())
    assert adapter._done.is_set()
    assert adapter._gate.acquire(blocking=False)
    adapter._gate.release()


def test_gpu_named_application_options_are_not_resource_requests():
    validate_task_resources({"resource_spec": {"site-1": {"gpu_cache_enabled": True}}}, "site-1")


@pytest.mark.parametrize(
    "mode,gpu_field",
    [("process", "num_of_gpus"), ("docker", "num_of_gpus"), ("k8s", "num_of_gpus"), ("slurm", "gpus_per_node")],
)
def test_cpu_validation_resolves_launcher_defaults_and_zero_site_overrides(mode, gpu_field):
    meta = {"launcher_spec": {"default": {mode: {gpu_field: 1}}, "site-1": {mode: {gpu_field: 0}}}}
    validate_task_resources(meta, "site-1")
    with pytest.raises(ValueError, match="GPUs"):
        validate_task_resources(meta, "site-2")
    # GPU allocations unrelated to the receiving site must not reject it.
    validate_task_resources({"launcher_spec": {"server": {mode: {gpu_field: 1}}}}, "site-1")
    validate_task_resources({"resource_spec": {"site-2": {mode: {gpu_field: 1}}}}, "site-1")


def test_cpu_validation_does_not_drop_nested_gpu_modes_with_portable_defaults():
    meta = {
        "resource_spec": {
            "@default": {"num_of_gpus": 1},
            "site-1": {"process": {"num_of_gpus": 0}, "slurm": {"gpus_per_node": 1}},
        }
    }
    with pytest.raises(ValueError, match="GPUs"):
        validate_task_resources(meta, "site-1")


def test_cpu_validation_falls_back_to_nested_resources_for_missing_launcher_modes():
    meta = {
        "launcher_spec": {"site-1": {"process": {"num_of_gpus": 0}}},
        "resource_spec": {"site-1": {"docker": {"num_of_gpus": 1}}},
    }
    with pytest.raises(ValueError, match="GPUs"):
        validate_task_resources(meta, "site-1")


def _statistics_config():
    config = _config()
    config["executors"][0]["executor"] = {"path": STATISTICS_EXECUTOR_PATH, "args": {"generator_id": "stats_generator"}}
    config["components"] = [
        {"id": "stats_generator", "path": "stats_probe.ProbeStatistics"},
        {
            "id": "min_count",
            "path": "nvflare.app_common.statistics.min_count_cleanser.MinCountCleanser",
            "args": {"min_count": 5},
        },
    ]
    config["task_result_filters"] = [
        {
            "tasks": ["fed_stats"],
            "filters": [
                {
                    "path": "nvflare.app_common.filters.statistics_privacy_filter.StatisticsPrivacyFilter",
                    "args": {"result_cleanser_ids": ["min_count"]},
                }
            ],
        }
    ]
    return config


def test_statistics_generator_runs_only_in_worker_while_privacy_filter_stays_resident(tmp_path):
    config = _statistics_config()
    authorized = []

    def authorize(config_dict, config_ctx, node):
        authorized.append(config_dict.get("path"))
        return ""

    # stats_probe does not exist in this process: CJ must not import it.
    configurator = _configure(tmp_path, config, authorize)
    assert STATISTICS_EXECUTOR_PATH in authorized
    assert "stats_probe.ProbeStatistics" in authorized
    assert set(configurator.components) == {"min_count"}
    assert len(configurator.task_result_filter_chains) == 1
    adapter = configurator.executors[0].executor
    assert isinstance(adapter, TaskWorkerExecutor)
    assert [c["id"] for c in adapter.component_specs] == ["stats_generator"]
    config["components"][0]["path"] = "changed.AfterPlanning"
    assert adapter.component_specs[0]["path"] == "stats_probe.ProbeStatistics"


@pytest.mark.parametrize("generator_id", [None, 1, "missing"])
def test_statistics_requires_explicit_configured_generator(generator_id):
    config = _statistics_config()
    config["executors"][0]["executor"]["args"]["generator_id"] = generator_id
    with pytest.raises(ValueError, match="generator_id"):
        prepare_task_execution(config)


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


def test_wrapped_client_api_settings_preserve_mode_and_deadline(tmp_path):
    config = _config(result_wait_timeout=12)
    executor = config["executors"][0]["executor"]
    executor["args"] = {"class_attributes": executor["args"]}
    configurator = _configure(tmp_path, config)
    assert configurator.executors[0].executor.result_wait_timeout == 12


@pytest.mark.parametrize("attributes", [None, [], "invalid"])
def test_wrapped_constructor_attributes_require_a_dict(attributes):
    config = _config()
    config["executors"][0]["executor"]["args"] = {"class_attributes": attributes}
    with pytest.raises(ValueError, match="class_attributes.*dict"):
        prepare_task_execution(config)


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


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_task_client_registers_framework_codecs_before_receiving_tensors(tmp_path, isolated_fobs, wrapped, alias):
    pytest.importorskip("torch")
    from nvflare.app_opt.pt.decomposers import TensorDecomposer
    from nvflare.fuel.utils import fobs
    from nvflare.fuel.utils.fobs import dots

    assert fobs.get_dot_handler(dots.TENSOR_DOWNLOAD) is None
    config = _config(params_exchange_format="pytorch", server_expected_format="pytorch")
    spec = config["executors"][0]["executor"]
    if wrapped:
        spec["args"] = {"class_attributes": spec["args"]}
    if alias:
        spec["name"] = "ClientAPIExecutor"
        del spec["path"]
    _configure(tmp_path, config)
    assert isinstance(fobs.get_dot_handler(dots.TENSOR_DOWNLOAD), TensorDecomposer)


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


def test_task_adapter_scopes_pass_through_to_get_task_and_releases_its_route():
    from nvflare.apis.fl_constant import ServerCommandNames
    from nvflare.private.defs import CellChannel

    route = (CellChannel.SERVER_COMMAND, ServerCommandNames.GET_TASK)
    unrelated = ("aux", "tensor-message")
    # The site's internal settings, as a provisioned kit writes them, name the CP's own listener.
    site_internal = {"host": "cp-1", "port": 8102, "listen_host": "0", "connection_security": "mtls"}
    manager = SimpleNamespace(int_scheme="stcp", int_resources=dict(site_internal))
    cell = SimpleNamespace(
        decode_pass_through_topics={unrelated},
        make_internal_listener=Mock(),
        register_request_cb=Mock(),
        core_cell=SimpleNamespace(connector_manager=manager),
    )
    context = FLContext()
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: cell))
    adapter = TaskWorkerExecutor({}, (), TaskExecutionPolicy.from_config({}))
    adapter.handle_event(EventType.START_RUN, context)
    assert cell.decode_pass_through_topics == {route, unrelated}
    # The site's scheme and security, the Process launcher's host, and no fixed port: never the CP's address.
    cell.make_internal_listener.assert_called_once_with(
        scheme="stcp", resources={"host": "localhost", "connection_security": "mtls"}
    )
    assert manager.int_resources == site_internal, "the site settings are left alone"
    assert cell.register_request_cb.call_count == 2
    adapter.handle_event(EventType.END_RUN, context)
    assert cell.decode_pass_through_topics == {unrelated}


def test_task_adapter_requires_a_cell_before_starting():
    context = FLContext()
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: None))
    adapter = TaskWorkerExecutor({}, (), TaskExecutionPolicy.from_config({}))
    with pytest.raises(RuntimeError, match="task lifetime requires a Cell"):
        adapter.handle_event(EventType.START_RUN, context)
    assert not adapter._started


def test_disk_resolution_cleans_downloaded_batches_if_a_later_batch_fails(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    from nvflare.app_opt.pt.decomposers import TensorDecomposer
    from nvflare.app_opt.pt.lazy_tensor_dict import LazyTensorDict
    from nvflare.fuel.utils import fobs
    from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef

    class FakeCell:
        def get_fobs_context(self, props=None):
            return {fobs.FOBSContextKey.CELL: self, **(props or {})}

    fobs.register(TensorDecomposer)
    scratch = tmp_path / "offload"
    scratch.mkdir()
    file = scratch / "chunk.safetensors"
    save_file({"T0": torch.ones(4)}, str(file))
    owner = LazyTensorDict({"T0": (str(file), "T0")}, str(scratch))
    calls = []

    def download(**kwargs):
        calls.append(kwargs["ref_id"])
        if len(calls) == 2:
            raise RuntimeError("later batch aborted")
        return None, owner

    monkeypatch.setattr("nvflare.app_opt.pt.decomposers.download_tensors_to_disk", download)
    value = [LazyDownloadRef("server", batch, "T0", dot=TensorDecomposer().get_download_dot()) for batch in ("a", "b")]
    context = FLContext()
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=FakeCell))
    context.set_prop(ReservedKey.RUN_NUM, "job")
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, SimpleNamespace(get_run_dir=lambda _: str(tmp_path / "job")))
    adapter = TaskWorkerExecutor({}, (), TaskExecutionPolicy.from_config({}))
    with pytest.raises(RuntimeError, match="later batch aborted"):
        adapter.resolve_task_data(value, context, abort_signal=Signal())
    assert calls == ["a", "b"]
    assert not scratch.exists()
