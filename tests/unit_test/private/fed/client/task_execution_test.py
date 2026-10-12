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
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, TaskResultReceipt
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_common.app_constant import AppConstants
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.job_config.task_execution import CLIENT_API_EXECUTOR_PATH, prepare_task_execution, validate_task_resources
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
from nvflare.private.fed.client.task_worker_executor import TASK_ATTEMPT_IDENTITY, TaskWorkerExecutor
from nvflare.private.fed.task_worker.artifacts import IncompleteTaskArtifactError
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize


def _config(**args):
    return {
        "format_version": 2,
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


def test_compute_dependencies_use_argument_values_and_preserve_dicts():
    config = _config()
    config["components"] = [
        {"id": "model", "path": "unused.Model", "args": {"data_loader_id": "loader"}},
        {"id": "loader", "path": "unused.Loader"},
        {"id": "receiver", "path": "unused.Receiver"},
    ]
    config["executors"][0]["executor"]["args"]["source_model"] = "model"
    ordinary = {"config_type": "dict", "path": "data", "options": {"name": "model"}}
    config["executors"][0]["executor"]["args"]["options"] = ordinary
    plan = prepare_task_execution(config)
    assert [c["id"] for c in plan.executors[0].components] == ["model", "loader"]
    assert [c["id"] for c in plan.job_components] == ["receiver"]
    assert plan.executors[0].executor["args"]["options"] == ordinary
    assert config["executors"][0]["executor"]["args"]["options"] == ordinary
    config["components"][1]["args"] = {"mutated": True}
    assert "args" not in plan.executors[0].components[1]


def test_each_executor_gets_its_own_dependency_graph():
    config = _config()
    config["components"] = [
        {"id": "train_model", "path": "unused.Model", "component_dependencies": ["loader"]},
        {"id": "eval_model", "path": "unused.Model", "args": {"data_loader_id": "loader"}},
        {"id": "loader", "path": "unused.Loader"},
        {"id": "receiver", "path": "unused.Receiver"},
    ]
    config["executors"] = [
        {"tasks": ["train"], "executor": {"path": "unused.Trainer", "args": {"model_id": "train_model"}}},
        {"tasks": ["eval"], "executor": {"path": "unused.Validator", "args": {"model_id": "eval_model"}}},
    ]
    plan = prepare_task_execution(config)
    assert [c["id"] for c in plan.executors[0].components] == ["train_model", "loader"]
    assert [c["id"] for c in plan.executors[1].components] == ["eval_model", "loader"]
    assert [c["id"] for c in plan.job_components] == ["receiver"]


@pytest.mark.parametrize("consumer", ["task_data_filters", "task_result_filters", "receiver"])
def test_compute_dependencies_cannot_be_shared_with_cj_services(consumer):
    config = _config(source_model="model")
    config["components"] = [{"id": "model", "path": "unused.Model"}]
    service = {"path": "unused.Service", "args": {"helper_id": "model"}}
    if consumer == "receiver":
        config["components"].append({"id": "receiver", **service})
    else:
        config[consumer] = [{"tasks": ["train"], "filters": [service]}]
    with pytest.raises(ValueError, match="shared by compute and resident CJ services.*model"):
        prepare_task_execution(config)


def test_cj_filter_dependencies_stay_resident_and_task_names_are_not_references():
    config = _config(source_model="train")
    config["components"] = [
        {"id": "train", "path": "unused.Model"},
        {"id": "filter_helper", "path": "unused.Helper"},
    ]
    config["task_data_filters"] = [
        {"tasks": ["train"], "filters": [{"path": "unused.Filter", "args": {"helper_id": "filter_helper"}}]}
    ]
    plan = prepare_task_execution(config)
    assert [c["id"] for c in plan.executors[0].components] == ["train"]
    assert [c["id"] for c in plan.job_components] == ["filter_helper"]


def test_nested_scope_is_rejected():
    config = _config()
    config["executors"][0]["executor"]["args"]["helper"] = {
        "path": "unused.Helper",
        "execution_scope": "job",
    }
    with pytest.raises(ValueError, match="execution_scope is not supported"):
        prepare_task_execution(config)


def test_collab_rpc_service_requires_job_lifetime():
    config = _config()
    config["executors"][0]["executor"] = {"path": "nvflare.collab.runtime.executor.CollabExecutor"}
    with pytest.raises(ValueError, match="resident federation RPC"):
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
    meta = {"launcher_spec": {"default": {"gpu_config": {"gpu_ids": [0]}}, "site-1": {"gpu_config": {"gpu_ids": []}}}}
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


def test_client_config_does_not_import_or_construct_task_graph(tmp_path):
    config = _config()
    config["executors"][0]["executor"] = {"path": "never_imported.TaskExecutor", "args": {"helper_id": "worker"}}
    config["components"] = [
        {"id": "worker", "path": "never_imported.Helper", "args": {"default_task_fetch_interval": -10}},
        {
            "id": "converter",
            "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
            "args": {"events_to_convert": ["analytix_log_stats"]},
        },
    ]
    configurator = _configure(tmp_path, config)
    assert "never_imported" not in sys.modules
    assert isinstance(configurator.executors[0].executor, TaskWorkerExecutor)
    assert set(configurator.components) == {"converter"}
    assert len(configurator.executors[0].executor.component_specs) == 1
    assert configurator.runner_config.default_task_fetch_interval == 0.5


def test_authorization_rejects_nested_compute_before_application_import(tmp_path):
    config = _config()
    config["executors"][0]["executor"] = {
        "path": "never_imported.TaskExecutor",
        "args": {"path": "/data", "helper": {"path": "never_imported.Forbidden"}},
    }
    seen = []

    def authorizer(spec, context, node):
        seen.append((spec["path"], node.path()))
        return "blocked helper" if spec["path"].endswith(".Forbidden") else None

    with pytest.raises(Exception, match="blocked helper"):
        _configure(tmp_path, config, authorizer)
    assert seen == [
        ("never_imported.TaskExecutor", "executors.#1.executor"),
        ("never_imported.Forbidden", "executors.#1.executor.args.helper"),
    ]
    assert "never_imported" not in sys.modules


@pytest.fixture
def native_task(tmp_path, monkeypatch):
    if not hasattr(os, "waitid"):
        pytest.skip("CPU Process task execution requires waitid/WNOWAIT")
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
    context = FLContext()
    context.set_prop(
        ReservedKey.ENGINE,
        SimpleNamespace(get_workspace=lambda: workspace, fire_event=Mock()),
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
    return root, custom, context, policy


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
    result = FLModel(params={"w": model.params["w"] + 1},
        metrics={"pid": os.getpid(), "main_thread": threading.current_thread() is threading.main_thread(),
                 "argv": sys.argv[1:], "site": flare.get_site_name(), "job": flare.get_job_id()},
        current_round=model.current_round)
    flare.send(result)
"""


def test_referenced_compute_dependency_is_recreated_for_each_assignment(native_task):
    _, custom, context, policy = native_task
    (custom / "compute_probe.py").write_text(
        """
import uuid
from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.shareable import Shareable

class DataLoader(FLComponent):
    def __init__(self):
        super().__init__()
        self.instance = str(uuid.uuid4())
        self.calls = 0

    def load(self):
        self.calls += 1
        return self.calls

class Trainer(Executor):
    def __init__(self, data_loader_id):
        super().__init__()
        self.data_loader_id = data_loader_id

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        loader = fl_ctx.get_engine().get_component(self.data_loader_id)
        return Shareable({"calls": loader.load(), "instance": loader.instance})
"""
    )
    config = _config()
    config["executors"][0]["executor"] = {
        "path": "compute_probe.Trainer",
        "args": {"data_loader_id": "loader"},
    }
    config["components"] = [{"id": "loader", "path": "compute_probe.DataLoader"}]
    plan = prepare_task_execution(config).executors[0]
    adapter = TaskWorkerExecutor(plan.executor, plan.components, policy)
    adapter.handle_event(EventType.START_RUN, context)
    results = []
    try:
        for attempt in (1, 2):
            context.set_prop(FLContextKey.TASK_ID, f"task-{attempt}", private=True, sticky=True)
            context.set_prop(FLContextKey.TASK_ATTEMPT_ID, f"attempt-{attempt}", private=True, sticky=True)
            results.append(adapter.execute("train", Shareable(), context, Signal()))
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
    assert context.get_engine().fire_event.call_count > 0
    identity = context.get_prop(TASK_ATTEMPT_IDENTITY)
    payload = root / "job-1" / ".nvflare" / "task-execution" / "attempts"
    files = list(payload.rglob("result.fobs"))
    assert len(files) == 1
    context.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
    context.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED, private=True, sticky=False)
    with context.get_prop(FLContextKey.TASK_RESULT_SOURCE_GUARD)():
        adapter.handle_event(EventType.AFTER_SEND_TASK_RESULT, context)
        adapter.handle_event(EventType.END_RUN, context)
        assert files[0].exists()
    assert not files[0].exists()
    assert identity.attempt_id == "attempt-1"


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
    with pytest.raises(IncompleteTaskArtifactError):
        adapter.execute("train", data, context, Signal())
    adapter.handle_event(EventType.END_RUN, context)
    assert not list((root / "job-1" / ".nvflare").rglob("completion.json"))
    assert list((root / "job-1" / ".nvflare").rglob("failure.json"))


def test_result_wait_timeout_cancels_the_real_worker(native_task):
    root, custom, context, policy = native_task
    script = "import time\nimport nvflare.client as flare\nflare.init()\nflare.receive()\ntime.sleep(10)\n"
    adapter, data = _adapter(custom, context, policy, script=script, result_wait_timeout=0.1)
    with pytest.raises(TimeoutError, match="timed out"):
        adapter.execute("train", data, context, Signal())
    adapter.handle_event(EventType.END_RUN, context)
    assert not adapter.supervisor.settlement_unconfirmed
    assert not list((root / "job-1" / ".nvflare").rglob("completion.json"))


def test_timely_local_send_does_not_timeout_during_script_finalization(native_task):
    _, custom, context, policy = native_task
    adapter, data = _adapter(
        custom, context, policy, script=SCRIPT + "\nimport time\ntime.sleep(0.4)\n", result_wait_timeout=0.2
    )
    result = adapter.execute("train", data, context, Signal())
    np.testing.assert_array_equal(from_shareable(result).data["w"], [2, 3])
    adapter.handle_event(EventType.END_RUN, context)


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
