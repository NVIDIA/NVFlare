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


import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, TaskResultReceipt
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.job_config.task_execution import STATISTICS_EXECUTOR_PATH, prepare_task_execution, validate_task_resources
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
from nvflare.private.fed.client.task_worker_executor import TASK_ATTEMPT_IDENTITY, TaskWorkerExecutor
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from tests.task_execution_test_utils import _config, _configure, _statistics_config
from tests.timing_utils import isolate_time


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


def test_task_adapter_scopes_pass_through_to_get_task_and_releases_its_route():
    from nvflare.apis.fl_constant import ServerCommandNames
    from nvflare.private.defs import CellChannel

    route = (CellChannel.SERVER_COMMAND, ServerCommandNames.GET_TASK)
    unrelated = ("aux", "tensor-message")
    cell = SimpleNamespace(decode_pass_through_topics={unrelated})
    context = FLContext()
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: cell))
    adapter = TaskWorkerExecutor({}, (), TaskExecutionPolicy.from_config({}))
    adapter.handle_event(EventType.START_RUN, context)
    assert cell.decode_pass_through_topics == {route, unrelated}
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


@pytest.fixture(autouse=True)
def _isolate_module_clock(monkeypatch):
    isolate_time(monkeypatch, "nvflare.private.fed.client.client_api_task_runner")
