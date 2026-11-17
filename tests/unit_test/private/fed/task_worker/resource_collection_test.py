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

"""Worker resource hooks preserve result publication and failure authority."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.executor import Executor
from nvflare.apis.shareable import Shareable
from nvflare.private.fed.resource_stats import collector, cpu_consumed
from nvflare.private.fed.resource_stats.collector import ExecutionResourceCollector
from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant
from nvflare.private.fed.resource_stats.probes import gpu_nvidia
from nvflare.private.fed.resource_stats.task_rollup import TaskResourceReducer
from nvflare.private.fed.task_worker import WorkerBootstrap, worker, write_bootstrap
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore, IncompleteTaskArtifactError
from tests.task_worker_test_utils import _identity, _workspace

_SNAPSHOT = {
    "duration_seconds": 10,
    "resource_capacity": {"cpu": {"units": 2}, "memory": {"bytes": 1024}, "gpu": {"count": 0}},
    "cpu_consumed": {"cpu_seconds": 4},
    "issues": [],
}


def _settle_resources(record, identity):
    reducer = TaskResourceReducer()
    assert reducer.register(identity)
    reducer.settle(identity, record, started=True, settled=True)
    return reducer.snapshot()


def _stage(tmp_path, *, reporting=True, cell=False):
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity("resources")
    store.create_attempt(identity)
    store.write_input(identity, Shareable({"value": 1}))
    bootstrap = WorkerBootstrap(
        identity=identity,
        artifact_root=store.root_dir,
        workspace_root=str(_workspace(tmp_path)),
        executor={"path": "nvflare.app_common.np.np_trainer.NPTrainer"},
        resource_reporting=reporting,
        cell_session={"connect_url": "tcp://localhost:8000", "worker_fqcn": "site-1.job-1.worker"} if cell else None,
    )
    path = store.bootstrap_path(identity)
    write_bootstrap(path, bootstrap)
    return store, identity, path


def _controlled_worker(monkeypatch, phases, *, fail=False):
    class Trainer(Executor):
        def execute(self, *_args):
            phases.append("execute")
            if fail:
                raise ValueError("training failure")
            return Shareable({"value": 2})

        def finalize(self, _ctx):
            phases.append("finalize")

    class Collection:
        def __init__(self):
            phases.append("start")

        def stop(self):
            phases.append("stop")
            return deepcopy(_SNAPSHOT)

    class Transport:
        def __init__(self, _config):
            pass

        def open(self):
            phases.append("open")

        def close(self):
            phases.append("close")

        def traffic_snapshot(self, *, incomplete=False):
            phases.append("traffic")
            assert incomplete
            return {
                "sent_bytes": 7,
                "received_bytes": 11,
                "status": "partial",
                "issues": ["observation_incomplete"],
            }

    monkeypatch.setattr(worker, "_build_executor", lambda _config: Trainer())
    monkeypatch.setattr(worker, "ExecutionResourceCollector", Collection)
    monkeypatch.setattr(worker, "CellWorkerTransport", Transport)


def test_worker_snapshots_use_completion_diagnostics_without_extra_files(tmp_path, monkeypatch):
    phases = []
    _controlled_worker(monkeypatch, phases)
    store, identity, path = _stage(tmp_path, cell=True)
    monkeypatch.setattr(
        worker, "time", SimpleNamespace(time=worker.time.time, monotonic=Mock(side_effect=[20.0, 25.5]))
    )
    monkeypatch.setattr(
        worker,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_utime=0.125, ru_stime=0.25, ru_maxrss=64)
        ),
    )
    checkpoint = FileTaskArtifactStore.stage_checkpoint
    commit = FileTaskArtifactStore.commit_staged_result

    def checkpoint_stage(self, expected):
        phases.append("checkpoint")
        return checkpoint(self, expected)

    def commit_result(self, *args, **kwargs):
        phases.append("commit")
        return commit(self, *args, **kwargs)

    monkeypatch.setattr(FileTaskArtifactStore, "stage_checkpoint", checkpoint_stage)
    monkeypatch.setattr(FileTaskArtifactStore, "commit_staged_result", commit_result)
    completion = worker.run_worker(path)
    assert phases == [
        "start",
        "open",
        "execute",
        "finalize",
        "checkpoint",
        "stop",
        "traffic",
        "commit",
        "close",
    ]
    report = completion.to_dict()["diagnostics"]["resource_stats"]
    assert set(report) == {"duration_seconds", "resource_capacity", "cpu_consumed", "message_traffic", "issues"}
    assert {key: value for key, value in completion.diagnostics.items() if key != "resource_stats"} == {
        "elapsed_seconds": 5.5,
        "user_cpu_seconds": 0.125,
        "system_cpu_seconds": 0.25,
        "max_rss_native_units": 64,
        "max_rss_unit": "bytes" if worker.sys.platform == "darwin" else "kibibytes",
    }
    assert report["duration_seconds"] == 10
    assert report["cpu_consumed"] == {"cpu_seconds": 4}
    assert report["message_traffic"] == {"sent_bytes": 7, "received_bytes": 11, "status": "partial"}
    assert report["issues"] == [{"metric": "message_traffic", "code": "observation_incomplete"}]
    assert store.read_result(identity)[0]["value"] == 2
    assert not Path(store.attempt_dir(identity), "resource_stats.json").exists()


def test_python_failure_still_reports_work_without_committing_result(tmp_path, monkeypatch):
    phases = []
    _controlled_worker(monkeypatch, phases, fail=True)
    store, identity, path = _stage(tmp_path, cell=True)
    with pytest.raises(ValueError, match="training failure"):
        worker.run_worker(path)
    assert phases == ["start", "open", "execute", "finalize", "stop", "traffic", "close"]
    assert store.read_failure(identity)["diagnostics"]["resource_stats"]["cpu_consumed"] == {"cpu_seconds": 4}
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_result(identity)
    assert Path(store.attempt_dir(identity), "failure.json").exists()


def _nested_value():
    value = 0
    for _ in range(62):
        value = {"nested": value}
    return value


@pytest.mark.parametrize(
    "bad",
    [object(), float("nan"), "x" * (64 * 1024), _nested_value()],
    ids=["non-json", "non-finite", "oversized", "too-deep"],
)
@pytest.mark.parametrize("fail", [False, True])
def test_unserializable_or_oversized_statistics_do_not_replace_application_outcome(tmp_path, monkeypatch, bad, fail):
    phases = []
    _controlled_worker(monkeypatch, phases, fail=fail)
    store, identity, path = _stage(tmp_path)
    monkeypatch.setattr(worker, "ExecutionResourceCollector", lambda: Mock(stop=lambda: {**_SNAPSHOT, "bad": bad}))
    if fail:
        with pytest.raises(ValueError, match="training failure"):
            worker.run_worker(path)
        assert "diagnostics" not in store.read_failure(identity)
    else:
        completion = worker.run_worker(path)
        assert store.read_result(identity)[0]["value"] == 2
        assert "resource_stats" not in completion.diagnostics


def test_unregistered_worker_does_not_start_collection(tmp_path, monkeypatch):
    phases = []
    _controlled_worker(monkeypatch, phases)
    store, identity, path = _stage(tmp_path, reporting=False)
    collection = Mock(side_effect=AssertionError("unexpected measurement"))
    monkeypatch.setattr(worker, "ExecutionResourceCollector", collection)
    completion = worker.run_worker(path)
    collection.assert_not_called()
    assert "resource_stats" not in completion.diagnostics


@pytest.mark.parametrize("snapshot", [None, {}, {"resource_capacity": {}, "cpu_consumed": {"cpu_seconds": "bad"}}])
def test_bad_optional_snapshot_cannot_fail_training(tmp_path, monkeypatch, snapshot):
    phases = []
    _controlled_worker(monkeypatch, phases)
    store, identity, path = _stage(tmp_path)
    monkeypatch.setattr(worker, "ExecutionResourceCollector", lambda **_kwargs: Mock(stop=lambda: snapshot))
    completion = worker.run_worker(path)
    assert store.read_result(identity)[0]["value"] == 2
    totals = _settle_resources(completion.to_dict()["diagnostics"].get("resource_stats"), identity)
    assert totals["cpu_consumed"]["status"] == "unavailable"
    assert totals["resource_time"]["status"] == "unavailable"


@pytest.mark.parametrize("fail", [False, True], ids=["completed", "python-failure"])
@pytest.mark.parametrize("gpu", [None, {"count": 2}])
def test_task_worker_freezes_startup_gpu_observation_on_success_or_failure(tmp_path, monkeypatch, fail, gpu):
    phases = []
    _controlled_worker(monkeypatch, phases, fail=fail)
    store, identity, path = _stage(tmp_path)
    events = []
    monkeypatch.setattr(cpu_consumed, "_active", None)
    monkeypatch.setattr(collector.cgroup_linux, "discover_cgroups", lambda: ([], []))
    monkeypatch.setattr(collector.cgroup_linux, "probe_cpu", lambda *_args, **_kwargs: {"units": 2})
    monkeypatch.setattr(collector.cgroup_linux, "probe_memory", lambda *_args, **_kwargs: {"bytes": 1024})

    def probe_gpu():
        events.append("gpu")
        return gpu

    gpu_query = Mock(side_effect=probe_gpu)
    monkeypatch.setattr(gpu_nvidia, "probe_gpu", gpu_query)
    usages = iter(((float(0), float(0)), (float(1), float(0))))

    def usage():
        events.append("cpu")
        return next(usages)

    monkeypatch.setattr(
        collector, "CpuConsumedAccountant", lambda **kw: CpuConsumedAccountant(usage_reader=usage, **kw)
    )
    ticks = iter((0, 1_000_000_000))

    def clock():
        events.append("clock")
        return next(ticks)

    monkeypatch.setattr(
        worker,
        "ExecutionResourceCollector",
        lambda: ExecutionResourceCollector(capacity_probe=collector.probe_capacity, clock_ns=clock),
    )
    if fail:
        with pytest.raises(ValueError, match="training failure"):
            worker.run_worker(path)
        with pytest.raises(IncompleteTaskArtifactError):
            store.read_result(identity)
        report = store.read_failure(identity)["diagnostics"]["resource_stats"]
    else:
        completion = worker.run_worker(path)
        assert store.read_result(identity)[0]["value"] == 2
        report = completion.to_dict()["diagnostics"]["resource_stats"]
    gpu_query.assert_called_once_with()
    assert events == ["gpu", "clock", "cpu", "cpu", "clock"]
    assert report["cpu_consumed"] == {"cpu_seconds": 1}
    assert report["duration_seconds"] == 1
    assert report["resource_capacity"]["cpu"] == {"units": 2}
    assert report["resource_capacity"]["memory"] == {"bytes": 1024}
    if gpu is None:
        assert "gpu" not in report["resource_capacity"]
        assert report["resource_capacity"]["status"] == "partial"
    else:
        assert report["resource_capacity"]["gpu"] == gpu
