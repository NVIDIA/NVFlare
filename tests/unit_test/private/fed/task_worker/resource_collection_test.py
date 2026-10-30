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
from unittest.mock import Mock

import pytest

from nvflare.apis.executor import Executor
from nvflare.apis.shareable import Shareable
from nvflare.private.fed.task_worker import WorkerBootstrap, worker, write_bootstrap
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore, IncompleteTaskArtifactError
from tests.task_worker_test_utils import _identity, _workspace

_SNAPSHOT = {
    "duration_seconds": "10",
    "resource_capacity": {
        "cpu": {"groups": [{"units": "2"}]},
        "memory": {"bytes": "1024"},
        "gpu": {"groups": []},
    },
    "cpu_consumed": {"cpu_seconds": "4"},
}


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
            assert not incomplete
            return {"sent_bytes": "7", "received_bytes": "11"}

    monkeypatch.setattr(worker, "_build_executor", lambda _config: Trainer())
    monkeypatch.setattr(worker, "ExecutionResourceCollector", Collection)
    monkeypatch.setattr(worker, "CellWorkerTransport", Transport)


def test_worker_shared_snapshot_precedes_commit_and_traffic_follows_close(tmp_path, monkeypatch):
    phases = []
    _controlled_worker(monkeypatch, phases)
    store, identity, path = _stage(tmp_path, cell=True)
    checkpoint = FileTaskArtifactStore.stage_checkpoint
    commit = FileTaskArtifactStore.commit_staged_result
    write = FileTaskArtifactStore.write_resource_stats

    def checkpoint_stage(self, expected):
        phases.append("checkpoint")
        return checkpoint(self, expected)

    def commit_result(self, *args, **kwargs):
        phases.append("commit")
        return commit(self, *args, **kwargs)

    def write_report(self, *args, **kwargs):
        phases.append("sidecar")
        return write(self, *args, **kwargs)

    monkeypatch.setattr(FileTaskArtifactStore, "stage_checkpoint", checkpoint_stage)
    monkeypatch.setattr(FileTaskArtifactStore, "commit_staged_result", commit_result)
    monkeypatch.setattr(FileTaskArtifactStore, "write_resource_stats", write_report)
    completion = worker.run_worker(path)
    assert phases == [
        "start",
        "open",
        "execute",
        "finalize",
        "checkpoint",
        "stop",
        "commit",
        "close",
        "traffic",
        "sidecar",
    ]
    assert completion.diagnostics == {"elapsed_seconds": 10.0, "cpu_seconds": 4.0}
    report = store.read_resource_stats(identity)
    assert report["duration_seconds"] == "10"
    assert report["cpu_consumed"] == {"cpu_seconds": "4"}
    assert report["message_traffic"] == {"sent_bytes": "7", "received_bytes": "11"}
    assert store.read_result(identity)[0]["value"] == 2


def test_python_failure_still_reports_work_without_committing_result(tmp_path, monkeypatch):
    phases = []
    _controlled_worker(monkeypatch, phases, fail=True)
    store, identity, path = _stage(tmp_path, cell=True)
    with pytest.raises(ValueError, match="training failure"):
        worker.run_worker(path)
    assert phases == ["start", "open", "execute", "finalize", "stop", "close", "traffic"]
    assert store.read_resource_stats(identity)["cpu_consumed"] == {"cpu_seconds": "4"}
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_result(identity)
    assert Path(store.attempt_dir(identity), "failure.json").exists()


def test_sidecar_failure_cannot_replace_successful_training_result(tmp_path, monkeypatch):
    phases = []
    _controlled_worker(monkeypatch, phases)
    store, identity, path = _stage(tmp_path)

    def fail_stats(*_args):
        raise OSError("statistics write failed")

    monkeypatch.setattr(FileTaskArtifactStore, "write_resource_stats", fail_stats)
    worker.run_worker(path)
    assert store.read_result(identity)[0]["value"] == 2
    assert store.read_resource_stats(identity) is None


def test_unregistered_worker_does_not_start_collection(tmp_path, monkeypatch):
    phases = []
    _controlled_worker(monkeypatch, phases)
    store, identity, path = _stage(tmp_path, reporting=False)
    collection = Mock(side_effect=AssertionError("unexpected measurement"))
    monkeypatch.setattr(worker, "ExecutionResourceCollector", collection)
    worker.run_worker(path)
    collection.assert_not_called()
    assert store.read_resource_stats(identity) is None


@pytest.mark.parametrize("snapshot", [None, {}, {"resource_capacity": {}, "cpu_consumed": {"cpu_seconds": "bad"}}])
def test_bad_optional_snapshot_cannot_fail_training(tmp_path, monkeypatch, snapshot):
    phases = []
    _controlled_worker(monkeypatch, phases)
    store, identity, path = _stage(tmp_path)
    monkeypatch.setattr(worker, "ExecutionResourceCollector", lambda: Mock(stop=lambda: snapshot))
    worker.run_worker(path)
    assert store.read_result(identity)[0]["value"] == 2
    report = store.read_resource_stats(identity)
    assert report["cpu_consumed"]["status"] == "unavailable"
    assert report["resource_capacity"]["status"] == "unavailable"
