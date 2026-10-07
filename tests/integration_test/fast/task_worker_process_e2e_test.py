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

"""Exercise the existing NumPy FedAvg fixture through real simulator processes.

These CPU Process integration checks do not qualify scheduler/GPU admission or
the production launch backends. The only application configuration difference
between job-based and task runs is the execution lifetime setting.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests/integration_test/data/jobs/hello-numpy-sag"

_TRACE_FILTER_SOURCE = """
import json
import os
import time
from pathlib import Path

from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import FLContextKey


class TraceFilter(Filter):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase

    def process(self, shareable, fl_ctx):
        workspace = fl_ctx.get_engine().get_workspace()
        path = Path(workspace.get_run_dir(fl_ctx.get_job_id())) / "filter-trace.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps({
                "phase": self.phase,
                "pid": os.getpid(),
                "task_id": fl_ctx.get_prop(FLContextKey.TASK_ID),
                "timestamp": time.time(),
            }) + "\\n")
        return shareable
"""

_STATE_TRACE_SOURCE = """
import json
import os
import time
from pathlib import Path

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.task_state import get_task_state
from nvflare.app_common.np.np_trainer import NPTrainer


def trace(fl_ctx, **fields):
    workspace = fl_ctx.get_engine().get_workspace()
    path = Path(workspace.get_run_dir(fl_ctx.get_job_id())) / "state-trace.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({
            "pid": os.getpid(),
            "task_id": fl_ctx.get_prop(FLContextKey.TASK_ID),
            "attempt_id": fl_ctx.get_prop(FLContextKey.TASK_ATTEMPT_ID),
            "timestamp": time.time(),
            **fields,
        }) + "\\n")


class StatefulNPTrainer(NPTrainer):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        state = get_task_state(fl_ctx)
        previous = state.get("steps", {"count": 0})["count"]
        state["steps"] = {"count": previous + 1}
        trace(fl_ctx, phase="steps", previous_steps=previous, steps=previous + 1)
        # Only the declared state/trace is added: model training is unchanged.
        return super().execute(task_name, shareable, fl_ctx, abort_signal)


class ScopeTrace(FLComponent):
    def __init__(self, scope):
        super().__init__()
        self.scope = scope

    def handle_event(self, event_type, fl_ctx):
        if event_type in (EventType.START_RUN, EventType.END_RUN):
            trace(fl_ctx, phase="component", scope=self.scope, event=event_type)
"""


def _run_simulator(job_dir, workspace, output_path):
    env = os.environ.copy()
    env.pop("NVFLARE_HOME", None)
    env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), env.get("PYTHONPATH", "")))
    command = [
        sys.executable,
        "-m",
        "nvflare.private.fed.app.simulator.simulator",
        str(job_dir),
        "-w",
        str(workspace),
        "-n",
        "2",
        "-t",
        "2",
    ]
    with output_path.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=150)
        finally:
            if process.poll() is None:
                # Only clean up this test's still-owned process tree. psutil
                # Process instances track creation time to guard PID reuse.
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                for child in reversed(children):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                psutil.wait_procs(children, timeout=5)
                process.wait(timeout=5)
    assert returncode == 0, output_path.read_text()[-20000:]


def _assert_task_worker_records(workspace):
    publications = []
    for site_name in ("site-1", "site-2"):
        paths = list((workspace / site_name).rglob("diagnostics.jsonl"))
        assert len(paths) == 1
        records = [json.loads(line) for line in paths[0].read_text().splitlines()]
        site_publications = [record for record in records if record["event"] == "publication"]
        assert len(site_publications) == 3
        assert len({record["supervisor_pid"] for record in site_publications}) == 1
        for record in site_publications:
            assert record["site_name"] == site_name
            assert record["publication_outcome"] == "accepted"
            assert record["worker_pid"] != record["supervisor_pid"]
            assert record["worker_ppid"] == record["supervisor_pid"]
            assert (
                record["launch_timestamp"]
                <= record["worker_started_timestamp"]
                <= record["worker_completed_timestamp"]
                <= record["settled_timestamp"]
                <= record["publication_timestamp"]
            )
            attempt_dir = paths[0].parent / "attempts" / record["attempt_id"]
            assert (attempt_dir / "completion.json").is_file()
            assert not (attempt_dir / "input.fobs").exists()
            assert not (attempt_dir / "result.fobs").exists()
        publications.extend(site_publications)
    assert len({record["attempt_id"] for record in publications}) == 6
    assert len({record["worker_pid"] for record in publications}) == 6
    return publications


def _configure_site_process_launcher(workspace):
    local_dir = workspace / "local"
    local_dir.mkdir(parents=True)
    (local_dir / "resources.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "task_launcher": {
                    "path": "nvflare.app_common.task_launcher.process_launcher.ProcessTaskLauncher",
                    "args": {"poll_interval": 0.01},
                },
            }
        )
    )


@pytest.mark.skipif(os.name != "posix", reason="The initial ProcessTaskLauncher requires POSIX process groups")
@pytest.mark.timeout(180)
@pytest.mark.parametrize("execution_lifetime", ["job", "task"])
def test_existing_numpy_fedavg_across_three_rounds(tmp_path, execution_lifetime):
    job_dir = tmp_path / "hello-numpy-sag"
    shutil.copytree(FIXTURE, job_dir)
    client_path = job_dir / "app/config/config_fed_client.json"
    original_client = json.loads(client_path.read_text())
    client = dict(original_client)
    if execution_lifetime == "task":
        client["execution_lifetime"] = "task"
        client_path.write_text(json.dumps(client, indent=2))
    assert {k: v for k, v in client.items() if k != "execution_lifetime"} == original_client
    assert (job_dir / "app/config/config_fed_server.json").read_bytes() == (
        FIXTURE / "app/config/config_fed_server.json"
    ).read_bytes()

    workspace = tmp_path / "workspace"
    output_path = tmp_path / "simulator.log"
    _run_simulator(job_dir, workspace, output_path)

    models = list((workspace / "server").rglob("server.npy"))
    assert len(models) == 1, f"Expected one server result under {workspace}; see {output_path}"
    # The shipped fixture starts from 1..9; unchanged NPTrainer adds one per
    # round, and equal averaging preserves that increment across both clients.
    np.testing.assert_array_equal(np.load(models[0], allow_pickle=False), np.arange(1, 10).reshape(3, 3) + 3)
    for site_name in ("site-1", "site-2"):
        site_models = list((workspace / site_name).rglob("best_numpy.npy"))
        assert len(site_models) == 1, f"Missing final model for {site_name}; see {output_path}"
        np.testing.assert_array_equal(np.load(site_models[0], allow_pickle=False), np.arange(1, 10).reshape(3, 3) + 3)
    if execution_lifetime == "task":
        _assert_task_worker_records(workspace)


@pytest.mark.skipif(os.name != "posix", reason="The initial ProcessTaskLauncher requires POSIX process groups")
@pytest.mark.timeout(180)
def test_public_job_setting_exports_a_runnable_task_job(tmp_path):
    from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
    from nvflare.app_common.np.np_model_persistor import NPModelPersistor
    from nvflare.app_common.np.np_trainer import NPTrainer
    from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
    from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
    from nvflare.job_config.api import FedJob

    job = FedJob(name="numpy-task-worker", min_clients=2, execution_lifetime="task")
    job.to_server(NPModelPersistor(), id="persistor")
    job.to_server(FullModelShareableGenerator(), id="shareable_generator")
    job.to_server(InTimeAccumulateWeightedAggregator(expected_data_kind="WEIGHTS"), id="aggregator")
    job.to_server(
        ScatterAndGather(
            min_clients=2,
            num_rounds=3,
            persistor_id="persistor",
            aggregator_id="aggregator",
            shareable_generator_id="shareable_generator",
            train_timeout=60,
        )
    )
    job.to_clients(NPTrainer(), tasks=["train"])
    export_root = tmp_path / "export"
    job.export_job(str(export_root))

    workspace = tmp_path / "workspace"
    _run_simulator(export_root / job.name, workspace, tmp_path / "simulator.log")
    models = list((workspace / "server").rglob("server.npy"))
    assert len(models) == 1
    np.testing.assert_array_equal(np.load(models[0], allow_pickle=False), np.arange(1, 10).reshape(3, 3) + 3)
    _assert_task_worker_records(workspace)


@pytest.mark.skipif(os.name != "posix", reason="The initial ProcessTaskLauncher requires POSIX process groups")
@pytest.mark.timeout(180)
def test_filters_stay_in_cj_and_result_filter_follows_worker_settlement(tmp_path):
    job_dir = tmp_path / "hello-numpy-with-filters"
    shutil.copytree(FIXTURE, job_dir)
    custom_dir = job_dir / "app/custom"
    custom_dir.mkdir(exist_ok=True)
    (custom_dir / "trace_filter.py").write_text(_TRACE_FILTER_SOURCE)
    meta_path = job_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["byoc"] = True  # This test intentionally supplies a custom filter.
    meta_path.write_text(json.dumps(meta))
    client_path = job_dir / "app/config/config_fed_client.json"
    client = json.loads(client_path.read_text())
    client["execution_lifetime"] = "task"
    for key, phase in (("task_data_filters", "input"), ("task_result_filters", "result")):
        client[key] = [
            {
                "tasks": ["train"],
                "filters": [{"path": "trace_filter.TraceFilter", "args": {"phase": phase}}],
            }
        ]
    client_path.write_text(json.dumps(client))

    workspace = tmp_path / "workspace"
    _configure_site_process_launcher(workspace)
    _run_simulator(job_dir, workspace, tmp_path / "simulator.log")
    publications = _assert_task_worker_records(workspace)
    for site_name in ("site-1", "site-2"):
        paths = list((workspace / site_name).rglob("filter-trace.jsonl"))
        assert len(paths) == 1
        events = [json.loads(line) for line in paths[0].read_text().splitlines()]
        assert len(events) == 6
        for record in (r for r in publications if r["site_name"] == site_name):
            task_events = [event for event in events if event["task_id"] == record["task_id"]]
            assert [event["phase"] for event in task_events] == ["input", "result"]
            assert all(event["pid"] == record["supervisor_pid"] for event in task_events)
            assert task_events[0]["timestamp"] <= record["launch_timestamp"]
            assert record["settled_timestamp"] <= task_events[1]["timestamp"] <= record["publication_timestamp"]


@pytest.mark.skipif(os.name != "posix", reason="The initial ProcessTaskLauncher requires POSIX process groups")
@pytest.mark.timeout(180)
def test_declared_state_survives_three_cold_starts_and_explicit_component_scope(tmp_path):
    from nvflare.apis.event_type import EventType

    job_dir = tmp_path / "hello-numpy-with-state"
    shutil.copytree(FIXTURE, job_dir)
    custom_dir = job_dir / "app/custom"
    custom_dir.mkdir(exist_ok=True)
    (custom_dir / "state_trace.py").write_text(_STATE_TRACE_SOURCE)
    meta_path = job_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["byoc"] = True  # This test adds an ordinary Executor wrapper and event probes.
    meta_path.write_text(json.dumps(meta))
    client_path = job_dir / "app/config/config_fed_client.json"
    client = json.loads(client_path.read_text())
    client["execution_lifetime"] = "task"
    client["task_state"] = {"names": ["steps"]}
    client["executors"][0]["executor"]["path"] = "state_trace.StatefulNPTrainer"
    client["components"] = [
        {"id": "worker_trace", "path": "state_trace.ScopeTrace", "args": {"scope": "task"}},
        {
            "id": "job_trace",
            "path": "state_trace.ScopeTrace",
            "args": {"scope": "job"},
            "execution_scope": "job",
        },
    ]
    # Neither probe is referenced by an executor/filter: the default worker
    # graph and explicit job placement must not depend on ID heuristics.
    assert "execution_scope" not in client["components"][0]
    client_path.write_text(json.dumps(client))
    assert (job_dir / "app/config/config_fed_server.json").read_bytes() == (
        FIXTURE / "app/config/config_fed_server.json"
    ).read_bytes()

    workspace = tmp_path / "workspace"
    _configure_site_process_launcher(workspace)
    _run_simulator(job_dir, workspace, tmp_path / "simulator.log")
    publications = _assert_task_worker_records(workspace)
    expected_model = np.arange(1, 10).reshape(3, 3) + 3
    models = list((workspace / "server").rglob("server.npy"))
    assert len(models) == 1
    np.testing.assert_array_equal(np.load(models[0], allow_pickle=False), expected_model)
    for site_name in ("site-1", "site-2"):
        site_root = workspace / site_name
        site_models = list(site_root.rglob("best_numpy.npy"))
        assert len(site_models) == 1
        np.testing.assert_array_equal(np.load(site_models[0], allow_pickle=False), expected_model)
        trace_paths = list(site_root.rglob("state-trace.jsonl"))
        assert len(trace_paths) == 1
        traces = [json.loads(line) for line in trace_paths[0].read_text().splitlines()]
        steps = [trace for trace in traces if trace["phase"] == "steps"]
        assert [trace["previous_steps"] for trace in steps] == [0, 1, 2]
        assert [trace["steps"] for trace in steps] == [1, 2, 3]
        site_publications = [record for record in publications if record["site_name"] == site_name]
        assert {trace["pid"] for trace in steps} == {record["worker_pid"] for record in site_publications}
        runtime_root = next(site_root.rglob("diagnostics.jsonl")).parent
        previous_publication = None
        for step, record in zip(steps, site_publications):
            assert (step["task_id"], step["attempt_id"]) == (record["task_id"], record["attempt_id"])
            if previous_publication is not None:
                # The next cold start consumes the prior exact-ACK promotion.
                assert previous_publication <= step["timestamp"]
            previous_publication = record["publication_timestamp"]
            attempt_dir = runtime_root / "attempts" / record["attempt_id"]
            completion = json.loads((attempt_dir / "completion.json").read_text())
            assert completion["identity"] == {
                key: record[key] for key in ("job_id", "site_name", "task_id", "task_name", "attempt_id")
            }
            assert completion["state_revision"] == step["steps"] - 1
            assert completion["state"]["kind"] == "state"
            assert completion["state"]["file_name"] == "state.fobs"
            assert not (attempt_dir / "state.fobs").exists()
        # Job state survives independently of cleaned transient input/result/
        # candidate-state payloads; immutable completion/diagnostics survive.
        checkpoint = json.loads((runtime_root / "state/current.json").read_text())
        assert checkpoint["revision"] == 3
        assert checkpoint["names"] == ["steps"]
        assert checkpoint["records"] == {"steps": {"encoding": "json", "value": {"count": 3}}}
        assert checkpoint["identity"] == completion["identity"]
        assert checkpoint["result_sha256"] == completion["result"]["sha256"]
        assert checkpoint["state_sha256"] == completion["state"]["sha256"]
        job_events = [trace for trace in traces if trace.get("scope") == "job"]
        assert [trace["event"] for trace in job_events] == [EventType.START_RUN, EventType.END_RUN]
        assert all(trace["pid"] == site_publications[0]["supervisor_pid"] for trace in job_events)
        worker_events = [trace for trace in traces if trace.get("scope") == "task"]
        assert len(worker_events) == 6
        for step in steps:
            events = [trace for trace in worker_events if trace["attempt_id"] == step["attempt_id"]]
            assert [trace["event"] for trace in events] == [EventType.START_RUN, EventType.END_RUN]
            assert all(trace["pid"] == step["pid"] for trace in events)
