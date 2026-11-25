# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, MachineStatus, RunnerTask, WorkspaceConstants
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.signal import Signal
from nvflare.apis.simulation_plan import SIMULATION_PLAN_PROP
from nvflare.apis.utils.event import fire_event_to_components
from nvflare.fuel.utils import gpu_utils
from nvflare.private.fed.app.simulator import simulator_runner as simulator_runner_module
from nvflare.private.fed.app.simulator import simulator_worker
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorClientRunner, SimulatorRunner
from nvflare.private.fed.client.client_runner import ClientRunner
from nvflare.private.fed.utils.fed_utils import split_gpus
from tests.timing_utils import CheckedThread, ThreadGroup, isolate_time, join_thread, wait_for


class MockCell:
    def get_root_url_for_child(self):
        return "tcp://0:123"

    def get_internal_listener_url(self):
        return "tcp://0:123"


class _BoundedClientPool(list):
    """Make a selector regression fail promptly instead of spinning under the pool lock."""

    probes = 0

    def __getitem__(self, index):
        self.probes += 1
        assert self.probes <= 100, "selector spun on a fully reserved pool"
        return super().__getitem__(index)


@pytest.fixture
def client_scheduler(tmp_path, monkeypatch):
    args = Namespace(workspace=str(tmp_path), log_config=None, set=[])
    clients = _BoundedClientPool(
        SimpleNamespace(client_name=f"site-{i}", simulate_running=False, token="token", cell=Mock())
        for i in range(1, 4)
    )
    for client in clients:
        for directory in ("startup", "local"):
            (tmp_path / client.client_name / directory).mkdir(parents=True)
    runner = SimulatorClientRunner(str(tmp_path / "custom"), args, clients, {}, args, {})
    select = runner.get_next_run_client

    def bounded_selection(gpu):
        clients.probes = 0
        return select(gpu)

    monkeypatch.setattr(runner, "get_next_run_client", bounded_selection)
    clock = isolate_time(monkeypatch, simulator_runner_module)
    clock.sleep = Mock()
    return runner, clients, threading.Lock()


@pytest.fixture
def client_task_boundary(client_scheduler, monkeypatch):
    runner, clients, lock = client_scheduler
    process, conn = Mock(), Mock()
    process.poll.return_value = None
    monkeypatch.setattr(simulator_runner_module, "get_open_ports", Mock(return_value=[12345]))
    monkeypatch.setattr(simulator_runner_module, "popen_in_new_session", Mock(return_value=process))
    monkeypatch.setattr(runner, "_create_connection", Mock(return_value=conn))
    with lock:
        client = runner.get_next_run_client(None)
    return runner, clients, lock, client, process, conn


def test_client_selection_reserves_before_another_thread_can_select(client_scheduler):
    runner, clients, lock = client_scheduler
    clients[0].simulate_running = clients[1].simulate_running = True
    selected, release = threading.Event(), threading.Event()
    choices = []

    def select_then_pause():
        with lock:
            choices.append(runner.get_next_run_client(None))
        selected.set()
        release.wait()

    with ThreadGroup(release) as workers:
        workers.start(select_then_pause)
        assert selected.wait(5), "first scheduler did not select a client"
        with lock:
            assert choices[0] is clients[2]
            assert clients[2].simulate_running
            assert runner.get_next_run_client(None) is None


def test_client_selection_is_round_robin_and_can_reuse_a_released_client(client_scheduler):
    runner, clients, lock = client_scheduler
    with lock:
        assert [runner.get_next_run_client(None) for _ in clients] == clients
        assert runner.get_next_run_client(None) is None
        clients[1].simulate_running = False
        assert runner.get_next_run_client(None) is clients[1]
        assert all(client.simulate_running for client in clients)


def test_scheduler_selects_replacement_only_after_worker_cleanup(client_task_boundary, monkeypatch):
    runner, clients, lock, client, process, conn = client_task_boundary
    client.simulate_running = False
    runner.run_client_index = -1
    conn.recv.side_effect = [False, True, "site-2", False]
    cleaning, release = threading.Event(), threading.Event()
    cleanups = []

    def cleanup(_process):
        cleanups.append(_process)
        if len(cleanups) == 1:
            cleaning.set()
            release.wait()

    monkeypatch.setattr(runner, "_cleanup_process", cleanup)
    with ThreadGroup(release) as workers:
        workers.start(lambda: runner.run_client_thread(2, None, lock, False))
        assert cleaning.wait(5), "worker did not reach cleanup"
        with lock:
            assert client.simulate_running
            assert not any(c.simulate_running for c in clients[1:])
    assert len(cleanups) == 2
    assert runner.clients_finished_end_run == {"site-2"}
    assert not runner.fatal_clients
    assert not any(c.simulate_running for c in clients)


def test_parallel_workers_receive_independent_startup_identities(client_scheduler, monkeypatch):
    runner, clients, lock = client_scheduler
    runner.build_ctx = {"client_name": "template"}
    process = Mock()
    process.poll.return_value = None
    connections = [Mock(), Mock()]
    for client, conn in zip(clients, connections):
        conn.recv.side_effect = [True, client.client_name, False]
    sending, second_sent, release = threading.Event(), threading.Event(), threading.Event()
    identities = {}

    def send_first(data):
        if isinstance(data, dict):
            sending.set()
            release.wait()
            identities["site-1"] = data[simulator_runner_module.SimulatorConstants.BUILD_CTX]["client_name"]

    def send_second(data):
        if isinstance(data, dict):
            identities["site-2"] = data[simulator_runner_module.SimulatorConstants.BUILD_CTX]["client_name"]
            second_sent.set()

    connections[0].send.side_effect = send_first
    connections[1].send.side_effect = send_second
    monkeypatch.setattr(simulator_runner_module, "get_open_ports", Mock(side_effect=[[10001], [10002]]))
    monkeypatch.setattr(simulator_runner_module, "popen_in_new_session", Mock(return_value=process))
    monkeypatch.setattr(runner, "_create_connection", lambda port, **kwargs: connections[port - 10001])
    monkeypatch.setattr(runner, "_cleanup_process", Mock())
    with lock:
        first, second = runner.get_next_run_client(None), runner.get_next_run_client(None)
    with ThreadGroup(release) as workers:
        workers.start(lambda: runner.do_one_task(first, 3, None, lock))
        assert sending.wait(5), "first worker did not reach its startup send"
        workers.start(lambda: runner.do_one_task(second, 3, None, lock))
        assert second_sent.wait(5), "second worker did not send its startup identity"
    assert identities == {"site-1": "site-1", "site-2": "site-2"}
    assert runner.build_ctx == {"client_name": "template"}


def test_selection_excludes_finished_and_failed_clients(client_scheduler):
    runner, clients, lock = client_scheduler
    runner.clients_finished_end_run.add("site-1")
    runner.fatal_clients.add("site-2")
    with lock:
        assert runner.get_next_run_client(None) is clients[2]
        assert runner.get_next_run_client(None) is None
    assert not clients[0].simulate_running
    assert not clients[1].simulate_running


def test_overlapping_schedulers_finish_each_client_once(client_scheduler, monkeypatch):
    runner, clients, lock = client_scheduler
    first_started, second_started, third_started = (threading.Event() for _ in range(3))
    release_first, release_second = threading.Event(), threading.Event()
    tasks = []

    def task(client, *args, task_name, **kwargs):
        tasks.append((client.client_name, task_name))
        if client is clients[0]:
            first_started.set()
            release_first.wait()
        elif client is clients[1]:
            second_started.set()
            release_second.wait()
        else:
            third_started.set()
        return True, client.client_name

    def between_tasks(_interval):
        if third_started.is_set():
            release_second.wait()
            wait_for(lambda: "site-2" in runner.clients_finished_end_run)

    monkeypatch.setattr(runner, "do_one_task", task)
    simulator_runner_module.time.sleep.side_effect = between_tasks
    with ThreadGroup(release_first, release_second, timeout=15) as workers:
        workers.start(lambda: runner.run_client_thread(2, None, lock, True))
        assert first_started.wait(5), "first scheduler did not start"
        workers.start(lambda: runner.run_client_thread(2, None, lock, True))
        assert second_started.wait(5), "second scheduler did not start"
        release_first.set()
        assert third_started.wait(5), "remaining client did not receive END_RUN"
        wait_for(lambda: runner.clients_finished_end_run == {"site-1", "site-3"})
    assert tasks == [("site-1", RunnerTask.TASK_EXEC), ("site-2", RunnerTask.TASK_EXEC), ("site-3", RunnerTask.END_RUN)]
    assert runner.clients_finished_end_run == {c.client_name for c in clients}
    assert not runner.fatal_clients
    assert not any(c.simulate_running for c in clients)


@pytest.mark.parametrize("failure", ["connection", "initial-send", "worker-death"])
def test_startup_failure_is_failed_not_completed_and_other_clients_end(client_task_boundary, failure):
    runner, clients, lock, client, process, conn = client_task_boundary
    client.simulate_running = False
    runner.run_client_index = -1
    remaining = [Mock(), Mock()]
    for candidate, connection in zip(clients[1:], remaining):
        connection.recv.side_effect = [True, candidate.client_name, False]
    if failure == "connection":
        runner._create_connection.side_effect = [RuntimeError("handshake failed"), *remaining]
    else:
        runner._create_connection.side_effect = [conn, *remaining]
        if failure == "initial-send":
            conn.send.side_effect = BrokenPipeError("startup send failed")
        else:
            # First health check observes death; cleanup and later workers use the normal fake boundary.
            process.poll.side_effect = [1, 1] + [None] * 10
    runner.run_client_thread(2, None, lock, True)
    assert runner.clients_finished_end_run == {"site-2", "site-3"}
    assert runner.fatal_clients == {"site-1"}
    conn.recv.assert_not_called()
    assert not any(c.simulate_running for c in clients)


@pytest.mark.parametrize("failure", ["exception", "reported"])
def test_local_failure_does_not_send_end_run_to_healthy_clients(client_scheduler, monkeypatch, failure):
    runner, clients, lock = client_scheduler
    failing_started, healthy_started, third_started = (threading.Event() for _ in range(3))
    release_failure, release_healthy = threading.Event(), threading.Event()
    tasks = []

    def task(client, *args, task_name, **kwargs):
        tasks.append((client.client_name, task_name))
        if client is clients[0]:
            failing_started.set()
            release_failure.wait()
            if failure == "exception":
                raise RuntimeError("connection setup failed")
            return True, None
        if client is clients[1]:
            healthy_started.set()
        else:
            third_started.set()
        release_healthy.wait()
        return True, client.client_name

    def between_tasks(_interval):
        if third_started.is_set():
            release_healthy.wait()
            wait_for(lambda: "site-2" in runner.clients_finished_end_run)

    monkeypatch.setattr(runner, "do_one_task", task)
    simulator_runner_module.time.sleep.side_effect = between_tasks
    with ThreadGroup(release_failure, release_healthy, timeout=15) as workers:
        workers.start(lambda: runner.run_client_thread(2, None, lock, True))
        assert failing_started.wait(5), "failing worker did not start"
        workers.start(lambda: runner.run_client_thread(2, None, lock, True))
        assert healthy_started.wait(5), "healthy worker did not start"
        release_failure.set()
        assert third_started.wait(5), "idle healthy client was not scheduled"
        assert tasks[-1] == ("site-3", RunnerTask.TASK_EXEC)
        assert not runner.clients_finished_end_run
        assert clients[1].simulate_running
    assert runner.fatal_clients == {"site-1"}
    assert runner.clients_finished_end_run == {"site-2", "site-3"}


@pytest.mark.parametrize("final_reply", [EOFError("truncated outcome"), True])
def test_partial_or_fatal_worker_reply_is_not_completed(client_task_boundary, final_reply):
    runner, clients, lock, client, process, conn = client_task_boundary
    client.simulate_running = False
    runner.run_client_index = -1
    conn.recv.side_effect = [True, "site-1", final_reply]

    runner.run_client_thread(2, None, lock, False)

    assert not runner.clients_finished_end_run
    assert runner.fatal_clients == {"site-1"}
    assert not client.simulate_running


@pytest.mark.parametrize("end_run_for_all", [False, True])
def test_scheduler_releases_and_records_failed_task(client_scheduler, monkeypatch, end_run_for_all):
    runner, clients, lock = client_scheduler
    monkeypatch.setattr(runner, "do_one_task", Mock(side_effect=RuntimeError("worker failed")))
    runner.run_client_thread(2, None, lock, end_run_for_all)
    assert runner.fatal_clients == ({c.client_name for c in clients} if end_run_for_all else {"site-1"})
    assert not runner.clients_finished_end_run
    assert not any(c.simulate_running for c in clients)


def test_busy_pool_keeps_current_worker_without_spinning(client_task_boundary):
    runner, clients, lock, client, process, conn = client_task_boundary
    for candidate in clients:
        candidate.simulate_running = True
    conn.recv.side_effect = [False, True, client.client_name, False]
    assert runner.do_one_task(client, 2, None, lock) == (True, client.client_name)
    assert [call.args[0] for call in conn.send.call_args_list[1:]] == [True, False]


def test_scheduler_waits_for_a_client_to_be_released(client_scheduler, monkeypatch):
    runner, clients, lock = client_scheduler
    for client in clients:
        client.simulate_running = True

    def release_after_retry(_interval):
        if simulator_runner_module.time.sleep.call_count == 2:
            with lock:
                clients[0].simulate_running = False

    simulator_runner_module.time.sleep.side_effect = release_after_retry
    task = Mock(return_value=(True, "site-1"))
    monkeypatch.setattr(runner, "do_one_task", task)
    runner.run_client_thread(2, None, lock, False)
    assert simulator_runner_module.time.sleep.call_count == 2
    assert task.call_args.args[0] is clients[0]
    assert runner.clients_finished_end_run == {"site-1"}
    assert not clients[0].simulate_running


@pytest.mark.parametrize("exit_before_kill", [False, True])
def test_worker_cleanup_reaps_after_kill_and_tolerates_exit_race(client_scheduler, exit_before_kill):
    runner, _, _ = client_scheduler
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), 0]
    if exit_before_kill:
        process.kill.side_effect = ProcessLookupError()

    runner._cleanup_process(process)

    methods = [call[0] for call in process.method_calls]
    assert methods == ["poll", "terminate", "wait", "kill"] + ([] if exit_before_kill else ["wait"])


@pytest.mark.parametrize("extension", ["json", "conf", "yaml"])
def test_exported_task_app_is_rejected_before_simulator_services_start(tmp_path, monkeypatch, extension):
    source = Path(__file__).resolve().parents[4] / "data" / "jobs" / "valid_job"
    job = tmp_path / "exported"
    shutil.copytree(source, job)
    config_path = job / "sag" / "config" / "config_fed_client.json"
    config = json.loads(config_path.read_text())
    config.update(format_version=3, execution_lifetime="task")
    config_path.unlink()
    config_path.with_suffix("." + extension).write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)
    runner = SimulatorRunner(str(job), str(tmp_path / "workspace"), n_clients=2, threads=1)
    create_server = Mock(side_effect=AssertionError("must reject before simulator server startup"))
    create_client = Mock(side_effect=AssertionError("must reject before simulator client startup"))
    monkeypatch.setattr(runner.deployer, "create_fl_server", create_server)
    monkeypatch.setattr(runner.deployer, "create_fl_client", create_client)

    assert runner.setup() is False

    create_server.assert_not_called()
    create_client.assert_not_called()


def test_async_plan_allows_explicit_site_job_with_smaller_physical_pool(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[4] / "data" / "jobs" / "valid_job"
    job = tmp_path / "explicit_job"
    shutil.copytree(source, job)
    meta_path = job / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta[JobMetaKey.MIN_CLIENTS] = 2
    meta[JobMetaKey.DEPLOY_MAP] = {"sag": ["server", "site-1"]}
    meta_path.write_text(json.dumps(meta))
    monkeypatch.setattr(simulator_runner_module, "SimulatorDeployer", Mock())
    runner = SimulatorRunner(str(job), str(tmp_path / "workspace"), clients="site-1")
    runner.args = Namespace(job_folder=str(job))

    with pytest.raises(RuntimeError, match="min 2 clients required"):
        runner.validate_job_data()
    plan = {
        "format_version": 1,
        "async_execution": {"logical_sequence": ["a", "b"], "arrival_sequence": [1, 0]},
    }
    _, _, validated_meta = runner.validate_job_data(plan)
    assert validated_meta[JobMetaKey.MIN_CLIENTS] == 2
    assert json.loads(meta_path.read_text())[JobMetaKey.MIN_CLIENTS] == 2


@pytest.fixture
def simulator_gpu_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(simulator_runner_module, "SimulatorDeployer", Mock())
    runner = SimulatorRunner("job", str(tmp_path / "workspace"), n_clients=2, gpu="0,1", threads=3)
    meta = {JobMetaKey.DEPLOY_MAP: {"app": ["site-1", "site-2"]}}
    monkeypatch.setattr(runner, "validate_job_data", Mock(return_value=(b"job", "job", meta)))
    monkeypatch.setattr(runner, "_deploy_apps", Mock())
    server = Mock()
    server.get_cell.return_value = MockCell()
    monkeypatch.setattr(runner.deployer, "create_fl_server", Mock(return_value=(None, server)))
    for name in (
        "nvflare_fobs_initialize",
        "init_security_content_service",
        "dynamic_log_config",
        "custom_fobs_initialize",
        "register_ext_decomposers",
        "Workspace",
    ):
        monkeypatch.setattr(simulator_runner_module, name, Mock())
    monkeypatch.setattr(simulator_runner_module.AuthorizationService, "initialize", Mock())
    monkeypatch.setattr(simulator_runner_module.AuditService, "the_auditor", None)
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: False)
    query = Mock(side_effect=AssertionError("unexpected nvidia-smi execution"))
    monkeypatch.setattr(gpu_utils.subprocess, "run", query)
    return runner, query


def test_simulator_gpu_setup_accepts_explicit_groups_without_nvidia_smi(simulator_gpu_setup):
    runner, query = simulator_gpu_setup

    assert runner.setup()

    assert split_gpus(runner.args.gpu) == ["0", "1"]
    assert runner.args.threads == 1
    assert runner.client_names == ["site-1", "site-2"]
    runner.deployer.create_fl_server.assert_called_once_with(runner.args)
    runner._deploy_apps.assert_called_once()
    query.assert_not_called()


def test_simulator_gpu_setup_still_fails_for_a_present_but_broken_provider(simulator_gpu_setup, monkeypatch, caplog):
    runner, query = simulator_gpu_setup
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: True)
    query.side_effect = None
    query.return_value = Mock(returncode=9, stdout="", stderr="query failed")

    assert runner.setup() is False

    assert "Failed to call nvidia-smi" in caplog.text
    runner.deployer.create_fl_server.assert_not_called()
    query.assert_called_once()


@pytest.mark.parametrize(
    "output,error",
    [("index\n0\n1\n", None), ("index\n0\n", "These GPUs are not available"), ("index\n", "GPU inventory")],
)
def test_simulator_gpu_setup_preserves_available_inventory_validation(
    simulator_gpu_setup, monkeypatch, caplog, output, error
):
    runner, query = simulator_gpu_setup
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: True)
    query.side_effect = None
    query.return_value = Mock(returncode=0, stdout=output, stderr="")

    assert runner.setup() is (error is None)

    query.assert_called_once()
    if error is None:
        runner.deployer.create_fl_server.assert_called_once()
    else:
        assert error in caplog.text
        runner.deployer.create_fl_server.assert_not_called()


class TestSimulatorRunner:
    @staticmethod
    def _write_resources(workspace):
        local_dir = os.path.join(workspace, WorkspaceConstants.SITE_FOLDER_NAME)
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, "resources.json"), "w") as f:
            json.dump({"class_allow_list": ["nvflare."]}, f)

    def setup_method(self, method):
        self.workspace_name = str(uuid.uuid4())
        self.cwd = os.getcwd()
        os.makedirs(os.path.join(self.cwd, self.workspace_name, WorkspaceConstants.STARTUP_FOLDER_NAME))
        self._write_resources(os.path.join(self.cwd, self.workspace_name))

    def teardown_method(self, method):
        os.chdir(self.cwd)
        shutil.rmtree(os.path.join(self.cwd, self.workspace_name))

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    @patch("nvflare.private.fed.server.fed_server.BaseServer.get_cell", return_value=MockCell())
    def test_valid_job_simulate_setup(self, mock_deploy, mock_admin, mock_register, mock_cell):
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        runner = SimulatorRunner(job_folder=job_folder, workspace=self.workspace_name, threads=1)
        assert runner.setup()

        expected_clients = ["site-1", "site-2"]
        client_names = []
        for client in runner.client_names:
            client_names.append(client)
        assert sorted(client_names) == sorted(expected_clients)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    @patch("nvflare.private.fed.server.fed_server.BaseServer.get_cell", return_value=MockCell())
    def test_client_names_setup(self, mock_deploy, mock_admin, mock_register, mock_cell):
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        runner = SimulatorRunner(job_folder=job_folder, workspace=self.workspace_name, clients="site-1", threads=1)
        assert runner.setup()

        expected_clients = ["site-1"]
        client_names = []
        for client in runner.client_names:
            client_names.append(client)
        assert sorted(client_names) == sorted(expected_clients)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    @patch("nvflare.private.fed.server.fed_server.BaseServer.get_cell", return_value=MockCell())
    def test_no_app_for_client(self, mock_deploy, mock_admin, mock_register, mock_cell):
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        runner = SimulatorRunner(job_folder=job_folder, workspace=self.workspace_name, n_clients=3, threads=1)
        assert not runner.setup()

    @pytest.mark.parametrize(
        "client_names, gpus, expected_split_names",
        [
            (["1", "2", "3", "4"], ["0", "1"], [["1", "3"], ["2", "4"]]),
            (["1", "2", "3", "4", "5"], ["0", "1"], [["1", "3", "5"], ["2", "4"]]),
            (["1", "2", "3", "4", "5"], ["0", "1", "2"], [["1", "4"], ["2", "5"], ["3"]]),
        ],
    )
    def test_split_names(self, client_names, gpus, expected_split_names):
        runner = SimulatorRunner(job_folder="", workspace="")
        split_names = runner.split_clients(client_names, gpus)
        assert sorted(split_names) == sorted(expected_split_names)

    @pytest.mark.parametrize("execution_error,expected_status", [(False, 0), (True, 2)])
    def test_server_fatal_error_sets_simulator_status(self, execution_error, expected_status):
        runner = SimulatorRunner(job_folder="", workspace="")
        runner.server = MagicMock()
        fl_ctx = runner.server.engine.new_context.return_value.__enter__.return_value
        fl_ctx.get_prop.side_effect = lambda key, default=None: (
            execution_error if key == FLContextKey.FATAL_SYSTEM_ERROR else None
        )

        assert runner._get_server_run_status() == expected_status

        assert fl_ctx.get_prop.call_args_list[0].args == (FLContextKey.FATAL_SYSTEM_ERROR, False)
        assert fl_ctx.get_prop.call_args_list[1].args == (SIMULATION_PLAN_PROP,)

    @pytest.mark.parametrize("server_workflow_done,expected_status", [(False, 2), (True, 0)])
    def test_client_fatal_error_sets_simulator_status(self, server_workflow_done, expected_status):
        runner = SimulatorRunner(job_folder="", workspace="")
        runner.server = MagicMock()
        fl_ctx = runner.server.engine.new_context.return_value.__enter__.return_value
        fl_ctx.get_prop.side_effect = lambda key, default=None: (
            False if key == FLContextKey.FATAL_SYSTEM_ERROR else None
        )
        runner.client_fatal_error = True
        runner.server_workflow_done = server_workflow_done

        assert runner._get_server_run_status() == expected_status

    @pytest.mark.parametrize(
        "gpus, expected_gpus",
        [
            ("[0,1],[1, 2]", ["0,1", "1,2"]),
            ("[0,1],[3]", ["0,1", "3"]),
            ("[0,1],[ 3 ]", ["0,1", "3"]),
            ("[02,1],[ a ]", ["02,1", "a"]),
            ("[]", [""]),
            ("[0,1],3", ["0,1", "3"]),
            ("[0,1],[1,2,3],3", ["0,1", "1,2,3", "3"]),
            ("0,1,2", ["0", "1", "2"]),
        ],
    )
    def test_split_gpus_success(self, gpus, expected_gpus):
        splitted_gpus = split_gpus(gpus)
        assert splitted_gpus == expected_gpus

    @pytest.mark.parametrize(
        "gpus",
        [
            "[0,1],3]",
            "0,1,[2",
            "[0,1]extra",
            "[1, [2, 3], 4]",
        ],
    )
    def test_split_gpus_fail(self, gpus):
        with pytest.raises(ValueError):
            split_gpus(gpus)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    @patch("nvflare.private.fed.server.fed_server.BaseServer.get_cell", return_value=MockCell())
    def test_start_server_app(self, mock_deploy, mock_admin, mock_register, mock_cell):
        with TemporaryDirectory() as workspace:
            self._write_resources(workspace)
            job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
            runner = SimulatorRunner(
                job_folder=job_folder,
                workspace=workspace,
            )
            runner.setup()

            with patch("nvflare.private.fed.simulator.simulator_server.SimulatorServer.run_engine"):
                with patch("nvflare.private.fed.simulator.simulator_server.SimulatorServer.create_job_cell"):
                    server_thread = CheckedThread(target=runner.start_server_app, args=[runner.args])
                    server_thread.start()

                    try:

                        def server_started():
                            server_thread.raise_if_failed()
                            return runner.server.engine.engine_info.status == MachineStatus.STARTED

                        wait_for(server_started, message="simulator server app did not start")
                        fl_ctx = runner.server.engine.new_context()
                        workspace_obj = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
                        assert workspace_obj.get_root_dir() == os.path.join(workspace, "server")
                        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
                        assert job_meta[JobMetaKey.JOB_NAME.value] == "sag"
                        assert runner.server.engine.get_cell() is runner.server.job_cell
                        assert runner.server.engine.run_manager.cell is runner.server.job_cell

                    finally:
                        runner.server.logger = Mock()
                        runner.server.engine.asked_to_stop = True
                        join_thread(server_thread)

    def test_get_new_sys_path_with_empty(self):
        args = Namespace(workspace="/tmp")
        args.set = []
        runner = SimulatorClientRunner(None, args, [], None, None, None)
        old_sys_path = copy.deepcopy(sys.path)
        sys.path.insert(0, "")
        sys.path.append("/temp/test")
        new_sys_path = runner._get_new_sys_path()
        assert old_sys_path == new_sys_path
        sys.path = old_sys_path

    def test_get_new_sys_path_with_multiple_empty(self):
        args = Namespace(workspace="/tmp")
        args.set = []
        runner = SimulatorClientRunner(None, args, [], None, None, None)
        old_sys_path = copy.deepcopy(sys.path)
        sys.path.insert(0, "")
        if len(sys.path) > 2:
            sys.path.insert(2, "")
        sys.path.append("/temp/test")
        new_sys_path = runner._get_new_sys_path()
        assert old_sys_path == new_sys_path
        sys.path = old_sys_path

    def test_get_new_sys_path(self):
        args = Namespace(workspace="/tmp")
        args.set = []
        runner = SimulatorClientRunner(None, args, [], None, None, None)
        old_sys_path = copy.deepcopy(sys.path)
        sys.path.append("/temp/test")
        new_sys_path = runner._get_new_sys_path()
        assert old_sys_path == new_sys_path
        sys.path = old_sys_path

    def test_create_connection_does_not_require_authkey(self):
        args = Namespace(workspace="/tmp")
        args.set = []
        runner = SimulatorClientRunner(None, args, [], None, None, None)
        conn = Mock()

        with patch("nvflare.private.fed.app.simulator.simulator_runner.Client", return_value=conn) as client:
            result = runner._create_connection(1234)

        assert result is conn
        client.assert_called_once_with(("localhost", 1234))

    def test_do_one_task_keeps_paths_with_spaces_intact(self, tmp_path):
        workspace = str(tmp_path / "work space")
        log_config = str(tmp_path / "log dir" / "log_config.json")
        executable = str(tmp_path / "py env" / "bin" / "python")
        for folder in ("startup", "local"):  # Workspace() requires both
            (tmp_path / "work space" / "site-1" / folder).mkdir(parents=True)
        args = Namespace(workspace=workspace, log_config=log_config)
        args.set = []
        runner = SimulatorClientRunner(
            str(tmp_path / "server custom"), args, [], None, Namespace(workspace=workspace), {}
        )
        client = Mock()
        client.client_name = "site-1"
        client.token = "token"
        client.cell.get_root_url_for_child.return_value = "tcp://root"
        client.cell.get_internal_listener_url.return_value = "tcp://parent"
        process = Mock()
        process.poll.return_value = None
        conn = Mock()
        conn.recv.side_effect = [True, None, False]  # stop_run, end_run_client, fatal_error

        with patch("nvflare.private.fed.app.simulator.simulator_runner.sys.executable", executable):
            with patch("nvflare.private.fed.app.simulator.simulator_runner.get_open_ports", return_value=[12345]):
                with patch.object(runner, "_create_connection", return_value=conn):
                    with patch(
                        "nvflare.private.fed.app.simulator.simulator_runner.popen_in_new_session", return_value=process
                    ) as popen:
                        runner.do_one_task(client, num_of_threads=0, gpu=None, lock=threading.Lock())

        launched = popen.call_args.args[0]
        assert launched[0] == executable
        assert launched[launched.index("-o") + 1] == os.path.join(workspace, "site-1")
        assert launched[launched.index("--logging_config") + 1] == log_config
        assert launched[launched.index("--simulator_root") + 1] == workspace

    @pytest.mark.parametrize("fatal_error,expected", [(False, []), (True, ["site-1"])])
    def test_do_one_task_records_client_fatal_error(self, tmp_path, fatal_error, expected):
        workspace = str(tmp_path)
        for folder in ("startup", "local"):
            (tmp_path / "site-1" / folder).mkdir(parents=True)
        args = Namespace(workspace=workspace, log_config=None)
        args.set = []
        runner = SimulatorClientRunner(str(tmp_path / "custom"), args, [], None, Namespace(workspace=workspace), {})
        client = Mock()
        client.client_name = "site-1"
        client.token = "token"
        process = Mock()
        process.poll.return_value = None
        conn = Mock()
        conn.recv.side_effect = [True, "site-1", fatal_error]  # stop_run, end_run_client, fatal_error

        with patch("nvflare.private.fed.app.simulator.simulator_runner.get_open_ports", return_value=[12345]):
            with patch.object(runner, "_create_connection", return_value=conn):
                with patch(
                    "nvflare.private.fed.app.simulator.simulator_runner.popen_in_new_session", return_value=process
                ):
                    runner.do_one_task(client, num_of_threads=0, gpu=None, lock=threading.Lock())

        assert runner.fatal_clients == set(expected)

    @pytest.mark.parametrize("task_name", [RunnerTask.END_RUN, RunnerTask.TASK_EXEC])
    def test_worker_reports_client_fatal_error(self, task_name):
        worker = simulator_worker.ClientTaskWorker()
        client = MagicMock()
        client.client_name = "site-1"
        client_runner = Mock()
        client_runner.fetch_and_run_one_task.return_value = (1.0, True)
        client_runner.run_abort_signal.triggered = True
        fl_ctx = client.run_manager.new_context.return_value.__enter__.return_value
        fl_ctx.get_prop.side_effect = lambda key, default=None: {
            FLContextKey.RUNNER: client_runner,
            FLContextKey.FATAL_SYSTEM_ERROR: True,
        }.get(key, default)

        with patch.object(worker, "fire_event"):
            _, stop_run, end_run_client, fatal_error = worker.do_one_task(client, Namespace(task_name=task_name))

        assert stop_run
        assert end_run_client == client.client_name
        assert fatal_error

    @pytest.mark.parametrize("task_name", [RunnerTask.END_RUN, RunnerTask.TASK_EXEC])
    @pytest.mark.parametrize(
        "failure_event", [None, EventType.ABORT_TASK, EventType.ABOUT_TO_END_RUN, EventType.END_RUN]
    )
    def test_worker_checks_actual_end_run_dispatch_context(self, task_name, failure_event):
        class Finalizer(FLComponent):
            def handle_event(self, event_type, fl_ctx):
                if event_type == failure_event:
                    raise OSError("could not write final artifact")

        worker = simulator_worker.ClientTaskWorker()
        client_runner = ClientRunner.__new__(ClientRunner)
        FLComponent.__init__(client_runner)
        client_runner.running_tasks = {}
        client_runner.task_lock = threading.Lock()
        client_runner._run_abort_requested = True
        client_runner.log_info = Mock()
        client_runner.log_debug = Mock()
        client_runner.fetch_and_run_one_task = Mock(return_value=(1.0, True))
        client_runner.run_abort_signal = Signal()
        client_runner.run_abort_signal.trigger(True)
        engine = Mock()
        client_runner.engine = engine
        manager = FLContextManager(engine=engine, private_stickers={FLContextKey.RUNNER: client_runner})
        contexts = []

        def new_context():
            ctx = manager.new_context()
            contexts.append(ctx)
            return ctx

        engine.new_context.side_effect = new_context
        finalizer = Finalizer()
        engine.fire_event.side_effect = lambda event, ctx: fire_event_to_components(event, [finalizer], ctx)
        client = SimpleNamespace(client_name="site-1", run_manager=engine)

        _, stop_run, end_run_client, fatal_error = worker.do_one_task(client, Namespace(task_name=task_name))

        handler_fails = failure_event is not None
        assert stop_run
        assert end_run_client == (None if handler_fails else "site-1")
        assert fatal_error is handler_fails
        assert not contexts[0].get_prop(FLContextKey.EXCEPTIONS)  # The worker's outer context cannot see the failure.
        # Readiness cleared earlier event errors, but they must still make the outcome fail.
        assert bool(contexts[1].get_prop(FLContextKey.EXCEPTIONS)) is (failure_event == EventType.END_RUN)

    def test_worker_create_connection_does_not_require_authkey(self):
        conn = Mock()
        listener = Mock()
        listener.accept.return_value = conn

        with patch(
            "nvflare.private.fed.app.simulator.simulator_worker.Listener", return_value=listener
        ) as listener_cls:
            result = simulator_worker._create_connection(1234)

        assert result is conn
        listener_cls.assert_called_once_with(("localhost", 1234))
