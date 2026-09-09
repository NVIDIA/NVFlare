# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nvflare.recipe.poc_env import PocEnv
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

PROJECT_CONFIG = {"name": "poc"}
SERVICE_CONFIG = {SC.FLARE_SERVER: "server", SC.FLARE_CLIENTS: ["site-1"]}


def _configure_successful_deploy(monkeypatch, env, prepare=None, submit=None):
    import nvflare.recipe.poc_env as poc_env_module

    if prepare is None:

        def prepare(**kwargs):
            Path(kwargs["workspace"]).mkdir(parents=True)

    if submit is None:

        def submit(job):
            return "job-id"

    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: [])
    monkeypatch.setattr(env, "_preflight_ports_before_provision", lambda: None)
    monkeypatch.setattr(poc_env_module, "prepare_poc_provision", prepare)
    monkeypatch.setattr(poc_env_module, "_start_poc", lambda **kwargs: None)
    monkeypatch.setattr(
        poc_env_module,
        "setup_service_config",
        lambda path: (PROJECT_CONFIG, SERVICE_CONFIG),
    )
    monkeypatch.setattr(poc_env_module, "_wait_for_poc_system_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(env, "_wait_for_services_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(env, "_get_session_manager", lambda: SimpleNamespace(submit_job=submit))


def test_poc_env_initialization():
    """Test PocEnv initialization with default values."""
    env = PocEnv()

    assert env.num_clients == 2
    assert env.gpu_ids == []
    assert env.study == "default"
    assert env.poc_workspace.startswith(f"{env._poc_workspace_root}.recipe-")


def test_deploy_rejects_overlapping_calls_on_same_environment():
    env = PocEnv()
    assert env._deployment_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="deployment in progress"):
            env.deploy(object())
    finally:
        env._deployment_lock.release()


def test_stop_waits_for_deployment_to_leave_instance_guard(monkeypatch):
    env = PocEnv()
    stop_called = threading.Event()
    stop_completed = threading.Event()
    stop_args = []

    def stop_impl(clean_up):
        stop_args.append(clean_up)

    def call_stop():
        stop_called.set()
        env.stop(clean_up=True)
        stop_completed.set()

    monkeypatch.setattr(env, "_stop", stop_impl)
    assert env._deployment_lock.acquire(blocking=False)
    stop_thread = threading.Thread(target=call_stop)
    stop_thread.start()
    try:
        assert stop_called.wait(timeout=1)
        assert not stop_completed.wait(timeout=0.05)
    finally:
        env._deployment_lock.release()
    stop_thread.join(timeout=1)

    assert not stop_thread.is_alive()
    assert stop_completed.is_set()
    assert stop_args == [True]


@patch("nvflare.recipe.poc_env.get_poc_workspace")
def test_poc_env_initialization_with_custom_values(mock_get_workspace, tmp_path):
    """Test PocEnv initialization with custom values."""
    configured_workspace = tmp_path / "poc"
    mock_get_workspace.return_value = str(configured_workspace)

    env = PocEnv(num_clients=3, gpu_ids=[0, 1])

    assert env._poc_workspace_root == str(configured_workspace)
    assert env.poc_workspace.startswith(f"{configured_workspace}.recipe-")
    assert env.poc_workspace != str(configured_workspace)
    assert env.num_clients == 3
    assert env.gpu_ids == [0, 1]


@patch("nvflare.recipe.poc_env.get_poc_workspace")
def test_poc_env_normalizes_workspace_trailing_separator(mock_get_workspace, tmp_path):
    configured_workspace = tmp_path / "poc-workspace"
    mock_get_workspace.return_value = f"{configured_workspace}{os.sep}"

    env = PocEnv()

    assert env._poc_workspace_root == str(configured_workspace)
    assert env.poc_workspace.startswith(f"{configured_workspace}.recipe-")
    assert os.path.dirname(env.poc_workspace) == str(tmp_path)


@patch("nvflare.recipe.poc_env.get_poc_workspace")
def test_poc_env_instances_use_distinct_workspaces(mock_get_workspace, tmp_path):
    configured_workspace = tmp_path / "poc"
    mock_get_workspace.return_value = str(configured_workspace)

    first = PocEnv()
    second = PocEnv()

    assert first.poc_workspace != second.poc_workspace
    assert first.poc_workspace.startswith(f"{configured_workspace}.recipe-")
    assert second.poc_workspace.startswith(f"{configured_workspace}.recipe-")


def test_poc_env_validation():
    """Test PocEnv validation for invalid configurations."""
    with pytest.raises(ValueError, match="Input should be greater than 0"):
        PocEnv(num_clients=0)

    with pytest.raises(ValueError, match="Input should be greater than 0"):
        PocEnv(num_clients=-1)

    with pytest.raises(ValueError, match="clients list cannot be empty"):
        PocEnv(clients=[])

    with pytest.raises(ValueError, match="Inconsistent"):
        PocEnv(num_clients=3, clients=["site1", "site2"])


def test_poc_env_none_num_clients_raises():
    """Test that PocEnv(num_clients=None) raises ValueError instead of crashing with TypeError."""
    with pytest.raises(ValueError, match="num_clients must be greater than 0"):
        PocEnv(num_clients=None, clients=None)


def test_poc_env_client_names():
    """Test PocEnv client name generation and validation."""
    env = PocEnv(num_clients=3)
    assert env.clients is None
    assert env.num_clients == 3

    custom_clients = ["client-a", "client-b"]
    env = PocEnv(clients=custom_clients)
    assert env.clients == custom_clients
    assert env.num_clients == 2

    env = PocEnv(num_clients=2, clients=["site-x", "site-y"])
    assert env.clients == ["site-x", "site-y"]
    assert env.num_clients == 2


@patch("nvflare.recipe.poc_env.get_poc_workspace")
def test_poc_env_initialization_with_study(mock_get_workspace, tmp_path):
    mock_get_workspace.return_value = str(tmp_path / "poc")

    env = PocEnv(num_clients=2, study="cancer-research")

    assert env.study == "cancer-research"


def test_poc_env_rejects_invalid_study_name():
    with pytest.raises(ValueError):
        PocEnv(study="Bad Study")


def test_deploy_preflight_failure_does_not_create_workspace(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    configured_workspace.mkdir()
    retained_result = configured_workspace / "prior-result.txt"
    retained_result.write_text("keep me")
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: ["missing.py"])
    env = PocEnv()

    with pytest.raises(ValueError, match="scripts do not exist locally"):
        env.deploy(object())

    assert retained_result.read_text() == "keep me"
    assert not os.path.exists(env.poc_workspace)


def test_deploy_rejects_running_configured_cli_workspace(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    configured_workspace.mkdir()
    retained_result = configured_workspace / "prior-result.txt"
    retained_result.write_text("keep me")
    provision_calls = []
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: [])
    monkeypatch.setattr(
        poc_env_module,
        "setup_service_config",
        lambda path: (PROJECT_CONFIG, SERVICE_CONFIG),
    )
    monkeypatch.setattr(
        PocEnv,
        "_running_services",
        staticmethod(
            lambda project_config, service_config, workspace: (
                ["server"] if workspace == str(configured_workspace) else []
            )
        ),
    )
    monkeypatch.setattr(
        poc_env_module,
        "prepare_poc_provision",
        lambda **kwargs: provision_calls.append(kwargs),
    )
    env = PocEnv()

    with pytest.raises(RuntimeError, match="nvflare poc stop"):
        env.deploy(object())

    assert provision_calls == []
    assert retained_result.read_text() == "keep me"
    assert not os.path.exists(env.poc_workspace)


def test_deploy_rejects_unavailable_custom_port_before_provision(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module
    import nvflare.tool.poc.poc_commands as poc_commands

    configured_workspace = tmp_path / "poc"
    project_conf = tmp_path / "project.yml"
    project_conf.write_text(
        "participants:\n"
        "  - name: server\n"
        "    type: server\n"
        "    fed_learn_port: 18002\n"
        "    admin_port: 18003\n"
    )
    provision_calls = []
    checked_ports = []
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: [])
    monkeypatch.setattr(
        poc_env_module,
        "prepare_poc_provision",
        lambda **kwargs: provision_calls.append(kwargs),
    )

    def check_port(port, host):
        checked_ports.append((port, host))
        return (False, "in_use") if port == 18002 else (True, None)

    monkeypatch.setattr(poc_commands, "_is_local_port_available", check_port)
    env = PocEnv(project_conf_path=str(project_conf))

    with pytest.raises(RuntimeError, match="other Recipe PocEnv") as exc_info:
        env.deploy(object())

    assert "18002" in str(exc_info.value)
    assert checked_ports == [(18002, "127.0.0.1"), (18003, "127.0.0.1")]
    assert provision_calls == []
    assert env._deployment_started is False
    assert not os.path.exists(env.poc_workspace)


def test_deploy_rechecks_ports_after_provision(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module
    import nvflare.tool.poc.poc_commands as poc_commands

    configured_workspace = tmp_path / "poc"
    project_config = {
        "name": "poc",
        "participants": [
            {
                "name": "server",
                "type": "server",
                "fed_learn_port": 18002,
                "admin_port": 18003,
            }
        ],
    }
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: [])
    provisioned_workspaces = []
    start_calls = []
    checked_ports = []

    def prepare(**kwargs):
        provisioned_workspaces.append(kwargs["workspace"])
        Path(kwargs["workspace"]).mkdir(parents=True)

    monkeypatch.setattr(poc_env_module, "prepare_poc_provision", prepare)
    monkeypatch.setattr(poc_env_module, "setup_service_config", lambda path: (project_config, SERVICE_CONFIG))

    def check_port(port, host):
        checked_ports.append((port, host))
        return (False, "in_use") if port == 18002 else (True, None)

    monkeypatch.setattr(poc_commands, "_is_local_port_available", check_port)
    monkeypatch.setattr(poc_env_module, "_start_poc", lambda **kwargs: start_calls.append(kwargs))
    env = PocEnv()

    with pytest.raises(RuntimeError, match="POC service port preflight failed") as exc_info:
        env.deploy(object())

    assert "18002" in str(exc_info.value)
    assert checked_ports == [
        (8002, "127.0.0.1"),
        (8003, "127.0.0.1"),
        (18002, "127.0.0.1"),
        (18003, "127.0.0.1"),
    ]
    assert provisioned_workspaces == [env.poc_workspace]
    assert env._deployment_started is True
    assert start_calls == []
    assert not os.path.exists(env.poc_workspace)


@pytest.mark.parametrize("endpoint", ["ssh://docker.example", "tcp://docker.example:2375"])
def test_remote_docker_daemon_skips_local_loopback_port_preflight(monkeypatch, endpoint):
    import nvflare.recipe.poc_env as poc_env_module

    docker_service_config = {**SERVICE_CONFIG, SC.IS_DOCKER_RUN: True}
    monkeypatch.setenv("DOCKER_CONTEXT", "remote-builder")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(poc_env_module, "_get_docker_endpoint", lambda docker_env: endpoint)
    monkeypatch.setattr(
        poc_env_module,
        "_build_poc_port_preflight",
        lambda project_config: pytest.fail("local port preflight must not inspect a remote Docker host"),
    )
    monkeypatch.setattr(PocEnv, "_get_docker_service_state", staticmethod(lambda _container_name: None))

    PocEnv._ensure_shared_resources_available(PROJECT_CONFIG, docker_service_config)


@pytest.mark.parametrize("endpoint", ["", "unix:///var/run/docker.sock"])
def test_local_or_unknown_docker_endpoint_keeps_local_loopback_port_preflight(monkeypatch, endpoint):
    import nvflare.recipe.poc_env as poc_env_module

    docker_service_config = {**SERVICE_CONFIG, SC.IS_DOCKER_RUN: True}
    port_checks = []
    monkeypatch.setattr(poc_env_module, "_get_docker_endpoint", lambda docker_env: endpoint)
    monkeypatch.setattr(
        poc_env_module,
        "_build_poc_port_preflight",
        lambda project_config: port_checks.append(project_config) or {"conflicts": []},
    )
    monkeypatch.setattr(PocEnv, "_get_docker_service_state", staticmethod(lambda _container_name: None))

    PocEnv._ensure_shared_resources_available(PROJECT_CONFIG, docker_service_config)

    assert port_checks == [PROJECT_CONFIG]


@pytest.mark.parametrize("lookup_stage", ["show", "inspect"])
@pytest.mark.parametrize("lookup_result", ["nonzero", "empty", "os_error", "timeout"])
def test_context_lookup_failure_rejects_port_conflict_without_consuming_env(
    tmp_path, monkeypatch, lookup_stage, lookup_result
):
    import nvflare.recipe.poc_env as poc_env_module
    import nvflare.tool.poc.poc_commands as poc_commands

    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "NVFL_DOCKER_SOCK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: [])
    env = PocEnv(docker_image="nvflare:test")
    monkeypatch.setattr(env, "_is_poc_workspace_running", lambda workspace: False)
    monkeypatch.setattr(
        poc_env_module,
        "prepare_poc_provision",
        lambda **kwargs: pytest.fail("a port conflict must be rejected before provisioning"),
    )
    monkeypatch.setattr(
        poc_env_module,
        "_start_poc",
        lambda **kwargs: pytest.fail("a port conflict must be rejected before starting services"),
    )
    context_calls = []

    def lookup_context(command, **kwargs):
        context_calls.append(command)
        assert command[:2] == ["docker", "context"]
        if command[2] != lookup_stage:
            assert command == ["docker", "context", "show"]
            return SimpleNamespace(returncode=0, stdout="default\n")
        if lookup_result == "os_error":
            raise OSError("Docker context lookup unavailable")
        if lookup_result == "timeout":
            raise subprocess.TimeoutExpired(command, timeout=5)
        return SimpleNamespace(returncode=1 if lookup_result == "nonzero" else 0, stdout="")

    checked_ports = []

    def check_port(port, host):
        checked_ports.append((port, host))
        return (False, "in_use") if port == 8002 else (True, None)

    monkeypatch.setattr(poc_commands.subprocess, "run", lookup_context)
    monkeypatch.setattr(poc_commands, "_is_local_port_available", check_port)

    # Retrying the same instance must reach preflight again, not the one-shot guard.
    for _ in range(2):
        with pytest.raises(RuntimeError, match="POC service port preflight failed.*8002"):
            env.deploy(object())
        assert env._deployment_started is False
        assert env._services_may_have_started is False
        assert not Path(env.poc_workspace).exists()

    expected_calls = [["docker", "context", "show"]]
    if lookup_stage == "inspect":
        expected_calls.append(["docker", "context", "inspect", "default", "--format", "{{.Endpoints.docker.Host}}"])
    assert context_calls == expected_calls * 2
    assert checked_ports == [(8002, "127.0.0.1"), (8003, "127.0.0.1")] * 2


def test_deploy_rejects_existing_docker_participant_name_without_stopping_it(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    docker_service_config = {**SERVICE_CONFIG, SC.IS_DOCKER_RUN: True}
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    env = PocEnv(docker_image="nvflare:test")
    start_calls = []
    stop_calls = []

    def prepare(**kwargs):
        Path(kwargs["workspace"]).mkdir(parents=True)

    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: [])
    monkeypatch.setattr(poc_env_module, "prepare_poc_provision", prepare)
    monkeypatch.setattr(
        poc_env_module,
        "setup_service_config",
        lambda path: (PROJECT_CONFIG, docker_service_config),
    )
    monkeypatch.setattr(PocEnv, "_docker_daemon_is_local", staticmethod(lambda: False))
    inspected_names = []

    def container_state(container_name):
        inspected_names.append(container_name)
        return False if container_name.startswith("nvflare-recipe-") else None

    monkeypatch.setattr(PocEnv, "_get_docker_service_state", staticmethod(container_state))
    monkeypatch.setattr(poc_env_module, "_start_poc", lambda **kwargs: start_calls.append(kwargs))
    monkeypatch.setattr(poc_env_module, "_stop_poc", lambda **kwargs: stop_calls.append(kwargs))

    with pytest.raises(RuntimeError, match=r"container name\(s\) already exist") as exc_info:
        env.deploy(object())

    assert env._docker_container_names["server"] in str(exc_info.value)
    assert env._docker_container_names["server"] in inspected_names
    assert start_calls == []
    assert stop_calls == []
    assert not os.path.exists(env.poc_workspace)


def test_concurrent_docker_deploy_failure_stops_only_its_own_containers(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    docker_service_config = {**SERVICE_CONFIG, SC.IS_DOCKER_RUN: True}
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    first = PocEnv(docker_image="nvflare:test")
    second = PocEnv(docker_image="nvflare:test")
    start_barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    running_containers = set()
    stop_calls = []
    removed_networks = []

    def prepare(**kwargs):
        Path(kwargs["workspace"]).mkdir(parents=True)

    def container_state(container_name):
        with state_lock:
            return True if container_name in running_containers else None

    def start(**kwargs):
        with state_lock:
            running_containers.update(kwargs["docker_container_names"].values())
        start_barrier.wait(timeout=2)
        if kwargs["poc_workspace"] == second.poc_workspace:
            raise RuntimeError("simulated losing Docker startup")

    def stop(**kwargs):
        names = dict(kwargs["docker_container_names"])
        stop_calls.append(names)
        with state_lock:
            for container_name in names.values():
                running_containers.discard(container_name)

    def remove_network(env):
        removed_networks.append(env._docker_network_name)
        env._docker_network_name = None

    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: [])
    monkeypatch.setattr(poc_env_module, "prepare_poc_provision", prepare)
    monkeypatch.setattr(
        poc_env_module,
        "setup_service_config",
        lambda path: (PROJECT_CONFIG, docker_service_config),
    )
    monkeypatch.setattr(PocEnv, "_docker_daemon_is_local", staticmethod(lambda: False))
    monkeypatch.setattr(PocEnv, "_get_docker_service_state", staticmethod(container_state))
    monkeypatch.setattr(PocEnv, "_wait_for_services_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(PocEnv, "_remove_docker_network", remove_network)
    monkeypatch.setattr(poc_env_module, "_wait_for_poc_system_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(poc_env_module, "_start_poc", start)
    monkeypatch.setattr(poc_env_module, "_stop_poc", stop)
    monkeypatch.setattr(
        PocEnv,
        "_get_session_manager",
        lambda self: SimpleNamespace(submit_job=lambda job: "job-id"),
    )

    results = {}

    def deploy(name, env):
        try:
            results[name] = env.deploy(object())
        except BaseException as error:
            results[name] = error

    first_thread = threading.Thread(target=deploy, args=("first", first))
    second_thread = threading.Thread(target=deploy, args=("second", second))
    first_thread.start()
    second_thread.start()
    first_thread.join(timeout=3)
    second_thread.join(timeout=3)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert results["first"] == "job-id"
    assert isinstance(results["second"], RuntimeError)
    assert "simulated losing Docker startup" in str(results["second"])
    assert set(first._docker_container_names.values()).isdisjoint(second._docker_container_names.values())
    assert first._docker_network_name != removed_networks[0]
    assert stop_calls == [second._docker_container_names]
    assert running_containers == set(first._docker_container_names.values())
    second_service_config = second._with_docker_container_names(second.poc_workspace, docker_service_config)
    assert second._running_services(PROJECT_CONFIG, second_service_config, second.poc_workspace) == []
    assert not os.path.exists(second.poc_workspace)


def test_deploy_does_not_modify_configured_cli_workspace(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    configured_workspace.mkdir()
    retained_result = configured_workspace / "prior-result.txt"
    retained_result.write_text("keep me")
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    env = PocEnv()
    _configure_successful_deploy(monkeypatch, env)

    assert env.deploy(object()) == "job-id"

    assert env.poc_workspace != str(configured_workspace)
    assert Path(env.poc_workspace).is_dir()
    assert retained_result.read_text() == "keep me"


def test_project_config_may_live_in_configured_cli_workspace(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    configured_workspace.mkdir()
    project_conf = configured_workspace / "project.yml"
    project_conf.write_text("name: retained")
    prepared = {}

    def prepare(**kwargs):
        prepared.update(kwargs)
        Path(kwargs["workspace"]).mkdir(parents=True)

    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    env = PocEnv(project_conf_path=str(project_conf))
    _configure_successful_deploy(monkeypatch, env, prepare=prepare)

    assert env.deploy(object()) == "job-id"

    assert prepared["project_conf_path"] == str(project_conf)
    assert prepared["workspace"] == env.poc_workspace
    assert project_conf.read_text() == "name: retained"


def test_failed_provisioning_cleans_only_run_workspace(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    configured_workspace.mkdir()
    retained_result = configured_workspace / "prior-result.txt"
    retained_result.write_text("keep me")
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: [])

    def write_then_fail(**kwargs):
        workspace = Path(kwargs["workspace"])
        workspace.mkdir(parents=True)
        (workspace / "partial-result.txt").write_text("partial")
        raise RuntimeError("provisioning failed")

    monkeypatch.setattr(poc_env_module, "prepare_poc_provision", write_then_fail)
    env = PocEnv()
    run_workspace = env.poc_workspace

    with pytest.raises(RuntimeError, match="provisioning failed"):
        env.deploy(object())

    assert retained_result.read_text() == "keep me"
    assert not os.path.exists(run_workspace)

    with pytest.raises(RuntimeError, match="already been used"):
        env.deploy(object())


@pytest.mark.parametrize("failure_stage", ["start", "readiness", "submission"])
def test_deploy_failure_cleans_only_run_workspace(tmp_path, monkeypatch, failure_stage):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    configured_workspace.mkdir()
    retained_result = configured_workspace / "prior-result.txt"
    retained_result.write_text("keep me")
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    env = PocEnv()

    def prepare(**kwargs):
        workspace = Path(kwargs["workspace"])
        workspace.mkdir(parents=True)
        (workspace / "partial-result.txt").write_text("partial")

    _configure_successful_deploy(monkeypatch, env, prepare=prepare)
    monkeypatch.setattr(env, "_check_poc_running", lambda: False)
    if failure_stage == "start":
        monkeypatch.setattr(
            poc_env_module,
            "_start_poc",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("start failed")),
        )
    elif failure_stage == "readiness":
        monkeypatch.setattr(
            env,
            "_wait_for_services_ready",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("readiness failed")),
        )
    else:
        monkeypatch.setattr(
            env,
            "_get_session_manager",
            lambda: SimpleNamespace(submit_job=lambda job: (_ for _ in ()).throw(RuntimeError("submission failed"))),
        )
    run_workspace = env.poc_workspace

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        env.deploy(object())

    assert retained_result.read_text() == "keep me"
    assert not os.path.exists(run_workspace)


def test_deploy_reports_incomplete_failure_cleanup(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    env = PocEnv()

    def prepare(**kwargs):
        Path(kwargs["workspace"]).mkdir(parents=True)

    def fail_submission(job):
        raise RuntimeError("submission failed")

    _configure_successful_deploy(monkeypatch, env, prepare=prepare, submit=fail_submission)
    stop_args = []
    monkeypatch.setattr(env, "_stop", lambda clean_up: stop_args.append(clean_up))
    monkeypatch.setattr(env, "_check_poc_running", lambda: True)
    run_workspace = env.poc_workspace

    with pytest.raises(RuntimeError, match="cleanup could not be completed safely") as exc_info:
        env.deploy(object())

    assert "submission failed" in str(exc_info.value)
    assert run_workspace in str(exc_info.value)
    assert "remove this workspace manually" in str(exc_info.value)
    assert "POC services remain running" in str(exc_info.value.__cause__)
    assert os.path.isdir(run_workspace)
    assert stop_args == [True]


def test_deploy_preserves_workspace_when_metadata_is_lost_after_start(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: [])
    startup_attempted = False

    def prepare(**kwargs):
        Path(kwargs["workspace"]).mkdir(parents=True)

    def setup(workspace):
        if workspace == str(configured_workspace) or not startup_attempted:
            return PROJECT_CONFIG, SERVICE_CONFIG
        raise RuntimeError("service configuration unavailable after startup")

    def start(**kwargs):
        nonlocal startup_attempted
        startup_attempted = True
        raise RuntimeError("start failed after losing service configuration")

    monkeypatch.setattr(poc_env_module, "prepare_poc_provision", prepare)
    monkeypatch.setattr(poc_env_module, "_start_poc", start)
    monkeypatch.setattr(poc_env_module, "setup_service_config", setup)
    monkeypatch.setattr(PocEnv, "_running_services", staticmethod(lambda *args: []))
    env = PocEnv()
    run_workspace = env.poc_workspace

    with pytest.raises(RuntimeError, match="cleanup could not be completed safely") as exc_info:
        env.deploy(object())

    assert "service configuration unavailable after startup" in str(exc_info.value)
    assert "Could not determine service state" in str(exc_info.value.__cause__)
    assert Path(run_workspace).is_dir()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(2)])
def test_cleanup_interruption_is_not_converted_to_runtime_error(tmp_path, monkeypatch, interruption):
    import nvflare.recipe.poc_env as poc_env_module

    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    env = PocEnv()

    def prepare(**kwargs):
        Path(kwargs["workspace"]).mkdir(parents=True)

    def fail_submission(job):
        raise RuntimeError("submission failed")

    _configure_successful_deploy(monkeypatch, env, prepare=prepare, submit=fail_submission)
    monkeypatch.setattr(
        env,
        "_clean_up_failed_deployment",
        lambda: (_ for _ in ()).throw(interruption),
    )

    with pytest.raises(type(interruption)):
        env.deploy(object())


def test_new_env_preserves_retained_recipe_workspace(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    retained_workspace = tmp_path / f"poc.recipe-{'b' * 32}"
    retained_workspace.mkdir()
    retained_result = retained_workspace / "result.txt"
    retained_result.write_text("keep me")
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    env = PocEnv()
    _configure_successful_deploy(monkeypatch, env)

    assert env.deploy(object()) == "job-id"
    assert retained_result.read_text() == "keep me"


def test_reusing_stopped_env_is_rejected_and_preserves_its_workspace(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    env = PocEnv()

    def prepare(**kwargs):
        workspace = Path(kwargs["workspace"])
        workspace.mkdir(parents=True)
        (workspace / "run-result.txt").write_text("complete")

    _configure_successful_deploy(monkeypatch, env, prepare=prepare)

    assert env.deploy(object()) == "job-id"
    run_workspace = env.poc_workspace
    env.stop(clean_up=False)

    with pytest.raises(RuntimeError, match="already been used"):
        env.deploy(object())

    assert env.poc_workspace == run_workspace
    assert (Path(run_workspace) / "run-result.txt").read_text() == "complete"


def test_stop_cleanup_removes_only_run_workspace(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    configured_workspace.mkdir()
    retained_result = configured_workspace / "prior-result.txt"
    retained_result.write_text("keep me")
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    env = PocEnv()
    run_workspace = Path(env.poc_workspace)
    run_workspace.mkdir()
    (run_workspace / "result.txt").write_text("temporary")
    monkeypatch.setattr(env, "_check_poc_running", lambda: False)

    env.stop(clean_up=True)

    assert not run_workspace.exists()
    assert retained_result.read_text() == "keep me"


@pytest.mark.parametrize(
    "workspace_name",
    [
        "poc",
        "poc.recipe-",
        f"poc.recipe-{'a' * 31}",
        f"poc.recipe-{'a' * 33}",
        f"poc.recipe-{'g' * 32}",
        f"other.recipe-{'a' * 32}",
        f"nested/poc.recipe-{'a' * 32}",
    ],
)
def test_cleanup_refuses_unmanaged_workspaces(tmp_path, monkeypatch, caplog, workspace_name):
    import nvflare.recipe.poc_env as poc_env_module

    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    env = PocEnv()
    workspace = tmp_path / workspace_name
    workspace.mkdir(parents=True)
    retained_result = workspace / "result.txt"
    retained_result.write_text("keep me")
    env.poc_workspace = str(workspace)

    with pytest.raises(RuntimeError, match="refusing to remove unmanaged POC workspace"):
        env._remove_recipe_workspace()
    with pytest.raises(RuntimeError, match="refusing to clean unmanaged POC workspace"):
        env._clean_up_failed_deployment()

    monkeypatch.setattr(env, "_check_poc_running", lambda: False)
    env.stop(clean_up=True)

    assert retained_result.read_text() == "keep me"
    assert "refusing to remove unmanaged POC workspace" in caplog.text


def test_wait_for_services_ready_rejects_an_exited_client(monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    env = PocEnv()
    monkeypatch.setattr(poc_env_module, "POC_START_READY_TIMEOUT", 0.01)
    monkeypatch.setattr(poc_env_module, "POC_READY_POLL_INTERVAL", 0.001)
    monkeypatch.setattr(env, "_running_services", lambda *args: ["server"])

    with pytest.raises(RuntimeError, match="not running: site-1"):
        env._wait_for_services_ready(PROJECT_CONFIG, SERVICE_CONFIG)


def test_running_services_uses_docker_container_state(monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    inspected = []

    def inspect_container(command, **kwargs):
        inspected.append(command)
        running = command[-1] != "site-1"
        return SimpleNamespace(returncode=0, stdout=f"{str(running).lower()}\n")

    monkeypatch.setenv("DOCKER_HOST", "tcp://docker.example:2375")
    monkeypatch.setattr(poc_env_module.subprocess, "run", inspect_container)
    service_config = {
        SC.FLARE_SERVER: "server",
        SC.FLARE_CLIENTS: ["site-1", "site-2"],
        SC.IS_DOCKER_RUN: True,
    }

    assert PocEnv._running_services(PROJECT_CONFIG, service_config, "/unused") == ["server", "site-2"]
    assert [command[-1] for command in inspected] == ["server", "site-1", "site-2"]


def test_docker_liveness_honors_poc_socket_override(monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    calls = []

    def inspect_container(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="true\n")

    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "remote-context")
    monkeypatch.setenv("NVFL_DOCKER_SOCK", "/run/user/1000/docker.sock")
    monkeypatch.setattr(poc_env_module.subprocess, "run", inspect_container)

    assert PocEnv._is_docker_service_running("site-1") is True
    command, kwargs = calls[0]
    assert command[-1] == "site-1"
    assert kwargs["env"]["DOCKER_HOST"] == "unix:///run/user/1000/docker.sock"
    assert "DOCKER_CONTEXT" not in kwargs["env"]


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(cmd="docker inspect", timeout=5),
        OSError("docker is unavailable"),
    ],
)
def test_docker_liveness_fails_closed_when_inspection_cannot_run(monkeypatch, failure):
    import nvflare.recipe.poc_env as poc_env_module

    monkeypatch.setattr(poc_env_module.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(failure))

    with pytest.raises(RuntimeError, match="Could not determine Docker POC service state"):
        PocEnv._is_docker_service_running("server")


def test_docker_liveness_distinguishes_missing_container_from_inspection_error(monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    missing = SimpleNamespace(returncode=1, stdout="", stderr="Error: No such object: server")
    monkeypatch.setattr(poc_env_module.subprocess, "run", lambda *args, **kwargs: missing)
    assert PocEnv._is_docker_service_running("server") is False

    unavailable = SimpleNamespace(returncode=1, stdout="", stderr="Cannot connect to the Docker daemon")
    monkeypatch.setattr(poc_env_module.subprocess, "run", lambda *args, **kwargs: unavailable)
    with pytest.raises(RuntimeError, match="Cannot connect to the Docker daemon"):
        PocEnv._is_docker_service_running("server")


def test_remove_docker_network_uses_selected_daemon_and_clears_identity(monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    env = PocEnv()
    env._docker_network_name = "nvflare-recipe-run"
    docker_env = {"DOCKER_HOST": "ssh://docker.example"}
    completed = SimpleNamespace(returncode=0, stdout="nvflare-recipe-run\n", stderr="")
    calls = []
    monkeypatch.setattr(poc_env_module, "_docker_cli_env", lambda: docker_env)
    monkeypatch.setattr(
        poc_env_module.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or completed,
    )

    env._remove_docker_network()

    assert calls[0][0][0] == ["docker", "network", "rm", "nvflare-recipe-run"]
    assert calls[0][1]["env"] is docker_env
    assert env._docker_network_name is None


@pytest.mark.parametrize("initially_running", [False, True])
@pytest.mark.parametrize("failure", ["active_endpoints", "daemon_error", "os_error", "timeout"])
def test_stop_preserves_workspace_on_network_failure_and_can_retry(
    tmp_path, monkeypatch, caplog, initially_running, failure
):
    import nvflare.recipe.poc_env as poc_env_module

    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    env = PocEnv(docker_image="nvflare:test")
    workspace = Path(env.poc_workspace)
    workspace.mkdir()
    retained_result = workspace / "result.txt"
    retained_result.write_text("keep me")
    env._docker_network_name = "nvflare-recipe-run"
    env._services_may_have_started = True
    env._session_manager = object()
    running = {"value": initially_running}
    stop_calls = []
    monkeypatch.setattr(env, "_check_poc_running", lambda: running["value"])
    monkeypatch.setattr(env, "_running_services", lambda *args: ["server"] if running["value"] else [])
    monkeypatch.setattr(
        poc_env_module,
        "setup_service_config",
        lambda path: (PROJECT_CONFIG, {**SERVICE_CONFIG, SC.IS_DOCKER_RUN: True}),
    )

    def stop(**kwargs):
        stop_calls.append(kwargs)
        running["value"] = False

    monkeypatch.setattr(poc_env_module, "_stop_poc", stop)
    # Exercise network removal itself; other Docker environment probes are unrelated here.
    monkeypatch.setattr(poc_env_module, "_docker_cli_env", lambda: {})
    network_calls = []

    def remove_network(command, **kwargs):
        network_calls.append(command)
        assert command == ["docker", "network", "rm", "nvflare-recipe-run"]
        if len(network_calls) == 1:
            if failure == "os_error":
                raise OSError("Docker unavailable")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, timeout=5)
            message = "network has active endpoints" if failure == "active_endpoints" else "Docker daemon unavailable"
            return SimpleNamespace(returncode=1, stdout="", stderr=message)
        return SimpleNamespace(returncode=0, stdout="nvflare-recipe-run\n", stderr="")

    monkeypatch.setattr(poc_env_module.subprocess, "run", remove_network)
    env.stop(clean_up=True)

    assert retained_result.read_text() == "keep me"
    assert env._docker_network_name == "nvflare-recipe-run"
    assert env._services_may_have_started is True
    assert env._session_manager is None
    assert f"preserving workspace {workspace}" in caplog.text
    assert "retry PocEnv.stop(clean_up=True)" in caplog.text

    env.stop(clean_up=True)

    assert not workspace.exists()
    assert env._docker_network_name is None
    assert env._services_may_have_started is False
    assert len(network_calls) == 2
    assert len(stop_calls) == int(initially_running)


@patch("nvflare.recipe.poc_env.get_poc_workspace")
@patch("nvflare.recipe.poc_env.get_prod_dir")
@patch("nvflare.recipe.poc_env.setup_service_config")
def test_get_admin_startup_kit_path(mock_setup, mock_get_prod_dir, mock_get_workspace):
    """Test getting admin startup kit path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_get_workspace.return_value = temp_dir
        prod_dir = os.path.join(temp_dir, "prod_00")
        mock_get_prod_dir.return_value = prod_dir
        mock_setup.return_value = ({"name": "test_project"}, {SC.FLARE_PROJ_ADMIN: "admin@nvidia.com"})
        env = PocEnv()
        admin_dir = os.path.join(prod_dir, "admin@nvidia.com")
        os.makedirs(admin_dir, exist_ok=True)

        assert env._get_admin_startup_kit_path() == admin_dir


@patch("nvflare.recipe.poc_env.get_poc_workspace")
@patch("nvflare.recipe.poc_env.get_prod_dir")
@patch("nvflare.recipe.poc_env.setup_service_config")
def test_get_admin_startup_kit_path_not_found(mock_setup, mock_get_prod_dir, mock_get_workspace):
    """Test getting admin startup kit path when directory doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_get_workspace.return_value = temp_dir
        prod_dir = os.path.join(temp_dir, "prod_00")
        mock_get_prod_dir.return_value = prod_dir
        mock_setup.return_value = ({"name": "test_project"}, {SC.FLARE_PROJ_ADMIN: "admin@nvidia.com"})
        env = PocEnv()

        with pytest.raises(RuntimeError, match="Admin startup kit not found"):
            env._get_admin_startup_kit_path()


@patch("nvflare.recipe.poc_env.setup_service_config")
@patch("nvflare.recipe.poc_env._stop_poc")
@patch("nvflare.recipe.poc_env.shutil.rmtree")
@patch("nvflare.recipe.poc_env.is_poc_running")
def test_stop_poc(mock_is_running, mock_remove_workspace, mock_stop_poc, mock_setup):
    """Test stop and clean POC functionality."""
    mock_setup.return_value = ({"name": "test"}, {SC.FLARE_SERVER: "server"})
    mock_is_running.return_value = True
    env = PocEnv()

    with patch("nvflare.tool.poc.poc_commands._clean_poc_config") as mock_clean_poc_config:
        with patch.object(PocEnv, "_running_services", side_effect=[["server"], []]):
            env.stop(clean_up=True)

    mock_stop_poc.assert_called_once_with(
        poc_workspace=env.poc_workspace,
        excluded=["admin@nvidia.com"],
        services_list=[],
    )
    mock_remove_workspace.assert_called_once_with(env.poc_workspace)
    mock_clean_poc_config.assert_not_called()


@patch("nvflare.recipe.poc_env.setup_service_config")
@patch("nvflare.recipe.poc_env._stop_poc")
@patch("nvflare.recipe.poc_env.shutil.rmtree")
@patch("nvflare.recipe.poc_env.is_poc_running")
def test_stop_preserves_workspace_when_service_state_is_unknown(
    mock_is_running, mock_remove_workspace, mock_stop_poc, mock_setup, caplog
):
    mock_setup.return_value = ({"name": "test"}, {SC.FLARE_SERVER: "server"})
    mock_is_running.return_value = True
    env = PocEnv()

    with patch.object(
        PocEnv,
        "_running_services",
        side_effect=[["server"], RuntimeError("Docker inspection unavailable")],
    ):
        env.stop(clean_up=True)

    mock_stop_poc.assert_called_once()
    mock_remove_workspace.assert_not_called()
    assert "Stop any remaining services and remove it manually" in caplog.text


def test_stop_with_unreadable_service_metadata_is_repeatable_and_attempts_shutdown(tmp_path, monkeypatch, caplog):
    import nvflare.recipe.poc_env as poc_env_module

    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    env = PocEnv()
    Path(env.poc_workspace).mkdir(parents=True)
    env._services_may_have_started = True
    env._session_manager = object()
    stop_calls = []
    monkeypatch.setattr(
        poc_env_module,
        "setup_service_config",
        lambda _workspace: (_ for _ in ()).throw(RuntimeError("service metadata unreadable")),
    )

    def stop(**kwargs):
        stop_calls.append(kwargs)
        raise RuntimeError("shutdown metadata unreadable")

    monkeypatch.setattr(poc_env_module, "_stop_poc", stop)

    env.stop(clean_up=True)
    env.stop(clean_up=True)

    assert len(stop_calls) == 2
    assert all(call["poc_workspace"] == env.poc_workspace for call in stop_calls)
    assert Path(env.poc_workspace).is_dir()
    assert env._services_may_have_started is True
    assert env._session_manager is None
    assert "preserving workspace" in caplog.text


def test_stop_is_idempotent_after_successful_deploy(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    env = PocEnv()
    runtime_running = {"value": False}
    stop_calls = []
    _configure_successful_deploy(monkeypatch, env)
    monkeypatch.setattr(
        poc_env_module,
        "_start_poc",
        lambda **kwargs: runtime_running.update(value=True),
    )
    monkeypatch.setattr(
        PocEnv,
        "_running_services",
        staticmethod(lambda *_args: ["server"] if runtime_running["value"] else []),
    )
    monkeypatch.setattr(poc_env_module, "is_poc_running", lambda *_args: True)

    def stop(**kwargs):
        stop_calls.append(kwargs)
        runtime_running["value"] = False

    monkeypatch.setattr(poc_env_module, "_stop_poc", stop)

    assert env.deploy(object()) == "job-id"
    env.stop(clean_up=True)
    env.stop(clean_up=True)

    assert len(stop_calls) == 1
    assert not Path(env.poc_workspace).exists()
    assert env._services_may_have_started is False


@patch("nvflare.recipe.poc_env.SessionManager")
@patch("nvflare.recipe.poc_env.setup_service_config")
@patch("nvflare.recipe.poc_env.get_prod_dir")
@patch("nvflare.recipe.poc_env.get_poc_workspace")
def test_poc_env_session_manager_passes_study(mock_get_workspace, mock_get_prod_dir, mock_setup, mock_session_manager):
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_get_workspace.return_value = temp_dir
        prod_dir = os.path.join(temp_dir, "prod_00")
        mock_get_prod_dir.return_value = prod_dir
        mock_setup.return_value = ({"name": "test_project"}, {SC.FLARE_PROJ_ADMIN: "admin@nvidia.com"})
        admin_dir = os.path.join(prod_dir, "admin@nvidia.com")
        os.makedirs(admin_dir, exist_ok=True)
        env = PocEnv(study="cancer-research")

        env._get_session_manager()

        session_params = mock_session_manager.call_args[0][0]
        assert session_params["study"] == "cancer-research"
