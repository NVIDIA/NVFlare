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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nvflare.recipe.poc_env import PocEnv, _recipe_runtime_lock_path
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

PROJECT_CONFIG = {"name": "poc"}
SERVICE_CONFIG = {SC.FLARE_SERVER: "server", SC.FLARE_CLIENTS: ["site-1"]}


@pytest.fixture(autouse=True)
def _isolated_recipe_runtime_lock(tmp_path, monkeypatch):
    """Keep process-held Recipe POC locks independent across unit tests."""
    import nvflare.recipe.poc_env as poc_env_module

    monkeypatch.setattr(poc_env_module, "_recipe_runtime_lock_path", lambda: str(tmp_path / "recipe-poc.lock"))


def _configure_successful_deploy(monkeypatch, env, prepare=None, submit=None):
    import nvflare.recipe.poc_env as poc_env_module

    if prepare is None:

        def prepare(**kwargs):
            Path(kwargs["workspace"]).mkdir(parents=True)

    if submit is None:

        def submit(job):
            return "job-id"

    monkeypatch.setattr(poc_env_module, "collect_non_local_scripts", lambda job: [])
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


def test_recipe_runtime_lock_path_is_host_and_user_scoped():
    assert _recipe_runtime_lock_path() == os.path.join(
        tempfile.gettempdir(), f".nvflare-recipe-poc-{os.geteuid()}.lock"
    )


def test_runtime_lock_rejects_unsafe_lock_file(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    env = PocEnv()
    monkeypatch.setattr(poc_env_module.stat, "S_ISREG", lambda mode: False)

    with pytest.raises(RuntimeError, match="unsafe Recipe POC runtime lock"):
        env._acquire_runtime_lock()

    assert env._runtime_lock_file is None


def test_runtime_lock_propagates_unexpected_lock_error(monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    env = PocEnv()
    monkeypatch.setattr(
        poc_env_module.fcntl,
        "flock",
        lambda fd, operation: (_ for _ in ()).throw(OSError("lock failed")),
    )

    with pytest.raises(OSError, match="lock failed"):
        env._acquire_runtime_lock()

    assert env._runtime_lock_file is None
    env._release_runtime_lock()


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
    assert env._runtime_lock_file is None


def test_deploy_rejects_another_active_recipe_environment(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    configured_workspace = tmp_path / "poc"
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(configured_workspace))
    first = PocEnv()
    second = PocEnv()
    provisioned_workspaces = []

    def prepare(**kwargs):
        provisioned_workspaces.append(kwargs["workspace"])
        Path(kwargs["workspace"]).mkdir(parents=True)

    _configure_successful_deploy(monkeypatch, first, prepare=prepare)
    assert first.deploy(object()) == "job-id"
    held_lock = first._runtime_lock_file
    first._acquire_runtime_lock()
    assert first._runtime_lock_file is held_lock

    _configure_successful_deploy(monkeypatch, second, prepare=prepare)
    with pytest.raises(RuntimeError, match="Another Recipe PocEnv deployment is active"):
        second.deploy(object())

    assert provisioned_workspaces == [first.poc_workspace]

    first.stop(clean_up=False)
    assert second.deploy(object()) == "job-id"
    assert provisioned_workspaces == [first.poc_workspace, second.poc_workspace]
    second.stop(clean_up=False)


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
    monkeypatch.setattr(env, "stop", lambda clean_up: None)
    monkeypatch.setattr(env, "_check_poc_running", lambda: True)
    run_workspace = env.poc_workspace

    with pytest.raises(RuntimeError, match="cleanup could not be completed safely") as exc_info:
        env.deploy(object())

    assert "submission failed" in str(exc_info.value)
    assert run_workspace in str(exc_info.value)
    assert "remove this workspace manually" in str(exc_info.value)
    assert "POC services remain running" in str(exc_info.value.__cause__)
    assert os.path.isdir(run_workspace)


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


def test_reusing_stopped_env_creates_new_workspace_and_preserves_prior_result(tmp_path, monkeypatch):
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
    first_workspace = env.poc_workspace
    env.stop(clean_up=False)
    assert (Path(first_workspace) / "run-result.txt").read_text() == "complete"

    assert env.deploy(object()) == "job-id"
    second_workspace = env.poc_workspace

    assert second_workspace != first_workspace
    assert Path(second_workspace).is_dir()
    assert (Path(first_workspace) / "run-result.txt").read_text() == "complete"


def test_redeploy_rejects_running_workspace_without_rotating(tmp_path, monkeypatch):
    import nvflare.recipe.poc_env as poc_env_module

    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    env = PocEnv()
    _configure_successful_deploy(monkeypatch, env)
    assert env.deploy(object()) == "job-id"
    active_workspace = env.poc_workspace
    monkeypatch.setattr(env, "_check_poc_running", lambda: True)

    with pytest.raises(RuntimeError, match="already has a running deployment"):
        env.deploy(object())

    assert env.poc_workspace == active_workspace


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
@patch("nvflare.recipe.poc_env._clean_poc")
@patch("nvflare.recipe.poc_env.is_poc_running")
def test_stop_poc(mock_is_running, mock_clean_poc, mock_stop_poc, mock_setup):
    """Test stop and clean POC functionality."""
    mock_setup.return_value = ({"name": "test"}, {SC.FLARE_SERVER: "server"})
    mock_is_running.return_value = True
    env = PocEnv()

    with patch.object(PocEnv, "_running_services", side_effect=[["server"], []]):
        env.stop(clean_up=True)

    mock_stop_poc.assert_called_once_with(
        poc_workspace=env.poc_workspace,
        excluded=["admin@nvidia.com"],
        services_list=[],
    )
    mock_clean_poc.assert_called_once_with(env.poc_workspace)


@patch("nvflare.recipe.poc_env.setup_service_config")
@patch("nvflare.recipe.poc_env._stop_poc")
@patch("nvflare.recipe.poc_env._clean_poc")
@patch("nvflare.recipe.poc_env.is_poc_running")
def test_stop_preserves_workspace_when_service_state_is_unknown(
    mock_is_running, mock_clean_poc, mock_stop_poc, mock_setup, caplog
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
    mock_clean_poc.assert_not_called()
    assert "Stop any remaining services and remove it manually" in caplog.text


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
