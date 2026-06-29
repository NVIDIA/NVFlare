# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import importlib.util
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.app_common.default_component_policy import DEFAULT_CLASS_ALLOW_LIST
from nvflare.app_common.widgets.component_path_authorizer import CLASS_ALLOW_LIST, ComponentPathAuthorizer
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.job_config.api import FedJob
from nvflare.recipe.sim_env import SimEnv


class _ResourcesWorkspace:
    def __init__(self, resources_file):
        self.resources_file = resources_file

    def get_resources_file_path(self):
        return self.resources_file


def _make_job(name="test_job"):
    return FedJob(name=name)


def _deploy_with_mocked_simulator(env, job):
    with patch("nvflare.recipe.sim_env.collect_non_local_scripts", return_value=[]):
        with patch.object(job, "simulator_run", return_value=0) as mock_run:
            env.deploy(job)
    return mock_run


def _default_resources_path(workspace_root, job_name):
    return workspace_root / job_name / WorkspaceConstants.SITE_FOLDER_NAME / WorkspaceConstants.DEFAULT_RESOURCES_CONFIG


def _mock_simulator_popen():
    process = MagicMock()
    process.wait.return_value = 0
    return patch("nvflare.job_config.fed_job_config.subprocess.Popen", return_value=process)


def test_sim_env_validation():
    # Test with valid inputs
    env = SimEnv(num_clients=3, clients=["client1", "client2", "client3"])
    assert env.num_clients == 3
    assert env.clients == ["client1", "client2", "client3"]

    # Test with inconsistent number of clients
    with pytest.raises(ValueError, match="Inconsistent number of clients"):
        SimEnv(num_clients=2, clients=["client1", "client2", "client3"])

    # Test with no clients specified (invalid)
    with pytest.raises(ValueError, match="Either 'num_clients' must be > 0 or 'clients' list must be provided"):
        SimEnv()

    # Test with empty clients list and zero num_clients (invalid)
    with pytest.raises(ValueError, match="Either 'num_clients' must be > 0 or 'clients' list must be provided"):
        SimEnv(num_clients=0, clients=[])

    # BUG-3 regression: when clients list is provided and num_clients=0,
    # env should derive client/thread counts from the list.
    env = SimEnv(num_clients=0, clients=["client1", "client2", "client3"])
    assert env.num_clients == 3
    assert env.num_threads == 3


def test_sim_env_deploy_with_explicit_clients_does_not_pass_n_clients(tmp_path):
    """SimEnv.deploy() must not pass n_clients when clients are explicit."""
    job = _make_job()
    env = SimEnv(clients=["site-1", "site-2"], workspace_root=str(tmp_path))

    mock_run = _deploy_with_mocked_simulator(env, job)

    _, kwargs = mock_run.call_args
    assert kwargs.get("n_clients") is None
    assert kwargs.get("clients") == ["site-1", "site-2"]
    assert not env.last_run_failed


def test_sim_env_deploy_raises_on_failed_simulation(tmp_path):
    job = _make_job()
    env = SimEnv(num_clients=2, workspace_root=str(tmp_path))

    with patch("nvflare.recipe.sim_env.collect_non_local_scripts", return_value=[]):
        with patch.object(job, "simulator_run", return_value=2):
            with pytest.raises(RuntimeError, match="Simulation failed with return code 2"):
                env.deploy(job)

    assert env.last_run_failed


def test_sim_env_bootstraps_standard_component_policy(tmp_path):
    job = _make_job("standard-policy")
    env = SimEnv(num_clients=2, workspace_root=str(tmp_path))

    _deploy_with_mocked_simulator(env, job)

    resources = json.loads(_default_resources_path(tmp_path, job.name).read_text())
    assert resources == {"format_version": 2, CLASS_ALLOW_LIST: list(DEFAULT_CLASS_ALLOW_LIST)}
    assert len(resources[CLASS_ALLOW_LIST]) == 81
    assert all(not path.endswith(".") for path in resources[CLASS_ALLOW_LIST])


def test_sim_env_adds_standard_policy_to_existing_resources(tmp_path):
    job = _make_job("missing-policy")
    resources_file = _default_resources_path(tmp_path, job.name)
    resources_file.parent.mkdir(parents=True)
    resources_file.write_text(json.dumps({"format_version": 2, "preserved": True}))
    env = SimEnv(num_clients=2, workspace_root=str(tmp_path))

    _deploy_with_mocked_simulator(env, job)

    resources = json.loads(resources_file.read_text())
    assert resources["preserved"] is True
    assert resources[CLASS_ALLOW_LIST] == list(DEFAULT_CLASS_ALLOW_LIST)


@pytest.mark.parametrize("policy", [[], ["example.custom.ExactComponent"]])
def test_sim_env_preserves_explicit_component_policy(tmp_path, policy):
    job = _make_job("explicit-policy")
    resources_file = tmp_path / job.name / WorkspaceConstants.SITE_FOLDER_NAME / WorkspaceConstants.RESOURCES_CONFIG
    resources_file.parent.mkdir(parents=True)
    original = {"format_version": 2, "preserved": True, CLASS_ALLOW_LIST: policy}
    resources_file.write_text(json.dumps(original))
    env = SimEnv(num_clients=2, workspace_root=str(tmp_path))

    _deploy_with_mocked_simulator(env, job)

    assert json.loads(resources_file.read_text()) == original
    assert not _default_resources_path(tmp_path, job.name).exists()


def test_sim_env_invalid_explicit_policy_remains_fail_closed(tmp_path):
    job = _make_job("invalid-policy")
    resources_file = _default_resources_path(tmp_path, job.name)
    resources_file.parent.mkdir(parents=True)
    resources_file.write_text(json.dumps({CLASS_ALLOW_LIST: "invalid"}))
    env = SimEnv(num_clients=2, workspace_root=str(tmp_path))

    _deploy_with_mocked_simulator(env, job)

    with pytest.raises(UnsafeComponentError, match="class_allow_list must be list"):
        ComponentPathAuthorizer().authorize_component_config(
            {"path": DEFAULT_CLASS_ALLOW_LIST[0]}, workspace=_ResourcesWorkspace(str(resources_file))
        )


def test_sim_env_rejects_invalid_resources_file(tmp_path):
    job = _make_job("invalid-resources")
    resources_file = _default_resources_path(tmp_path, job.name)
    resources_file.parent.mkdir(parents=True)
    resources_file.write_text("[]")
    env = SimEnv(num_clients=2, workspace_root=str(tmp_path))

    with patch("nvflare.recipe.sim_env.collect_non_local_scripts", return_value=[]):
        with patch.object(job, "simulator_run", return_value=0) as mock_run:
            with pytest.raises(ValueError, match="must contain a JSON object"):
                env.deploy(job)

    mock_run.assert_not_called()


def test_sim_env_byoc_job_does_not_expand_standard_policy(tmp_path):
    module_path = tmp_path / "simenv_custom_executor.py"
    module_path.write_text(
        "from nvflare.apis.executor import Executor\n\n"
        "class CustomExecutor(Executor):\n"
        "    def execute(self, task_name, shareable, fl_ctx, abort_signal):\n"
        "        return shareable\n"
    )
    spec = importlib.util.spec_from_file_location("simenv_custom_executor", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    try:
        job = FedJob(name="byoc-policy", min_clients=1)
        job.to_server(ScatterAndGather(min_clients=1, num_rounds=1))
        job.to_clients(module.CustomExecutor())
        export_root = tmp_path / "exported"
        job.export_job(str(export_root))
        assert any((entry / "custom").is_dir() for entry in (export_root / job.name).iterdir() if entry.is_dir())

        workspace_root = tmp_path / "simulation"
        env = SimEnv(num_clients=1, workspace_root=str(workspace_root))
        with _mock_simulator_popen() as popen:
            env.deploy(job)

        assert popen.called
        resources = json.loads(_default_resources_path(workspace_root, job.name).read_text())
        assert resources[CLASS_ALLOW_LIST] == list(DEFAULT_CLASS_ALLOW_LIST)
        assert f"{module.CustomExecutor.__module__}.{module.CustomExecutor.__name__}" not in resources[CLASS_ALLOW_LIST]
    finally:
        sys.modules.pop(spec.name, None)
