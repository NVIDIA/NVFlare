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

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from nvflare.recipe.poc_env import POCEnv


def test_poc_env_initialization():
    """Test POCEnv initialization with default values."""
    env = POCEnv()
    assert env.num_clients == 2
    assert env.gpu_ids == []
    assert env.auto_stop is True
    assert env.auto_clean is True
    assert env.monitor_duration == 0
    assert env._poc_started_by_us is False


def test_poc_env_initialization_with_custom_values():
    """Test POCEnv initialization with custom values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env = POCEnv(
            poc_workspace=temp_dir,
            num_clients=3,
            gpu_ids=[0, 1],
            auto_stop=False,
            auto_clean=False,
            monitor_duration=30,
        )
        assert env.poc_workspace == temp_dir
        assert env.num_clients == 3
        assert env.gpu_ids == [0, 1]
        assert env.auto_stop is False
        assert env.auto_clean is False
        assert env.monitor_duration == 30


@patch("nvflare.recipe.poc_env.is_poc_running")
def test_is_running_success(mock_is_poc_running):
    """Test is_running method when POC is running."""
    mock_is_poc_running.return_value = True

    with patch("nvflare.recipe.poc_env.setup_service_config") as mock_setup:
        mock_setup.return_value = ({"name": "test"}, {"server": "server"})

        env = POCEnv()
        assert env.is_running() is True

        mock_setup.assert_called_once_with(env.poc_workspace)
        mock_is_poc_running.assert_called_once()


@patch("nvflare.recipe.poc_env.is_poc_running")
def test_is_running_failure(mock_is_poc_running):
    """Test is_running method when setup fails."""
    with patch("nvflare.recipe.poc_env.setup_service_config") as mock_setup:
        mock_setup.side_effect = Exception("Setup failed")

        env = POCEnv()
        assert env.is_running() is False

        mock_setup.assert_called_once_with(env.poc_workspace)
        mock_is_poc_running.assert_not_called()


@patch("nvflare.recipe.poc_env._clean_poc")
def test_clean_workspace(mock_clean_poc):
    """Test manual workspace cleaning."""
    env = POCEnv()
    env.clean_workspace()

    mock_clean_poc.assert_called_once_with(env.poc_workspace)


@patch("nvflare.recipe.poc_env._start_poc")
def test_start_services_when_not_running(mock_start_poc):
    """Test starting services when POC is not running."""
    env = POCEnv()

    with patch.object(env, "is_running", return_value=False):
        env.start_services()

        mock_start_poc.assert_called_once_with(
            poc_workspace=env.poc_workspace, gpu_ids=[], excluded=["admin@nvidia.com"], services_list=[]
        )
        assert env._poc_started_by_us is True


@patch("nvflare.recipe.poc_env._start_poc")
def test_start_services_when_already_running(mock_start_poc):
    """Test starting services when POC is already running."""
    env = POCEnv()

    with patch.object(env, "is_running", return_value=True):
        env.start_services()

        mock_start_poc.assert_not_called()
        assert env._poc_started_by_us is False


@patch("nvflare.recipe.poc_env._stop_poc")
def test_stop_services_when_running(mock_stop_poc):
    """Test stopping services when POC is running."""
    env = POCEnv()

    with patch.object(env, "is_running", return_value=True):
        env.stop_services()

        mock_stop_poc.assert_called_once_with(
            poc_workspace=env.poc_workspace, excluded=["admin@nvidia.com"], services_list=[]
        )


@patch("nvflare.recipe.poc_env._stop_poc")
def test_stop_services_when_not_running(mock_stop_poc):
    """Test stopping services when POC is not running."""
    env = POCEnv()

    with patch.object(env, "is_running", return_value=False):
        env.stop_services()

        mock_stop_poc.assert_not_called()


@patch("nvflare.recipe.poc_env.prepare_poc_provision")
@patch("nvflare.recipe.poc_env.setup_service_config")
@patch("nvflare.recipe.poc_env.validate_poc_workspace")
def test_prepare_poc_workspace(mock_validate, mock_setup, mock_prepare):
    """Test POC workspace preparation."""
    mock_setup.return_value = ({"name": "test"}, {"server": "server"})

    env = POCEnv(num_clients=3)
    env._prepare_poc_workspace()

    mock_prepare.assert_called_once_with(
        clients=["site-1", "site-2", "site-3"],
        number_of_clients=3,
        workspace=env.poc_workspace,
        docker_image=None,
        use_he=False,
        project_conf_path="",
        examples_dir=None,
    )


@patch("nvflare.recipe.poc_env.new_secure_session")
@patch("nvflare.recipe.poc_env.setup_service_config")
def test_get_admin_startup_kit_path(mock_setup, mock_session):
    """Test getting admin startup kit path."""
    mock_setup.return_value = ({"name": "test_project"}, {"server": "server"})

    with tempfile.TemporaryDirectory() as temp_dir:
        env = POCEnv(poc_workspace=temp_dir)

        # Create the expected admin directory structure
        admin_dir = os.path.join(temp_dir, "test_project", "prod_00", "admin@nvidia.com")
        os.makedirs(admin_dir, exist_ok=True)

        result = env._get_admin_startup_kit_path()
        assert result == admin_dir


@patch("nvflare.recipe.poc_env.new_secure_session")
@patch("nvflare.recipe.poc_env.setup_service_config")
def test_get_admin_startup_kit_path_not_found(mock_setup, mock_session):
    """Test getting admin startup kit path when directory doesn't exist."""
    mock_setup.return_value = ({"name": "test_project"}, {"server": "server"})

    with tempfile.TemporaryDirectory() as temp_dir:
        env = POCEnv(poc_workspace=temp_dir)

        with pytest.raises(RuntimeError, match="Admin startup kit not found"):
            env._get_admin_startup_kit_path()


@patch("nvflare.recipe.poc_env.new_secure_session")
def test_submit_job_via_api(mock_session):
    """Test job submission via Flare API."""
    mock_sess = MagicMock()
    mock_sess.submit_job.return_value = "job_12345"
    mock_session.return_value = mock_sess

    env = POCEnv()

    with patch.object(env, "_get_admin_startup_kit_path", return_value="/fake/admin/dir"):
        result = env._submit_job_via_api("/fake/job/path")

        assert result == "job_12345"
        mock_session.assert_called_once_with(username="admin@nvidia.com", startup_kit_location="/fake/admin/dir")
        mock_sess.submit_job.assert_called_once_with("/fake/job/path")
        mock_sess.close.assert_called_once()


@patch("nvflare.recipe.poc_env.new_secure_session")
def test_monitor_job(mock_session):
    """Test job monitoring."""
    mock_sess = MagicMock()
    mock_sess.monitor_job.return_value = "completed"
    mock_session.return_value = mock_sess

    env = POCEnv()

    with patch.object(env, "_get_admin_startup_kit_path", return_value="/fake/admin/dir"):
        env._monitor_job("job_12345", 30)

        mock_session.assert_called_once_with(username="admin@nvidia.com", startup_kit_location="/fake/admin/dir")
        mock_sess.monitor_job.assert_called_once_with("job_12345", timeout=30)
        mock_sess.close.assert_called_once()


@patch("nvflare.recipe.poc_env.tempfile.TemporaryDirectory")
def test_deploy_job_integration(mock_temp_dir):
    """Test complete job deployment flow."""
    # Mock temporary directory
    mock_temp_dir_obj = MagicMock()
    mock_temp_dir_obj.__enter__.return_value = "/tmp/temp_dir"
    mock_temp_dir_obj.__exit__.return_value = None
    mock_temp_dir.return_value = mock_temp_dir_obj

    # Mock job
    mock_job = MagicMock()
    mock_job.name = "test_job"
    mock_job.export_job = MagicMock()

    env = POCEnv(monitor_duration=5)

    with (
        patch.object(env, "is_running", return_value=True),
        patch.object(env, "_submit_job_via_api", return_value="job_12345"),
        patch.object(env, "_monitor_job") as mock_monitor,
    ):

        result = env.deploy(mock_job)

        assert result == "job_12345"
        mock_job.export_job.assert_called_once_with("/tmp/temp_dir")
        env._submit_job_via_api.assert_called_once_with("/tmp/temp_dir/test_job")
        mock_monitor.assert_called_once_with("job_12345", 5)


@patch("nvflare.recipe.poc_env.tempfile.TemporaryDirectory")
def test_deploy_job_with_auto_stop_and_clean(mock_temp_dir):
    """Test job deployment with auto stop and clean enabled."""
    # Mock temporary directory
    mock_temp_dir_obj = MagicMock()
    mock_temp_dir_obj.__enter__.return_value = "/tmp/temp_dir"
    mock_temp_dir_obj.__exit__.return_value = None
    mock_temp_dir.return_value = mock_temp_dir_obj

    # Mock job
    mock_job = MagicMock()
    mock_job.name = "test_job"
    mock_job.export_job = MagicMock()

    env = POCEnv(auto_stop=True, auto_clean=True)
    env._poc_started_by_us = True  # Simulate that we started the services

    # Mock is_running to return True initially (POC already running), then False after stop
    is_running_side_effect = [True, False]  # First call: True, subsequent calls: False

    with (
        patch.object(env, "is_running", side_effect=is_running_side_effect),
        patch.object(env, "_submit_job_via_api", return_value="job_12345"),
        patch.object(env, "_monitor_job"),
        patch.object(env, "_stop_poc_services") as mock_stop,
        patch.object(env, "_clean_poc_workspace") as mock_clean,
    ):

        result = env.deploy(mock_job)

        assert result == "job_12345"
        mock_stop.assert_called_once()
        mock_clean.assert_called_once()
