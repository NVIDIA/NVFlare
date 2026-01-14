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

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner


class TestScriptRunner:
    """Test cases for ScriptRunner class with launch_once and shutdown_timeout parameters."""

    @pytest.fixture
    def mock_file_system(self):
        """Mock file system operations."""
        with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
            yield

    @pytest.fixture
    def base_script_runner_params(self):
        """Base parameters for creating ScriptRunner instances."""
        return {
            "script": "train.py",
            "script_args": "--epochs 10",
            "framework": FrameworkType.PYTORCH,
        }

    def test_default_launch_once_and_shutdown_timeout(self, base_script_runner_params):
        """Test that launch_once defaults to True and shutdown_timeout defaults to 0.0."""
        runner = ScriptRunner(**base_script_runner_params)

        assert runner._launch_once is True
        assert runner._shutdown_timeout == 0.0

    @pytest.mark.parametrize(
        "launch_once,shutdown_timeout",
        [
            (True, 0.0),  # Default values
            (False, 0.0),  # launch_once=False with default timeout
            (True, 10.0),  # launch_once=True with custom timeout
            (False, 15.0),  # launch_once=False with custom timeout
            (True, 100.0),  # Large timeout value
        ],
    )
    def test_launch_once_and_shutdown_timeout_configurations(
        self, base_script_runner_params, launch_once, shutdown_timeout
    ):
        """Test various launch_once and shutdown_timeout configurations."""
        runner = ScriptRunner(
            launch_external_process=True,
            launch_once=launch_once,
            shutdown_timeout=shutdown_timeout,
            **base_script_runner_params,
        )

        assert runner._launch_once == launch_once
        assert runner._shutdown_timeout == shutdown_timeout
        assert runner._launch_external_process is True

    def test_in_process_with_launch_parameters(self, base_script_runner_params):
        """Test that launch parameters can be set even in in-process mode."""
        runner = ScriptRunner(
            launch_external_process=False,  # In-process mode
            launch_once=False,
            shutdown_timeout=10.0,
            **base_script_runner_params,
        )

        # Values should be stored even though they won't be used in in-process mode
        assert runner._launch_once is False
        assert runner._shutdown_timeout == 10.0
        assert runner._launch_external_process is False

    def test_subprocess_launcher_creation_with_default_values(self, mock_file_system, base_script_runner_params):
        """Test that SubprocessLauncher is created with default launch_once and shutdown_timeout."""
        runner = ScriptRunner(launch_external_process=True, **base_script_runner_params)

        job = FedJob(name="test_job")
        ctx = job.clients.set_defaults()
        comp_ids = runner.add_to_fed_job(job, ctx)

        # Verify launcher was created
        assert "launcher_id" in comp_ids
        launcher_id = comp_ids["launcher_id"]

        # Get the launcher component
        launcher = job._client_config_data["components"][launcher_id]
        assert launcher is not None
        assert isinstance(launcher, SubprocessLauncher)

        # Check default values
        assert launcher._launch_once is True
        assert launcher._shutdown_timeout == 0.0

    def test_subprocess_launcher_creation_with_custom_values(self, mock_file_system, base_script_runner_params):
        """Test that SubprocessLauncher is created with custom launch_once and shutdown_timeout."""
        runner = ScriptRunner(
            launch_external_process=True, launch_once=False, shutdown_timeout=20.0, **base_script_runner_params
        )

        job = FedJob(name="test_job")
        ctx = job.clients.set_defaults()
        comp_ids = runner.add_to_fed_job(job, ctx)

        # Verify launcher was created
        assert "launcher_id" in comp_ids
        launcher_id = comp_ids["launcher_id"]

        # Get the launcher component
        launcher = job._client_config_data["components"][launcher_id]
        assert launcher is not None
        assert isinstance(launcher, SubprocessLauncher)

        # Check custom values were passed
        assert launcher._launch_once is False
        assert launcher._shutdown_timeout == 20.0

    def test_exported_job_contains_launch_parameters(self, mock_file_system, base_script_runner_params):
        """Test that exported job configuration contains launch_once and shutdown_timeout parameters."""
        runner = ScriptRunner(
            launch_external_process=True, launch_once=False, shutdown_timeout=25.0, **base_script_runner_params
        )

        job = FedJob(name="test_launch_params_job")
        ctx = job.clients.set_defaults()
        runner.add_to_fed_job(job, ctx)

        # Export the job
        with tempfile.TemporaryDirectory() as temp_dir:
            job.export_job(temp_dir)

            # Check client config
            client_config_path = os.path.join(
                temp_dir, "test_launch_params_job", "app", "config", "config_fed_client.json"
            )
            assert os.path.exists(client_config_path)

            with open(client_config_path, "r") as f:
                client_config = json.load(f)

            # Find the launcher component
            launcher_component = None
            for component in client_config["components"]:
                if component["id"] == "launcher":
                    launcher_component = component
                    break

            assert launcher_component is not None
            assert "args" in launcher_component

            # Verify the launch parameters are in the exported config
            launcher_args = launcher_component["args"]
            assert "launch_once" in launcher_args
            assert launcher_args["launch_once"] is False
            assert "shutdown_timeout" in launcher_args
            assert launcher_args["shutdown_timeout"] == 25.0

    @pytest.mark.parametrize(
        "framework",
        [
            FrameworkType.PYTORCH,
            FrameworkType.TENSORFLOW,
            FrameworkType.NUMPY,
            FrameworkType.RAW,
        ],
    )
    def test_launch_parameters_with_different_frameworks(self, mock_file_system, framework):
        """Test that launch parameters work correctly with different frameworks."""
        runner = ScriptRunner(
            script="train.py",
            launch_external_process=True,
            launch_once=False,
            shutdown_timeout=30.0,
            framework=framework,
        )

        assert runner._launch_once is False
        assert runner._shutdown_timeout == 30.0
        assert runner._framework == framework

    def test_custom_launcher_not_overridden(self, mock_file_system, base_script_runner_params):
        """Test that providing a custom launcher doesn't get overridden by launch parameters."""
        # Create a custom launcher with specific settings
        custom_launcher = SubprocessLauncher(script="custom_script.sh", launch_once=True, shutdown_timeout=5.0)

        runner = ScriptRunner(
            launch_external_process=True,
            launcher=custom_launcher,  # Provide custom launcher
            launch_once=False,  # These should be ignored since we provide custom launcher
            shutdown_timeout=100.0,
            **base_script_runner_params,
        )

        job = FedJob(name="test_custom_launcher_job")
        ctx = job.clients.set_defaults()
        comp_ids = runner.add_to_fed_job(job, ctx)

        # Get the launcher component
        launcher_id = comp_ids["launcher_id"]
        launcher = job._client_config_data["components"][launcher_id]

        # Should be the custom launcher, not a newly created one
        assert launcher is custom_launcher
        assert launcher._launch_once is True  # Custom launcher's value
        assert launcher._shutdown_timeout == 5.0  # Custom launcher's value
