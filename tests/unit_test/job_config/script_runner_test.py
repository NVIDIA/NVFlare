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

        # Verify the runner stores default values
        assert runner._launch_once is True
        assert runner._shutdown_timeout == 0.0

    def test_subprocess_launcher_creation_with_custom_values(self, mock_file_system, base_script_runner_params):
        """Test that SubprocessLauncher is created with custom launch_once and shutdown_timeout."""
        runner = ScriptRunner(
            launch_external_process=True, launch_once=False, shutdown_timeout=20.0, **base_script_runner_params
        )

        # Verify the runner stores custom values
        assert runner._launch_once is False
        assert runner._shutdown_timeout == 20.0

    def test_exported_job_contains_launch_parameters(self, base_script_runner_params):
        """Test that exported job configuration contains launch_once and shutdown_timeout parameters."""
        from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
        from nvflare.job_config.api import FedJob

        # Create a temporary script file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as script_file:
            script_file.write("# Dummy training script for testing\n")
            script_path = script_file.name

        try:
            runner = ScriptRunner(
                script=script_path,
                script_args="--epochs 10",
                launch_external_process=True,
                launch_once=False,
                shutdown_timeout=25.0,
                framework=base_script_runner_params["framework"],
            )

            job = FedJob(name="test_launch_params_job")

            # Add a minimal server component (required for export)
            controller = ScatterAndGather(
                min_clients=1,
                num_rounds=1,
                wait_time_after_min_received=0,
            )
            job.to_server(controller)

            # Add the runner to clients
            job.to_clients(runner)

            # Export the job
            with tempfile.TemporaryDirectory() as temp_dir:
                job.export_job(temp_dir)

                # Check client config - use safer path construction
                client_config_path = os.path.join(
                    temp_dir, "test_launch_params_job", "app", "config", "config_fed_client.json"
                )

                assert os.path.exists(client_config_path), f"Client config not found at {client_config_path}"

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
        finally:
            # Clean up the temporary script file
            if os.path.exists(script_path):
                os.unlink(script_path)

    @pytest.mark.parametrize(
        "framework",
        [
            FrameworkType.PYTORCH,
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

    def test_custom_launcher_passed_through(self, mock_file_system, base_script_runner_params):
        """Test that providing a custom launcher through BaseScriptRunner works."""
        from nvflare.job_config.script_runner import BaseScriptRunner

        # Create a custom launcher with specific settings
        custom_launcher = SubprocessLauncher(script="custom_script.sh", launch_once=True, shutdown_timeout=5.0)

        # Use BaseScriptRunner which accepts a launcher parameter
        runner = BaseScriptRunner(
            launch_external_process=True,
            launcher=custom_launcher,  # Provide custom launcher
            launch_once=False,  # These should be stored but custom launcher takes precedence
            shutdown_timeout=100.0,
            **base_script_runner_params,
        )

        # The runner should store the parameters
        assert runner._launch_once is False
        assert runner._shutdown_timeout == 100.0
        assert runner._launcher is custom_launcher


class TestScriptRunnerMemoryManagement:
    """Test cases for ScriptRunner memory management parameters."""

    @pytest.fixture
    def base_script_runner_params(self):
        """Base parameters for creating ScriptRunner instances."""
        return {
            "script": "train.py",
            "script_args": "--epochs 10",
            "framework": FrameworkType.PYTORCH,
        }

    def test_default_memory_parameters(self, base_script_runner_params):
        """Test that memory management parameters default to disabled."""
        runner = ScriptRunner(**base_script_runner_params)

        assert runner._memory_gc_rounds == 0
        assert runner._torch_cuda_empty_cache is False

    @pytest.mark.parametrize(
        "memory_gc_rounds,torch_cuda_empty_cache",
        [
            (0, False),  # Disabled
            (1, True),  # Every round with cuda cache
            (5, False),  # Every 5 rounds without cuda cache
            (10, True),  # Every 10 rounds with cuda cache
        ],
    )
    def test_memory_parameter_configurations(self, base_script_runner_params, memory_gc_rounds, torch_cuda_empty_cache):
        """Test various memory management configurations."""
        runner = ScriptRunner(
            memory_gc_rounds=memory_gc_rounds,
            torch_cuda_empty_cache=torch_cuda_empty_cache,
            **base_script_runner_params,
        )

        assert runner._memory_gc_rounds == memory_gc_rounds
        assert runner._torch_cuda_empty_cache == torch_cuda_empty_cache

    def test_memory_parameters_with_external_process(self, base_script_runner_params):
        """Test memory parameters with external process mode."""
        runner = ScriptRunner(
            launch_external_process=True,
            memory_gc_rounds=3,
            torch_cuda_empty_cache=True,
            **base_script_runner_params,
        )

        assert runner._memory_gc_rounds == 3
        assert runner._torch_cuda_empty_cache is True
        assert runner._launch_external_process is True

    def test_memory_parameters_with_in_process(self, base_script_runner_params):
        """Test memory parameters with in-process mode."""
        runner = ScriptRunner(
            launch_external_process=False,
            memory_gc_rounds=2,
            torch_cuda_empty_cache=True,
            **base_script_runner_params,
        )

        assert runner._memory_gc_rounds == 2
        assert runner._torch_cuda_empty_cache is True
        assert runner._launch_external_process is False

    @pytest.mark.parametrize(
        "framework",
        [
            FrameworkType.PYTORCH,
            FrameworkType.NUMPY,
        ],
    )
    def test_memory_parameters_with_different_frameworks(self, framework):
        """Test that memory parameters work with different frameworks."""
        runner = ScriptRunner(
            script="train.py",
            memory_gc_rounds=1,
            torch_cuda_empty_cache=True,
            framework=framework,
        )

        assert runner._memory_gc_rounds == 1
        assert runner._torch_cuda_empty_cache is True
        assert runner._framework == framework
