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

import tempfile
from unittest.mock import MagicMock, patch

from nvflare.recipe import PocEnv, ProdEnv, Run, SimEnv


class TestRunClass:
    """Test the refactored Run class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_env = MagicMock()
        self.job_id = "test_job_123"
        self.run = Run(exec_env=self.mock_env, job_id=self.job_id)

    def test_initialization(self):
        """Test Run initialization."""
        assert self.run.exec_env == self.mock_env
        assert self.run.job_id == self.job_id

    def test_get_job_id(self):
        """Test get_job_id method."""
        assert self.run.get_job_id() == self.job_id

    def test_get_status_delegates_to_env(self):
        """Test that get_status delegates to exec_env."""
        self.mock_env.get_job_status.return_value = "RUNNING"

        result = self.run.get_status()

        assert result == "RUNNING"
        self.mock_env.get_job_status.assert_called_once_with(self.job_id)

    def test_get_status_returns_none_for_sim(self):
        """Test that get_status can return None (e.g., for simulation)."""
        self.mock_env.get_job_status.return_value = None

        result = self.run.get_status()

        assert result is None
        self.mock_env.get_job_status.assert_called_once_with(self.job_id)

    def test_get_result_delegates_to_env(self):
        """Test that get_result delegates to exec_env."""
        self.mock_env.get_job_result.return_value = "/path/to/result"

        result = self.run.get_result(timeout=30.0)

        assert result == "/path/to/result"
        self.mock_env.get_job_result.assert_called_once_with(self.job_id, timeout=30.0)

    def test_get_result_default_timeout(self):
        """Test get_result with default timeout."""
        self.mock_env.get_job_result.return_value = "/path/to/result"

        result = self.run.get_result()

        assert result == "/path/to/result"
        self.mock_env.get_job_result.assert_called_once_with(self.job_id, timeout=0.0)

    def test_abort_delegates_to_env(self):
        """Test that abort delegates to exec_env."""
        self.run.abort()

        self.mock_env.abort_job.assert_called_once_with(self.job_id)

    def test_run_with_different_environments(self):
        """Test Run works with different environment types."""
        # Test with simulation environment (returns None for status)
        sim_env = MagicMock()
        sim_env.get_job_status.return_value = None
        sim_env.get_job_result.return_value = "/sim/workspace/job_123"

        sim_run = Run(exec_env=sim_env, job_id="sim_job")
        assert sim_run.get_status() is None
        assert sim_run.get_result() == "/sim/workspace/job_123"

        # Test with production environment
        prod_env = MagicMock()
        prod_env.get_job_status.return_value = "COMPLETED"
        prod_env.get_job_result.return_value = "/prod/downloads/job_result"

        prod_run = Run(exec_env=prod_env, job_id="prod_job")
        assert prod_run.get_status() == "COMPLETED"
        assert prod_run.get_result() == "/prod/downloads/job_result"


class TestRunIntegration:
    """Integration tests for Run with actual environment classes."""

    def test_run_with_sim_env(self):
        """Test Run with actual SimEnv."""
        sim_env = SimEnv(num_clients=2, workspace_root="/tmp/test_sim")
        run = Run(exec_env=sim_env, job_id="test_job")

        # Simulation environment should return None for status
        assert run.get_status() is None

        # Should return workspace path for results
        result = run.get_result()
        assert result == "/tmp/test_sim/test_job"

    def test_run_with_poc_env(self):
        """Test Run with actual PocEnv (mocked dependencies)."""

        with patch("nvflare.recipe.poc_env.get_poc_workspace", return_value="/tmp/poc"):
            poc_env = PocEnv(num_clients=2)
            run = Run(exec_env=poc_env, job_id="poc_test_job")

            # Test that methods delegate properly (actual session calls would be mocked in real tests)
            with patch.object(poc_env, "get_job_status", return_value="RUNNING") as mock_status:
                assert run.get_status() == "RUNNING"
                mock_status.assert_called_once_with("poc_test_job")

    def test_run_with_prod_env(self):
        """Test Run with actual ProdEnv (mocked dependencies)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prod_env = ProdEnv(startup_kit_location=temp_dir)
            run = Run(exec_env=prod_env, job_id="prod_test_job")

            # Test that methods delegate properly (actual session calls would be mocked in real tests)
            with patch.object(prod_env, "abort_job") as mock_abort:
                run.abort()
                mock_abort.assert_called_once_with("prod_test_job")
