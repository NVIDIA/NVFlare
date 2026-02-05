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

import pytest

from nvflare.recipe.run import Run


class TestRunClass:
    """Test the Run class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_env = MagicMock()
        self.job_id = "test_job_123"
        self.run = Run(exec_env=self.mock_env, job_id=self.job_id)

    def test_initialization(self):
        """Test Run initialization."""
        assert self.run.exec_env == self.mock_env
        assert self.run.job_id == self.job_id
        assert self.run._stopped is False
        assert self.run._cached_status is None
        assert self.run._cached_result is None
        assert self.run.logger is not None

    def test_get_job_id(self):
        """Test get_job_id method."""
        assert self.run.get_job_id() == self.job_id

    def test_get_status_delegates_to_env(self):
        """Test that get_status delegates to exec_env when not stopped."""
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

    def test_get_result_waits_caches_and_stops(self):
        """Test that get_result waits for job, caches status, and stops env."""
        self.mock_env.get_job_result.return_value = {"model": "weights"}
        self.mock_env.get_job_status.return_value = "FINISHED"

        result = self.run.get_result(timeout=30.0)

        assert result == {"model": "weights"}
        self.mock_env.get_job_result.assert_called_once_with(self.job_id, timeout=30.0)
        self.mock_env.get_job_status.assert_called_once_with(self.job_id)
        self.mock_env.stop.assert_called_once_with(clean_up=True)
        assert self.run._stopped is True
        assert self.run._cached_status == "FINISHED"
        assert self.run._cached_result == {"model": "weights"}

    def test_get_result_default_timeout(self):
        """Test get_result with default timeout."""
        self.mock_env.get_job_result.return_value = {"data": "result"}
        self.mock_env.get_job_status.return_value = "FINISHED"

        result = self.run.get_result()

        assert result == {"data": "result"}
        self.mock_env.get_job_result.assert_called_once_with(self.job_id, timeout=0.0)

    def test_get_status_returns_cached_after_stopped(self):
        """Test that get_status returns cached value after get_result is called."""
        self.mock_env.get_job_result.return_value = {"data": "result"}
        self.mock_env.get_job_status.return_value = "FINISHED"

        # Call get_result first - this stops POC and caches status
        self.run.get_result()

        # Reset mock to verify get_status doesn't call exec_env again
        self.mock_env.get_job_status.reset_mock()

        # get_status should return cached value
        status = self.run.get_status()
        assert status == "FINISHED"
        self.mock_env.get_job_status.assert_not_called()

    def test_get_result_returns_cached_after_stopped(self):
        """Test that get_result returns cached value when called again."""
        self.mock_env.get_job_result.return_value = {"data": "result"}
        self.mock_env.get_job_status.return_value = "FINISHED"

        # First call
        result1 = self.run.get_result()

        # Reset mocks
        self.mock_env.get_job_result.reset_mock()
        self.mock_env.stop.reset_mock()

        # Second call should return cached result
        result2 = self.run.get_result()
        assert result2 == {"data": "result"}
        self.mock_env.get_job_result.assert_not_called()
        self.mock_env.stop.assert_not_called()

    def test_get_result_stops_even_on_exception(self):
        """Test that stop is called even if get_job_result raises exception."""
        self.mock_env.get_job_result.side_effect = RuntimeError("Connection failed")

        result = self.run.get_result()

        # Exception is caught, result is None
        assert result is None
        self.mock_env.stop.assert_called_once_with(clean_up=True)
        assert self.run._stopped is True
        assert self.run._cached_status is None
        assert self.run._cached_result is None

    def test_get_result_caches_none_on_status_exception(self):
        """Test that caches are set to None if get_job_status raises exception."""
        self.mock_env.get_job_result.return_value = {"data": "result"}
        self.mock_env.get_job_status.side_effect = Exception("Status error")

        result = self.run.get_result()

        # Result is still returned, but exception causes caches to be set to None
        assert result == {"data": "result"}
        assert self.run._cached_status is None
        assert self.run._cached_result is None
        assert self.run._stopped is True

    def test_abort_delegates_to_env(self):
        """Test that abort delegates to exec_env when not stopped."""
        self.run.abort()

        self.mock_env.abort_job.assert_called_once_with(self.job_id)

    def test_abort_does_nothing_after_stopped(self):
        """Test that abort does nothing after get_result has been called."""
        self.mock_env.get_job_result.return_value = {"data": "result"}
        self.mock_env.get_job_status.return_value = "FINISHED"

        # Stop via get_result
        self.run.get_result()

        # Reset mock
        self.mock_env.abort_job.reset_mock()

        # Abort should not call exec_env.abort_job
        self.run.abort()
        self.mock_env.abort_job.assert_not_called()

    def test_get_status_before_get_result_does_not_stop(self):
        """Test that get_status does not stop POC."""
        self.mock_env.get_job_status.return_value = "RUNNING"

        # Call get_status multiple times
        self.run.get_status()
        self.run.get_status()
        self.run.get_status()

        # stop should never be called
        self.mock_env.stop.assert_not_called()
        assert self.run._stopped is False


@pytest.mark.skip(reason="Integration tests require full environment setup")
class TestRunIntegration:
    """Integration tests for Run with actual environment classes."""

    def test_run_with_sim_env(self):
        """Test Run with actual SimEnv."""
        from nvflare.recipe import SimEnv

        sim_env = SimEnv(num_clients=2, workspace_root="/tmp/test_sim")
        run = Run(exec_env=sim_env, job_id="test_job")

        # Simulation environment should return None for status
        assert run.get_status() is None

        # Should return workspace path for results
        result = run.get_result()
        assert result == "/tmp/test_sim/test_job"

    def test_run_with_poc_env(self):
        """Test Run with actual PocEnv (mocked dependencies)."""
        from nvflare.recipe import PocEnv

        with patch("nvflare.recipe.poc_env.get_poc_workspace", return_value="/tmp/poc"):
            poc_env = PocEnv(num_clients=2)
            run = Run(exec_env=poc_env, job_id="poc_test_job")

            # Test that methods delegate properly (actual session calls would be mocked in real tests)
            with patch.object(poc_env, "get_job_status", return_value="RUNNING") as mock_status:
                assert run.get_status() == "RUNNING"
                mock_status.assert_called_once_with("poc_test_job")

    def test_run_with_prod_env(self):
        """Test Run with actual ProdEnv (mocked dependencies)."""
        from nvflare.recipe import ProdEnv

        with tempfile.TemporaryDirectory() as temp_dir:
            prod_env = ProdEnv(startup_kit_location=temp_dir)
            run = Run(exec_env=prod_env, job_id="prod_test_job")

            # Test that methods delegate properly (actual session calls would be mocked in real tests)
            with patch.object(prod_env, "abort_job") as mock_abort:
                run.abort()
                mock_abort.assert_called_once_with("prod_test_job")
