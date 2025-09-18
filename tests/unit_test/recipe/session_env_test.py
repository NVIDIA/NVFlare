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

from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.recipe.session_env import SessionEnv


class _SessionEnv(SessionEnv):
    """Concrete implementation of SessionEnv for testing."""

    def __init__(self, session_params: Dict[str, any] = None, extra: dict = None):
        super().__init__(extra)
        self.session_params = session_params or {
            "username": "test@nvidia.com",
            "startup_kit_location": "/fake/path",
            "timeout": 10.0,
        }

    def _get_session_params(self) -> Dict[str, any]:
        return self.session_params

    def deploy(self, job):
        return "test_job_id"


class TestSessionEnvBaseFunctionality:
    """Test the base SessionEnv functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.env = _SessionEnv()

    @patch("nvflare.recipe.session_env.new_secure_session")
    def test_get_job_status(self, mock_session):
        """Test get_job_status method."""
        mock_sess = MagicMock()
        mock_sess.get_job_status.return_value = "RUNNING"
        mock_session.return_value = mock_sess

        result = self.env.get_job_status("job_123")

        assert result == "RUNNING"
        mock_session.assert_called_once_with(**self.env.session_params)
        mock_sess.get_job_status.assert_called_once_with("job_123")
        mock_sess.close.assert_called_once()

    @patch("nvflare.recipe.session_env.new_secure_session")
    def test_get_job_status_exception(self, mock_session):
        """Test get_job_status handles exceptions."""
        mock_session.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to get job status"):
            self.env.get_job_status("job_123")

    @patch("nvflare.recipe.session_env.new_secure_session")
    def test_abort_job(self, mock_session):
        """Test abort_job method."""
        mock_sess = MagicMock()
        mock_sess.abort_job.return_value = "Job aborted successfully"
        mock_session.return_value = mock_sess

        # Capture print output
        with patch("builtins.print") as mock_print:
            self.env.abort_job("job_123")

        mock_session.assert_called_once_with(**self.env.session_params)
        mock_sess.abort_job.assert_called_once_with("job_123")
        mock_sess.close.assert_called_once()
        mock_print.assert_called_with("Job job_123 aborted successfully with message: Job aborted successfully")

    @patch("nvflare.recipe.session_env.new_secure_session")
    def test_abort_job_exception(self, mock_session):
        """Test abort_job handles exceptions gracefully."""
        mock_session.side_effect = Exception("Connection failed")

        with patch("builtins.print") as mock_print:
            self.env.abort_job("job_123")

        mock_print.assert_called_with("Failed to abort job job_123: Connection failed")

    @patch("nvflare.recipe.session_env.new_secure_session")
    def test_get_job_result_finished(self, mock_session):
        """Test get_job_result when job finishes successfully."""
        mock_sess = MagicMock()
        mock_sess.monitor_job.return_value = MonitorReturnCode.JOB_FINISHED
        mock_sess.download_job_result.return_value = "/path/to/result"
        mock_session.return_value = mock_sess

        with patch("builtins.print"):
            result = self.env.get_job_result("job_123", timeout=30.0)

        assert result == "/path/to/result"
        mock_sess.monitor_job.assert_called_once()
        mock_sess.download_job_result.assert_called_once_with("job_123")
        mock_sess.close.assert_called_once()

    @patch("nvflare.recipe.session_env.new_secure_session")
    def test_get_job_result_timeout(self, mock_session):
        """Test get_job_result when timeout occurs."""
        mock_sess = MagicMock()
        mock_sess.monitor_job.return_value = MonitorReturnCode.TIMEOUT
        mock_session.return_value = mock_sess

        with patch("builtins.print") as mock_print:
            result = self.env.get_job_result("job_123", timeout=30.0)

        assert result is None
        mock_print.assert_any_call(
            "Job job_123 did not complete within 30.0 seconds. Job is still running. Try calling get_result() again with a longer timeout."
        )

    @patch("nvflare.recipe.session_env.new_secure_session")
    def test_get_job_result_ended_by_callback(self, mock_session):
        """Test get_job_result when ended by callback."""
        mock_sess = MagicMock()
        mock_sess.monitor_job.return_value = MonitorReturnCode.ENDED_BY_CB
        mock_session.return_value = mock_sess

        with patch("builtins.print") as mock_print:
            result = self.env.get_job_result("job_123")

        assert result is None
        mock_print.assert_any_call(
            "Job monitoring was stopped early by callback. Result may not be available yet. Check job status and try again."
        )

    @patch("nvflare.recipe.session_env.new_secure_session")
    def test_get_job_result_unexpected_code(self, mock_session):
        """Test get_job_result with unexpected return code."""
        mock_sess = MagicMock()
        mock_sess.monitor_job.return_value = "UNKNOWN_CODE"
        mock_session.return_value = mock_sess

        with pytest.raises(RuntimeError, match="Unexpected monitor return code"):
            with patch("builtins.print"):
                self.env.get_job_result("job_123")

    @patch("nvflare.recipe.session_env.new_secure_session")
    def test_get_job_result_exception(self, mock_session):
        """Test get_job_result handles exceptions."""
        mock_session.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to get job result"):
            self.env.get_job_result("job_123")

    def test_session_params_delegation(self):
        """Test that session params are properly delegated to subclasses."""
        custom_params = {"username": "custom@user.com", "startup_kit_location": "/custom/path", "timeout": 15.0}
        env = _SessionEnv(session_params=custom_params)

        assert env._get_session_params() == custom_params


def test_session_env_is_abstract():
    """Test that SessionEnv cannot be instantiated directly."""
    from nvflare.recipe.session_env import SessionEnv

    with pytest.raises(TypeError):
        SessionEnv()
