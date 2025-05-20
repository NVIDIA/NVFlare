#!/usr/bin/env python3

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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.app_opt.lightning.loggers.client_logger import ClientLogger


class TestClientLogger:
    @pytest.fixture(autouse=True)  # This will run for all tests
    def setup_default_context(self):
        with patch("nvflare.client.api.default_context", MagicMock()) as mock_context:
            yield mock_context

    @pytest.fixture
    def mock_mlflow_writer(self):
        with patch("nvflare.app_opt.lightning.loggers.client_logger.MLflowWriter") as mock:
            yield mock.return_value

    @pytest.fixture
    def mock_obj_logger(self):
        with patch("nvflare.app_opt.lightning.loggers.client_logger.get_obj_logger") as mock:
            yield mock.return_value

    @pytest.fixture
    def logger(self, mock_mlflow_writer, mock_obj_logger):
        return ClientLogger(prefix="test")

    def test_init(self, logger):
        """Test logger initialization."""
        assert logger._prefix == "test"
        assert logger.LOGGER_JOIN_CHAR == "-"
        assert logger.name == "nvflare"

    def test_log_hyperparams(self, logger, mock_obj_logger):
        """Test hyperparameter logging warning."""
        params = {"learning_rate": 0.01}
        logger.log_hyperparams(params)
        mock_obj_logger.warning.assert_called_once_with("log_hyperparams is not supported.")

    def test_log_metrics(self, logger, mock_mlflow_writer):
        """Test metric logging with prefix."""
        metrics = {"accuracy": 0.95, "loss": 0.1}
        step = 1

        logger.log_metrics(metrics, step)

        # Verify metrics are prefixed correctly
        expected_metrics = {"test-accuracy": 0.95, "test-loss": 0.1}
        mock_mlflow_writer.log_metrics.assert_called_once_with(metrics=expected_metrics, step=step)

    def test_log_metrics_no_prefix(self, mock_mlflow_writer):
        """Test metric logging without prefix."""
        logger = ClientLogger()  # No prefix
        metrics = {"accuracy": 0.95}
        step = 1

        logger.log_metrics(metrics, step)

        # Verify metrics are not prefixed
        mock_mlflow_writer.log_metrics.assert_called_once_with(metrics=metrics, step=step)

    def test_log_metrics_non_zero_rank(self, mock_mlflow_writer, mock_obj_logger):
        """Test metric logging from non-zero rank raises error."""
        logger = ClientLogger()
        metrics = {"accuracy": 0.95}

        with patch("nvflare.app_opt.lightning.loggers.client_logger.rank_zero_only") as mock:
            mock.rank = 1
            with pytest.raises(AssertionError, match="experiment tried to log from global_rank != 0"):
                logger.log_metrics(metrics)

    def test_version(self, logger):
        """Test version property returns NVFLARE version."""
        import nvflare

        assert logger.version == nvflare.__version__
