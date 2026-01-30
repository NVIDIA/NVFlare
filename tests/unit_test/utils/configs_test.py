# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import JobConstants
from nvflare.utils.configs import get_client_config_value, get_server_config_value


class TestConfigUtils:
    """Test cases for nvflare.utils.configs utility functions."""

    @pytest.fixture
    def mock_fl_ctx(self):
        """Create a mock FLContext with workspace setup."""
        fl_ctx = Mock()
        engine = Mock()
        workspace = Mock()

        fl_ctx.get_engine.return_value = engine
        engine.get_workspace.return_value = workspace
        fl_ctx.get_job_id.return_value = "test_job_id"

        return fl_ctx, workspace

    def test_get_client_config_value_success(self, mock_fl_ctx):
        """Test successfully reading a value from config_fed_client.json."""
        fl_ctx, workspace = mock_fl_ctx

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            workspace.get_app_config_dir.return_value = config_dir

            # Create a config file with test data
            config_file = os.path.join(config_dir, JobConstants.CLIENT_JOB_CONFIG)
            test_config = {"EXTERNAL_PRE_INIT_TIMEOUT": 900.0}
            with open(config_file, "w") as f:
                json.dump(test_config, f)

            # Test reading existing key
            assert get_client_config_value(fl_ctx, "EXTERNAL_PRE_INIT_TIMEOUT") == 900.0

    def test_get_client_config_value_missing_key(self, mock_fl_ctx):
        """Test reading a non-existent key returns default value."""
        fl_ctx, workspace = mock_fl_ctx

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            workspace.get_app_config_dir.return_value = config_dir

            # Create a config file
            config_file = os.path.join(config_dir, JobConstants.CLIENT_JOB_CONFIG)
            with open(config_file, "w") as f:
                json.dump({"existing_key": "value"}, f)

            # Test reading missing key with default
            assert get_client_config_value(fl_ctx, "missing_key") is None
            assert get_client_config_value(fl_ctx, "missing_key", default=123) == 123

    def test_get_client_config_value_missing_file(self, mock_fl_ctx):
        """Test reading when config file doesn't exist returns default."""
        fl_ctx, workspace = mock_fl_ctx

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            workspace.get_app_config_dir.return_value = config_dir

            # Don't create the config file
            assert get_client_config_value(fl_ctx, "any_key") is None
            assert get_client_config_value(fl_ctx, "any_key", default=456) == 456

    def test_get_server_config_value_success(self, mock_fl_ctx):
        """Test successfully reading a value from config_fed_server.json."""
        fl_ctx, workspace = mock_fl_ctx

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            workspace.get_app_config_dir.return_value = config_dir

            # Create a config file with test data
            config_file = os.path.join(config_dir, JobConstants.SERVER_JOB_CONFIG)
            test_config = {"custom_param": "custom_value", "timeout": 600}
            with open(config_file, "w") as f:
                json.dump(test_config, f)

            # Test reading existing keys
            assert get_server_config_value(fl_ctx, "custom_param") == "custom_value"
            assert get_server_config_value(fl_ctx, "timeout") == 600

    def test_get_server_config_value_missing_key(self, mock_fl_ctx):
        """Test reading a non-existent key returns default value."""
        fl_ctx, workspace = mock_fl_ctx

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            workspace.get_app_config_dir.return_value = config_dir

            # Create a config file
            config_file = os.path.join(config_dir, JobConstants.SERVER_JOB_CONFIG)
            with open(config_file, "w") as f:
                json.dump({"existing_key": "value"}, f)

            # Test reading missing key with default
            assert get_server_config_value(fl_ctx, "missing_key") is None
            assert get_server_config_value(fl_ctx, "missing_key", default="default_value") == "default_value"

    def test_get_server_config_value_missing_file(self, mock_fl_ctx):
        """Test reading when config file doesn't exist returns default."""
        fl_ctx, workspace = mock_fl_ctx

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            workspace.get_app_config_dir.return_value = config_dir

            # Don't create the config file
            assert get_server_config_value(fl_ctx, "any_key") is None
            assert get_server_config_value(fl_ctx, "any_key", default=789) == 789
