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
from nvflare.recipe import secret_file_ref, secret_ref
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

    def test_get_client_config_value_resolves_nested_env_secret_refs(self, mock_fl_ctx, monkeypatch):
        fl_ctx, workspace = mock_fl_ctx
        monkeypatch.setenv("TEST_CLIENT_CONFIG_SECRET", "client-secret-value")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            workspace.get_app_config_dir.return_value = config_dir
            config_file = os.path.join(config_dir, JobConstants.CLIENT_JOB_CONFIG)
            placeholder = secret_ref("TEST_CLIENT_CONFIG_SECRET")
            with open(config_file, "w") as f:
                json.dump({"service": {"token": placeholder}}, f)

            assert get_client_config_value(fl_ctx, "service", resolve_refs=False) == {"token": placeholder}
            assert get_client_config_value(fl_ctx, "service") == {"token": "client-secret-value"}
            with open(config_file) as f:
                assert json.load(f)["service"]["token"] == placeholder

    def test_get_client_config_value_preserves_ordinary_placeholders(self, mock_fl_ctx, monkeypatch):
        fl_ctx, workspace = mock_fl_ctx
        monkeypatch.setenv("TEST_CLIENT_CONFIG_SECRET", "client-secret-value")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            workspace.get_app_config_dir.return_value = config_dir
            config_file = os.path.join(config_dir, JobConstants.CLIENT_JOB_CONFIG)
            with open(config_file, "w") as f:
                json.dump({"service": "{SITE_NAME}:${secret:TEST_CLIENT_CONFIG_SECRET}"}, f)

            assert get_client_config_value(fl_ctx, "service") == "{SITE_NAME}:client-secret-value"

    def test_get_client_config_value_missing_secret_ref_raises_instead_of_returning_default(
        self, mock_fl_ctx, monkeypatch
    ):
        fl_ctx, workspace = mock_fl_ctx
        monkeypatch.delenv("TEST_MISSING_CONFIG_SECRET", raising=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            workspace.get_app_config_dir.return_value = config_dir
            config_file = os.path.join(config_dir, JobConstants.CLIENT_JOB_CONFIG)
            with open(config_file, "w") as f:
                json.dump({"service": "${secret:TEST_MISSING_CONFIG_SECRET}"}, f)

            with pytest.raises(ValueError, match="TEST_MISSING_CONFIG_SECRET"):
                get_client_config_value(fl_ctx, "service", default="must-not-be-returned")

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

    def test_get_server_config_value_resolves_file_secret_ref(self, mock_fl_ctx):
        fl_ctx, workspace = mock_fl_ctx

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            workspace.get_app_config_dir.return_value = config_dir
            secret_file = os.path.join(tmpdir, "mounted-secret")
            with open(secret_file, "w") as f:
                f.write("server-secret-value")
            config_file = os.path.join(config_dir, JobConstants.SERVER_JOB_CONFIG)
            placeholder = secret_file_ref(secret_file)
            with open(config_file, "w") as f:
                json.dump({"service_password": placeholder}, f)

            assert get_server_config_value(fl_ctx, "service_password") == "server-secret-value"
            with open(config_file) as f:
                assert json.load(f)["service_password"] == placeholder

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
