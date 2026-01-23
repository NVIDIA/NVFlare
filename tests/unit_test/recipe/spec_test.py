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

"""Tests for recipe utilities and ExecEnv script validation."""

import json
import logging
import os
import tempfile

import pytest

from nvflare.job_config.api import FedApp, FedJob
from nvflare.job_config.fed_app_config import ClientAppConfig
from nvflare.recipe.utils import _collect_non_local_scripts


class TestCollectNonLocalScriptsUtility:
    """Test the _collect_non_local_scripts utility function."""

    def setup_method(self):
        self.job = FedJob(name="test_job", min_clients=1)
        self.client_app = FedApp(ClientAppConfig())
        self.job._deploy_map["site-1"] = self.client_app

    def test_no_scripts_returns_empty_list(self):
        """Test with no scripts added."""
        result = _collect_non_local_scripts(self.job)
        assert result == []

    def test_local_script_not_included(self):
        """Test that local scripts are not included in the result."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"# test script")
            temp_path = f.name

        try:
            self.client_app.add_external_script(temp_path)
            result = _collect_non_local_scripts(self.job)
            assert result == []
        finally:
            os.unlink(temp_path)

    def test_non_local_absolute_path_included(self):
        """Test that non-local absolute paths are included."""
        non_local_script = "/preinstalled/remote_script.py"
        self.client_app.add_external_script(non_local_script)

        result = _collect_non_local_scripts(self.job)
        assert non_local_script in result

    def test_multiple_non_local_scripts(self):
        """Test with multiple non-local scripts."""
        scripts = [
            "/preinstalled/script1.py",
            "/preinstalled/script2.py",
            "/preinstalled/script3.py",
        ]
        for script in scripts:
            self.client_app.add_external_script(script)

        result = _collect_non_local_scripts(self.job)
        for script in scripts:
            assert script in result

    def test_mixed_local_and_non_local_scripts(self):
        """Test with mix of local and non-local scripts."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"# test script")
            local_script = f.name

        try:
            self.client_app.add_external_script(local_script)

            non_local_script = "/preinstalled/remote_script.py"
            self.client_app.add_external_script(non_local_script)

            result = _collect_non_local_scripts(self.job)
            assert non_local_script in result
            assert local_script not in result
        finally:
            os.unlink(local_script)

    def test_multiple_apps(self):
        """Test collection across multiple apps in deploy_map."""
        client_app2 = FedApp(ClientAppConfig())
        self.job._deploy_map["site-2"] = client_app2

        script1 = "/preinstalled/script1.py"
        script2 = "/preinstalled/script2.py"
        self.client_app.add_external_script(script1)
        client_app2.add_external_script(script2)

        result = _collect_non_local_scripts(self.job)
        assert script1 in result
        assert script2 in result


class TestSimEnvValidation:
    """Test SimEnv script validation in deploy()."""

    def test_deploy_with_non_local_script_raises_error(self):
        """Test that SimEnv.deploy() raises error for non-local scripts."""
        from nvflare.recipe.sim_env import SimEnv

        job = FedJob(name="test_job", min_clients=1)
        client_app = FedApp(ClientAppConfig())
        job._deploy_map["site-1"] = client_app

        non_local_script = "/preinstalled/remote_script.py"
        client_app.add_external_script(non_local_script)

        env = SimEnv(num_clients=2)
        with pytest.raises(ValueError, match="scripts do not exist locally"):
            env.deploy(job)


class TestProdEnvValidation:
    """Test ProdEnv script validation in deploy()."""

    def test_deploy_with_non_local_script_logs_warning(self, caplog):
        """Test that ProdEnv.deploy() logs warning for non-local scripts."""
        from unittest.mock import MagicMock, patch

        from nvflare.recipe.prod_env import ProdEnv

        job = FedJob(name="test_job", min_clients=1)
        client_app = FedApp(ClientAppConfig())
        job._deploy_map["site-1"] = client_app

        non_local_script = "/preinstalled/remote_script.py"
        client_app.add_external_script(non_local_script)

        # Mock the startup kit location and session manager
        with patch("os.path.exists", return_value=True):
            env = ProdEnv(startup_kit_location="/fake/startup_kit")

        # Mock the session manager to avoid actual deployment
        env._session_manager = MagicMock()
        env._session_manager.submit_job.return_value = "test-job-id"

        with caplog.at_level(logging.WARNING):
            env.deploy(job)

        assert "not found locally" in caplog.text
        assert "pre-installed on the production system" in caplog.text


class TestRecipeConfigMethods:
    """Test add_server_config and add_client_config methods in Recipe."""

    @pytest.fixture
    def temp_script(self):
        """Create a temporary script file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Test training script\nimport nvflare.client as flare\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_add_server_config(self, temp_script):
        """Test add_server_config adds params to server app."""
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_job",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
        )

        config = {"np_download_chunk_size": 2097152}
        recipe.add_server_config(config)

        server_app = recipe.job._deploy_map.get("server")
        assert server_app is not None
        assert server_app.app_config.additional_params == config

    def test_add_client_config(self, temp_script):
        """Test add_client_config applies to all clients and specific clients."""
        from nvflare.apis.job_def import ALL_SITES
        from nvflare.recipe.fedavg import FedAvgRecipe

        # Test all clients
        recipe = FedAvgRecipe(
            name="test_job",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
        )
        config = {"timeout": 600}
        recipe.add_client_config(config)

        all_clients_app = recipe.job._deploy_map.get(ALL_SITES)
        assert all_clients_app is not None
        assert all_clients_app.app_config.additional_params == config

    def test_config_in_generated_json(self, temp_script):
        """Test that configs appear in generated JSON files."""
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_config_gen",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
        )

        recipe.add_server_config({"server_param": 123})
        recipe.add_client_config({"client_param": 456})

        with tempfile.TemporaryDirectory() as tmpdir:
            recipe.job.export_job(tmpdir)
            job_dir = os.path.join(tmpdir, "test_config_gen")

            # Verify server config
            with open(os.path.join(job_dir, "app", "config", "config_fed_server.json")) as f:
                server_config = json.load(f)
            assert server_config.get("server_param") == 123

            # Verify client config
            with open(os.path.join(job_dir, "app", "config", "config_fed_client.json")) as f:
                client_config = json.load(f)
            assert client_config.get("client_param") == 456

    def test_config_type_error(self, temp_script):
        """Test TypeError is raised for non-dict arguments."""
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_job",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
        )

        with pytest.raises(TypeError, match="config must be a dict"):
            recipe.add_server_config("not_a_dict")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="config must be a dict"):
            recipe.add_client_config(123)  # type: ignore[arg-type]
