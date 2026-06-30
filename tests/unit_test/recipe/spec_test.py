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

import importlib
import json
import logging
import os
import tempfile
from decimal import Decimal

import pytest

from nvflare.apis.job_def import JobMetaKey
from nvflare.job_config.api import FedApp, FedJob
from nvflare.job_config.fed_app_config import ClientAppConfig
from nvflare.recipe.utils import collect_non_local_scripts, set_per_site_config, set_recipe_meta


class TestCollectNonLocalScriptsUtility:
    """Test the collect_non_local_scripts utility function."""

    def setup_method(self):
        self.job = FedJob(name="test_job", min_clients=1)
        self.client_app = FedApp(ClientAppConfig())
        self.job._deploy_map["site-1"] = self.client_app

    def test_no_scripts_returns_empty_list(self):
        """Test with no scripts added."""
        result = collect_non_local_scripts(self.job)
        assert result == []

    def test_local_script_not_included(self):
        """Test that local scripts are not included in the result."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"# test script")
            temp_path = f.name

        try:
            self.client_app.add_external_script(temp_path)
            result = collect_non_local_scripts(self.job)
            assert result == []
        finally:
            os.unlink(temp_path)

    def test_non_local_absolute_path_included(self):
        """Test that non-local absolute paths are included."""
        non_local_script = "/preinstalled/remote_script.py"
        self.client_app.add_external_script(non_local_script)

        result = collect_non_local_scripts(self.job)
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

        result = collect_non_local_scripts(self.job)
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

            result = collect_non_local_scripts(self.job)
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

        result = collect_non_local_scripts(self.job)
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
            model={"class_path": "model.DummyModel", "args": {}},
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
            model={"class_path": "model.DummyModel", "args": {}},
        )
        config = {"timeout": 600}
        recipe.add_client_config(config)

        all_clients_app = recipe.job._deploy_map.get(ALL_SITES)
        assert all_clients_app is not None
        assert all_clients_app.app_config.additional_params == config

    def test_add_client_file_adds_to_ext_scripts_and_ext_dirs(self, temp_script):
        """Test add_client_file stores file paths in ext_scripts and dirs in ext_dirs."""
        from nvflare.apis.job_def import ALL_SITES
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_job_files",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
            model={"class_path": "model.DummyModel", "args": {}},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recipe.add_client_file(temp_script)
            recipe.add_client_file(temp_dir)

            all_clients_app = recipe.job._deploy_map.get(ALL_SITES)
            assert all_clients_app is not None
            assert temp_script in all_clients_app.app_config.ext_scripts
            assert temp_dir in all_clients_app.app_config.ext_dirs

    def test_add_client_file_preserves_per_site_clients_without_all_sites(self, temp_script):
        """Test add_client_file keeps per-site topology and does not create ALL_SITES app."""
        from nvflare.apis.job_def import ALL_SITES
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_job_per_site_files",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
            model={"class_path": "model.DummyModel", "args": {}},
            per_site_config={"site-1": {}, "site-2": {}},
        )

        recipe.add_client_file(temp_script)

        assert ALL_SITES not in recipe.job._deploy_map
        site_1_app = recipe.job._deploy_map.get("site-1")
        site_2_app = recipe.job._deploy_map.get("site-2")
        assert site_1_app is not None
        assert site_2_app is not None
        assert temp_script in site_1_app.app_config.ext_scripts
        assert temp_script in site_2_app.app_config.ext_scripts

    def test_add_client_file_with_specific_clients_only_updates_selected_sites(self, temp_script):
        """Test add_client_file(..., clients=[...]) only adds file to specified sites."""
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_job_targeted_files",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
            model={"class_path": "model.DummyModel", "args": {}},
            per_site_config={"site-1": {}, "site-2": {}, "site-3": {}},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("targeted file for site-2 only")
            targeted_file = f.name

        recipe.add_client_file(targeted_file, clients=["site-2"])

        try:
            site_1_app = recipe.job._deploy_map.get("site-1")
            site_2_app = recipe.job._deploy_map.get("site-2")
            site_3_app = recipe.job._deploy_map.get("site-3")
            assert site_1_app is not None
            assert site_2_app is not None
            assert site_3_app is not None
            assert targeted_file not in site_1_app.app_config.ext_scripts
            assert targeted_file in site_2_app.app_config.ext_scripts
            assert targeted_file not in site_3_app.app_config.ext_scripts
        finally:
            os.unlink(targeted_file)

    def test_add_server_file_adds_to_server_ext_scripts_and_ext_dirs(self, temp_script):
        """Test add_server_file stores file paths in ext_scripts and dirs in ext_dirs."""
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_job_server_files",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
            model={"class_path": "model.DummyModel", "args": {}},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recipe.add_server_file(temp_script)
            recipe.add_server_file(temp_dir)

            server_app = recipe.job._deploy_map.get("server")
            assert server_app is not None
            assert temp_script in server_app.app_config.ext_scripts
            assert temp_dir in server_app.app_config.ext_dirs

    def test_config_in_generated_json(self, temp_script):
        """Test that configs appear in generated JSON files."""
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_config_gen",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
            model={"class_path": "model.DummyModel", "args": {}},
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
            model={"class_path": "model.DummyModel", "args": {}},
        )

        with pytest.raises(TypeError, match="config must be a dict"):
            recipe.add_server_config("not_a_dict")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="config must be a dict"):
            recipe.add_client_config(123)  # type: ignore[arg-type]

    def test_add_decomposers_type_error(self, temp_script):
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_job",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
            model={"class_path": "model.DummyModel", "args": {}},
        )

        with pytest.raises(TypeError, match="decomposer must be str or Decomposer"):
            recipe.add_decomposers([object()])  # type: ignore[list-item]

    def test_add_decomposers_uses_distinct_registers_for_server_and_clients(self, temp_script, monkeypatch):
        import nvflare.recipe.spec as spec_module
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_job",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
            model={"class_path": "model.DummyModel", "args": {}},
        )

        captured = {}

        def _capture_server(obj, **kwargs):
            captured["server_obj"] = obj
            captured["server_kwargs"] = kwargs

        def _capture_clients(obj, **kwargs):
            captured["client_obj"] = obj
            captured["client_kwargs"] = kwargs

        class _DummyRegister:
            def __init__(self, class_names):
                self.class_names = class_names

        monkeypatch.setattr(spec_module, "DecomposerRegister", _DummyRegister)
        monkeypatch.setattr(recipe.job, "to_server", _capture_server)
        monkeypatch.setattr(recipe, "_add_to_client_apps", _capture_clients)

        recipe.add_decomposers(["pkg.mod.Dec"])

        assert captured["server_kwargs"] == {"id": "decomposer_reg"}
        assert captured["client_kwargs"] == {"id": "decomposer_reg"}
        assert captured["server_obj"] is not captured["client_obj"]


class TestRecipeMetaHelper:
    """Test generic helper-provided recipe metadata."""

    def test_set_recipe_meta_sets_recognized_top_level_meta_props(self, tmp_path):
        from nvflare.recipe.spec import Recipe

        class BasicRecipe(Recipe):
            def __init__(self):
                job = FedJob(name="test_recipe_meta", min_clients=1, meta_props={"owner": "alice"})
                job.to_server({"server_arg": True})
                super().__init__(job)

        recipe = BasicRecipe()
        resource_spec = {
            "site-1": {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 4},
            "site-2": {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 2},
        }
        launcher_spec = {
            "site-1": {"docker": {"image": "nvflare-site1:latest"}},
            "site-2": {"docker": {"image": "nvflare-site2:latest"}},
        }
        mandatory_clients = ["site-1", "site-2"]

        set_recipe_meta(recipe, JobMetaKey.STUDY, "default")
        set_recipe_meta(recipe, JobMetaKey.RESOURCE_SPEC, resource_spec)
        set_recipe_meta(recipe, JobMetaKey.JOB_LAUNCHER_SPEC, launcher_spec)
        set_recipe_meta(recipe, JobMetaKey.MANDATORY_CLIENTS, mandatory_clients)
        set_recipe_meta(recipe, JobMetaKey.SCOPE, "private")
        set_recipe_meta(recipe, JobMetaKey.CUSTOM_PROPS, {"team": "research"})

        job_config = recipe.job.job
        assert job_config.min_clients == 1
        assert job_config.mandatory_clients is None
        assert job_config.resource_specs == {}
        assert job_config.meta_props == {
            "owner": "alice",
            "study": "default",
            "resource_spec": resource_spec,
            "launcher_spec": launcher_spec,
            "mandatory_clients": mandatory_clients,
            "scope": "private",
            "custom_props": {"team": "research"},
        }

        recipe.job.export_job(str(tmp_path))
        with open(tmp_path / "test_recipe_meta" / "meta.json") as f:
            exported_meta = json.load(f)

        assert exported_meta["min_clients"] == 1
        assert exported_meta["mandatory_clients"] == mandatory_clients
        assert exported_meta["resource_spec"] == resource_spec
        assert exported_meta["launcher_spec"] == launcher_spec
        assert exported_meta["scope"] == "private"
        assert exported_meta["study"] == "default"
        assert exported_meta["custom_props"] == {"team": "research"}
        assert exported_meta["owner"] == "alice"

    def test_set_recipe_meta_stores_values_that_differ_from_fed_job_config(self, tmp_path):
        from nvflare.recipe.spec import Recipe

        class BasicRecipe(Recipe):
            def __init__(self):
                job = FedJob(name="test_recipe_meta_conflict", min_clients=2)
                job.job.resource_specs = {"base-site": {"num_of_gpus": 0}}
                job.to_server({"server_arg": True})
                super().__init__(job)

        recipe = BasicRecipe()
        resource_spec = {"site-1": {"num_of_gpus": 1}}

        set_recipe_meta(recipe, JobMetaKey.MIN_CLIENTS, 3)
        set_recipe_meta(recipe, JobMetaKey.RESOURCE_SPEC, resource_spec)

        job_config = recipe.job.job
        assert job_config.min_clients == 2
        assert job_config.resource_specs == {"base-site": {"num_of_gpus": 0}}
        assert job_config.meta_props["min_clients"] == 3
        assert job_config.meta_props["resource_spec"] == resource_spec

        recipe.job.export_job(str(tmp_path))
        with open(tmp_path / "test_recipe_meta_conflict" / "meta.json") as f:
            exported_meta = json.load(f)

        assert exported_meta["min_clients"] == 3
        assert exported_meta["resource_spec"] == resource_spec

    def test_set_recipe_meta_replaces_different_value_from_existing_meta_props(self):
        from nvflare.recipe.spec import Recipe

        class BasicRecipe(Recipe):
            def __init__(self):
                super().__init__(
                    FedJob(name="test_recipe_meta_props_conflict", min_clients=1, meta_props={"study": "a"})
                )

        recipe = BasicRecipe()
        original_meta_props = recipe.job.job.meta_props
        set_recipe_meta(recipe, JobMetaKey.STUDY, "b")

        assert recipe.job.job.meta_props is original_meta_props
        assert recipe.job.job.meta_props["study"] == "b"

    def test_set_recipe_meta_allows_existing_meta_props_that_differ_from_generated_values(self):
        from nvflare.recipe.spec import Recipe

        class BasicRecipe(Recipe):
            def __init__(self):
                job = FedJob(name="test_recipe_meta_generated_conflict", min_clients=2, meta_props={"min_clients": 5})
                super().__init__(job)

        recipe = BasicRecipe()
        set_recipe_meta(recipe, JobMetaKey.STUDY, "default")

        assert recipe.job.job.meta_props["min_clients"] == 5
        assert recipe.job.job.meta_props["study"] == "default"

    @pytest.mark.parametrize("key", [JobMetaKey.DEPLOY_MAP, JobMetaKey.JOB_NAME])
    def test_set_recipe_meta_rejects_non_user_settable_keys(self, key):
        from nvflare.recipe.spec import Recipe

        class BasicRecipe(Recipe):
            def __init__(self):
                job = FedJob(name="test_recipe_meta_restricted_key", min_clients=1)
                job.to_server({"server_arg": True})
                super().__init__(job)

        with pytest.raises(ValueError, match=rf"{key.value}.*cannot be set through set_recipe_meta"):
            set_recipe_meta(BasicRecipe(), key, "value")

    def test_set_recipe_meta_deep_copies_user_values(self):
        from nvflare.recipe.spec import Recipe

        class BasicRecipe(Recipe):
            def __init__(self):
                super().__init__(FedJob(name="test_recipe_meta_copy", min_clients=1))

        recipe = BasicRecipe()
        launcher_spec = {"site-1": {"docker": {"image": "nvflare:latest"}}}
        mandatory_clients = ["site-1"]

        set_recipe_meta(recipe, JobMetaKey.JOB_LAUNCHER_SPEC, launcher_spec)
        set_recipe_meta(recipe, JobMetaKey.MANDATORY_CLIENTS, mandatory_clients)
        launcher_spec["site-1"]["docker"]["image"] = "changed"
        mandatory_clients.append("site-2")

        assert recipe.job.job.meta_props["launcher_spec"] == {"site-1": {"docker": {"image": "nvflare:latest"}}}
        assert recipe.job.job.meta_props["mandatory_clients"] == ["site-1"]

    def test_set_recipe_meta_requires_recipe_with_fed_job(self):
        with pytest.raises(TypeError, match="recipe must provide a FedJob through recipe.job"):
            set_recipe_meta(object(), JobMetaKey.STUDY, "default")

    @pytest.mark.parametrize(
        "key, value, error_type, match",
        [
            (1, "value", TypeError, "key must be a JobMetaKey"),
            ("study", "default", TypeError, "key must be a JobMetaKey"),
            (JobMetaKey.STUDY, True, TypeError, "must be one of int, float, str, dict, or list"),
            (JobMetaKey.STUDY, None, TypeError, "must be one of int, float, str, dict, or list"),
            (JobMetaKey.STUDY, Decimal("1.5"), TypeError, "must be one of int, float, str, dict, or list"),
            (JobMetaKey.STUDY, 1 + 2j, TypeError, "must be one of int, float, str, dict, or list"),
        ],
    )
    def test_set_recipe_meta_validates_key_and_supported_value_types(self, key, value, error_type, match):
        from nvflare.recipe.spec import Recipe

        class BasicRecipe(Recipe):
            def __init__(self):
                super().__init__(FedJob(name="test_recipe_meta_validation", min_clients=1))

        with pytest.raises(error_type, match=match):
            set_recipe_meta(BasicRecipe(), key, value)


class _DummyExecEnv:
    def __init__(self):
        self.extra = {}

    def get_extra_prop(self, prop_name, default=None):
        return self.extra.get(prop_name, default)

    def deploy(self, job):
        return "dummy-job-id"

    def get_job_status(self, job_id):
        return None

    def abort_job(self, job_id):
        return None

    def get_job_result(self, job_id, timeout: float = 0.0):
        return None


class TestRecipeExecuteExportParamIsolation:
    """Test that execute/export do not permanently mutate recipe additional_params."""

    @pytest.fixture
    def temp_script(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Test training script\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_execute_server_params_do_not_accumulate(self, temp_script):
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_execute_param_isolation",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
            model={"class_path": "model.DummyModel", "args": {}},
        )

        env = _DummyExecEnv()
        server_app = recipe.job._deploy_map.get("server")
        assert server_app is not None
        assert server_app.app_config.additional_params == {}
        recipe.execute(env, server_exec_params={"param_a": 1})
        assert server_app.app_config.additional_params == {}

        recipe.execute(env, server_exec_params={"param_b": 2})
        assert server_app.app_config.additional_params == {}

    def test_execute_empty_server_params_temporarily_clear_then_restore_snapshot(self, temp_script):
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_execute_empty_param_isolation",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
            model={"class_path": "model.DummyModel", "args": {}},
        )

        env = _DummyExecEnv()
        server_app = recipe.job._deploy_map.get("server")
        assert server_app is not None
        server_app.app_config.additional_params.update({"persisted": 1})

        seen_during_execute = {}

        def _capture_deploy(job):
            server = job._deploy_map.get("server")
            assert server is not None
            seen_during_execute.update(server.app_config.additional_params)
            return "dummy-job-id"

        env.deploy = _capture_deploy

        recipe.execute(env, server_exec_params={})

        assert seen_during_execute == {}
        assert server_app.app_config.additional_params == {"persisted": 1}

    def test_execute_then_export_no_cross_contamination(self, temp_script):
        from nvflare.recipe.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_execute_export_isolation",
            num_rounds=2,
            min_clients=2,
            train_script=temp_script,
            model={"class_path": "model.DummyModel", "args": {}},
        )

        env = _DummyExecEnv()
        recipe.execute(env, server_exec_params={"from_execute": 1})

        with tempfile.TemporaryDirectory() as tmpdir:
            recipe.export(job_dir=tmpdir, server_exec_params={"from_export": 2})
            server_cfg_path = os.path.join(
                tmpdir, "test_execute_export_isolation", "app", "config", "config_fed_server.json"
            )
            with open(server_cfg_path) as f:
                server_cfg = json.load(f)

        assert "from_execute" not in server_cfg
        assert server_cfg.get("from_export") == 2

        server_app = recipe.job._deploy_map.get("server")
        assert server_app is not None
        assert server_app.app_config.additional_params == {}


def test_recipe_spec_import_strips_export_flags_from_sys_argv(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--export-dir", "/tmp/out", "--other", "value"])

    import nvflare.recipe.spec as spec_module

    importlib.reload(spec_module)

    assert sys.argv == ["python", "job.py", "--other", "value"]
    assert spec_module._peek_recipe_args() == (True, "/tmp/out")


def test_recipe_spec_import_bare_export_uses_default_dir(monkeypatch):
    import sys

    # The canonical invocation: `python job.py --export` with no --export-dir.
    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--other"])

    import nvflare.recipe.spec as spec_module

    importlib.reload(spec_module)

    assert sys.argv == ["python", "job.py", "--other"]
    assert spec_module._peek_recipe_args() == (True, spec_module.DEFAULT_EXPORT_DIR)


def test_recipe_spec_import_strips_export_dir_equals_form(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--export-dir=out", "--other"])

    import nvflare.recipe.spec as spec_module

    importlib.reload(spec_module)

    assert sys.argv == ["python", "job.py", "--other"]
    assert spec_module._peek_recipe_args() == (True, "out")


def test_consume_recipe_args_dangling_export_dir_does_not_raise(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--export-dir"])

    import nvflare.recipe.spec as spec_module

    importlib.reload(spec_module)  # malformed input must not raise on import

    # Transactional: the whole pass is abandoned -- export stays disabled and sys.argv
    # is left untouched so the caller's own parser can surface the leftover flags.
    assert spec_module._peek_recipe_args() == (False, spec_module.DEFAULT_EXPORT_DIR)
    assert sys.argv == ["python", "job.py", "--export", "--export-dir"]


def test_consume_recipe_args_freezes_import_time_decision(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--export-dir"])

    import nvflare.recipe.spec as spec_module

    importlib.reload(spec_module)

    # A later direct call returns the recorded import-time decision even if sys.argv
    # has since changed -- it must not re-scan and flip to a different answer.
    monkeypatch.setattr(sys, "argv", ["python", "job.py", "--export", "--export-dir", "out"])
    assert spec_module._consume_recipe_args() == (False, spec_module.DEFAULT_EXPORT_DIR)


def test_export_processes_falsy_env(tmp_path):
    from nvflare.recipe.fedavg import FedAvgRecipe
    from nvflare.recipe.spec import ExecEnv

    class _FalsyEnv(ExecEnv):
        def __bool__(self):
            return False

        def deploy(self, job):
            return "dummy-job-id"

        def get_job_status(self, job_id):
            return None

        def abort_job(self, job_id):
            return None

        def get_job_result(self, job_id, timeout: float = 0.0):
            return None

    recipe = FedAvgRecipe(
        name="test_export_falsy_env",
        num_rounds=2,
        min_clients=2,
        train_script=__file__,
        model={"class_path": "model.DummyModel", "args": {}},
    )

    seen = {}

    def _capture_process_env(env):
        seen["env"] = env

    recipe.process_env = _capture_process_env

    with tempfile.TemporaryDirectory() as tmpdir:
        recipe.export(job_dir=tmpdir, env=_FalsyEnv())

    assert "env" in seen


class TestRecipePerSiteConfigHelper:
    """Test generic helper-provided per-site recipe configuration."""

    def test_set_per_site_config_stores_sites_and_calls_recipe_hook(self):
        from nvflare.recipe.spec import Recipe

        class RecordingRecipe(Recipe):
            def __init__(self):
                super().__init__(FedJob(name="test_per_site_config", min_clients=1))
                self.applied_config = None

            def _apply_per_site_config(self, config):
                self.applied_config = config

        recipe = RecordingRecipe()
        config = {
            "site-1": {"data_path": "xxx", "batch_size": 4, "unknown_to_helper": True},
            "site-2": {"data_path": "yyy", "batch_size": 2},
        }

        set_per_site_config(recipe, config)

        assert recipe.configured_sites() == ["site-1", "site-2"]
        assert recipe.applied_config == config
        assert recipe.applied_config is not config
        assert recipe.applied_config["site-1"] is config["site-1"]

    def test_set_per_site_config_hook_mutation_does_not_change_configured_sites(self):
        from nvflare.recipe.spec import Recipe

        class MutatingRecipe(Recipe):
            def __init__(self):
                super().__init__(FedJob(name="test_per_site_config_hook_mutation", min_clients=1))

            def _apply_per_site_config(self, config):
                del config["site-1"]
                config["site-2"] = {"rewritten": True}
                config["site-3"] = {}

        recipe = MutatingRecipe()

        set_per_site_config(recipe, {"site-1": {}, "site-2": {"batch_size": 4}})

        assert recipe.configured_sites() == ["site-1", "site-2"]

    def test_set_per_site_config_does_not_create_client_targets_by_itself(self):
        from nvflare.recipe.spec import Recipe

        class BasicRecipe(Recipe):
            def __init__(self):
                super().__init__(FedJob(name="test_per_site_no_targets", min_clients=1))

        recipe = BasicRecipe()

        set_per_site_config(recipe, {"site-1": {}, "site-2": {}})

        assert recipe.configured_sites() == ["site-1", "site-2"]
        assert recipe.job._deploy_map == {}

    def test_configured_sites_prefers_helper_config_over_legacy_constructor_config(self):
        from nvflare.recipe.spec import Recipe

        class LegacyRecipe(Recipe):
            def __init__(self):
                self.per_site_config = {"legacy-1": {}, "legacy-2": {}}
                super().__init__(FedJob(name="test_legacy_per_site_config", min_clients=1))

        recipe = LegacyRecipe()
        assert recipe.configured_sites() == ["legacy-1", "legacy-2"]

        set_per_site_config(recipe, {"helper-1": {}})

        assert recipe.configured_sites() == ["helper-1"]

    def test_empty_helper_config_still_overrides_legacy_constructor_config(self):
        from nvflare.recipe.spec import Recipe

        class LegacyRecipe(Recipe):
            def __init__(self):
                self.per_site_config = {"legacy-1": {}}
                super().__init__(FedJob(name="test_empty_helper_per_site_config", min_clients=1))

        recipe = LegacyRecipe()

        set_per_site_config(recipe, {})

        assert recipe.configured_sites() == []

    def test_configured_sites_does_not_infer_from_job_meta(self):
        from nvflare.recipe.spec import Recipe

        class MetaRecipe(Recipe):
            def __init__(self):
                super().__init__(
                    FedJob(
                        name="test_meta_sites_not_configured_sites",
                        min_clients=1,
                        mandatory_clients=["site-1"],
                        meta_props={
                            "resource_spec": {"site-1": {"num_of_gpus": 1}},
                            "launcher_spec": {"site-2": {"docker": {"image": "example/image:latest"}}},
                        },
                    )
                )

        recipe = MetaRecipe()

        assert recipe.configured_sites() == []

    @pytest.mark.parametrize(
        "config, match",
        [
            ("not-a-dict", "config must be a dict"),
            ({1: {}}, "per-site config key must be a str"),
            ({"site-1": "not-a-dict"}, "per-site config for site 'site-1' must be a dict"),
        ],
    )
    def test_set_per_site_config_validates_generic_shape_only(self, config, match):
        from nvflare.recipe.spec import Recipe

        class BasicRecipe(Recipe):
            def __init__(self):
                super().__init__(FedJob(name="test_per_site_validation", min_clients=1))

        with pytest.raises(TypeError, match=match):
            set_per_site_config(BasicRecipe(), config)
