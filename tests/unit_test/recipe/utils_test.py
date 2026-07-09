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
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.job_def import JobMetaKey
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import Recipe
from nvflare.recipe.utils import (
    extract_persistor_id,
    prepare_initial_ckpt,
    resolve_initial_ckpt,
    set_recipe_meta,
    setup_custom_persistor,
    validate_ckpt,
)


@pytest.fixture
def temp_workdir():
    """Fixture that creates a temp directory and changes into it, restoring cwd after."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(original_cwd)


class TestValidateCkpt:
    """Tests for validate_ckpt function."""

    def test_none_ckpt(self):
        """None should pass validation."""
        validate_ckpt(None)  # Should not raise

    def test_absolute_path_not_exists(self):
        """Absolute path that doesn't exist should pass (server-side path)."""
        validate_ckpt("/server/path/to/checkpoint.pt")  # Should not raise

    def test_absolute_path_exists(self):
        """Absolute path that exists locally should pass."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            validate_ckpt(ckpt_path)  # Should not raise
        finally:
            os.unlink(ckpt_path)

    def test_relative_path_exists(self, temp_workdir):
        """Relative path that exists locally should pass."""
        ckpt_file = "checkpoint.pt"
        open(ckpt_file, "w").close()
        validate_ckpt(ckpt_file)  # Should not raise

    def test_relative_path_not_exists(self):
        """Relative path that doesn't exist should raise ValueError."""
        with pytest.raises(ValueError, match="does not exist locally"):
            validate_ckpt("non_existent_checkpoint.pt")

    def test_relative_path_subdirectory_exists(self, temp_workdir):
        """Relative path in subdirectory that exists should pass."""
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_file = "checkpoints/model.pt"
        open(ckpt_file, "w").close()
        validate_ckpt(ckpt_file)  # Should not raise

    def test_relative_path_subdirectory_not_exists(self):
        """Relative path in subdirectory that doesn't exist should raise."""
        with pytest.raises(ValueError, match="does not exist locally"):
            validate_ckpt("checkpoints/non_existent.pt")


class TestPrepareInitialCkpt:
    """Tests for prepare_initial_ckpt function."""

    def test_none_ckpt(self):
        """None should return None."""
        job = MagicMock()
        result = prepare_initial_ckpt(None, job)
        assert result is None
        job.add_file_to_server.assert_not_called()

    def test_absolute_path_server_side(self):
        """Absolute path should be returned as-is (server-side path)."""
        job = MagicMock()
        abs_path = "/workspace/models/checkpoint.pt"
        result = prepare_initial_ckpt(abs_path, job)
        assert result == abs_path
        job.add_file_to_server.assert_not_called()

    def test_absolute_path_local_file(self):
        """Absolute path to local file should still be returned as-is (user intent)."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            job = MagicMock()
            result = prepare_initial_ckpt(ckpt_path, job)
            # Absolute paths are treated as server-side, not bundled
            assert result == ckpt_path
            job.add_file_to_server.assert_not_called()
        finally:
            os.unlink(ckpt_path)

    def test_relative_path_bundled(self, temp_workdir):
        """Relative path should be bundled and basename returned."""
        ckpt_file = "checkpoint.pt"
        open(ckpt_file, "w").close()

        job = MagicMock()
        result = prepare_initial_ckpt(ckpt_file, job)

        # Should bundle the file
        job.add_file_to_server.assert_called_once_with(ckpt_file)
        # Should return basename
        assert result == "checkpoint.pt"

    def test_relative_path_subdirectory_bundled(self, temp_workdir):
        """Relative path in subdirectory should be bundled and basename returned."""
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_file = "checkpoints/model.pt"
        open(ckpt_file, "w").close()

        job = MagicMock()
        result = prepare_initial_ckpt(ckpt_file, job)

        # Should bundle the full relative path
        job.add_file_to_server.assert_called_once_with(ckpt_file)
        # Should return basename only
        assert result == "model.pt"

    def test_multiple_calls_different_files(self, temp_workdir):
        """Multiple calls with different files should bundle each."""
        ckpt1 = "ckpt1.pt"
        ckpt2 = "ckpt2.pt"
        open(ckpt1, "w").close()
        open(ckpt2, "w").close()

        job = MagicMock()

        result1 = prepare_initial_ckpt(ckpt1, job)
        assert result1 == "ckpt1.pt"

        result2 = prepare_initial_ckpt(ckpt2, job)
        assert result2 == "ckpt2.pt"

        assert job.add_file_to_server.call_count == 2


class TestPersistorUtils:
    """Tests for persistor utility helpers."""

    def test_extract_persistor_id(self):
        assert extract_persistor_id({"persistor_id": "persistor_a"}) == "persistor_a"
        assert extract_persistor_id({"persistor_id": 123}) == ""
        assert extract_persistor_id("persistor_b") == "persistor_b"
        assert extract_persistor_id(None) == ""

    def test_setup_custom_persistor_returns_empty_when_not_provided(self):
        job = MagicMock()

        result = setup_custom_persistor(job=job, model_persistor=None)

        assert result == ""
        job.to_server.assert_not_called()

    def test_setup_custom_persistor_registers_component(self):
        job = MagicMock()
        custom_persistor = object()
        job.to_server.return_value = "custom_persistor"

        result = setup_custom_persistor(job=job, model_persistor=custom_persistor)

        assert result == "custom_persistor"
        job.to_server.assert_called_once_with(custom_persistor, id="persistor")

    def test_setup_custom_persistor_extracts_dict_result(self):
        job = MagicMock()
        custom_persistor = object()
        job.to_server.return_value = {"persistor_id": "custom_from_dict"}

        result = setup_custom_persistor(job=job, model_persistor=custom_persistor)

        assert result == "custom_from_dict"

    def test_resolve_initial_ckpt_prefers_prepared_value(self):
        job = MagicMock()

        result = resolve_initial_ckpt(
            initial_ckpt="relative/path/model.pt",
            prepared_initial_ckpt="already_prepared.pt",
            job=job,
        )

        assert result == "already_prepared.pt"
        job.add_file_to_server.assert_not_called()

    def test_resolve_initial_ckpt_uses_prepare_when_prepared_missing(self, monkeypatch):
        calls = {}

        def fake_prepare(initial_ckpt, job):
            calls["initial_ckpt"] = initial_ckpt
            calls["job"] = job
            return "prepared_by_helper.pt"

        monkeypatch.setattr("nvflare.recipe.utils.prepare_initial_ckpt", fake_prepare)
        job = MagicMock()

        result = resolve_initial_ckpt(
            initial_ckpt="relative/path/model.pt",
            prepared_initial_ckpt=None,
            job=job,
        )

        assert result == "prepared_by_helper.pt"
        assert calls["initial_ckpt"] == "relative/path/model.pt"
        assert calls["job"] is job


class TestRecipePackageExports:
    """Tests for public API exports from nvflare.recipe."""

    def test_add_cross_site_evaluation_importable_from_recipe(self):
        """add_cross_site_evaluation must be importable from the top-level nvflare.recipe package."""
        from nvflare.recipe import add_cross_site_evaluation

        assert callable(add_cross_site_evaluation)

    def test_add_final_global_evaluation_importable_from_recipe(self):
        from nvflare.recipe import add_final_global_evaluation

        assert callable(add_final_global_evaluation)

    def test_set_per_site_config_importable_from_recipe(self):
        """set_per_site_config must be importable from the top-level nvflare.recipe package."""
        from nvflare.recipe import set_per_site_config

        assert callable(set_per_site_config)

    def test_set_recipe_meta_importable_from_recipe(self):
        """set_recipe_meta must be importable from the top-level nvflare.recipe package."""
        from nvflare.recipe import set_recipe_meta

        assert callable(set_recipe_meta)


class TestRecipeMetaHelper:
    """Test generic helper-provided recipe metadata."""

    def _make_recipe(self, name, **fed_job_kwargs):
        return Recipe(FedJob(name=name, **fed_job_kwargs))

    def _export_meta(self, recipe, tmp_path):
        recipe.job.export_job(str(tmp_path))
        with open(tmp_path / recipe.job.name / "meta.json") as f:
            return json.load(f)

    def test_set_recipe_meta_sets_recognized_top_level_meta_props(self, tmp_path):
        recipe = self._make_recipe("test_recipe_meta", min_clients=1, meta_props={"owner": "alice"})
        recipe.job.to_server({"server_arg": True})

        resource_spec = {
            "site-1": {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 4},
            "site-2": {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 2},
        }
        launcher_spec = {
            "site-1": {"docker": {"image": "nvflare-site1:latest"}},
            "site-2": {"docker": {"image": "nvflare-site2:latest"}},
        }

        set_recipe_meta(recipe, JobMetaKey.RESOURCE_SPEC, resource_spec)
        set_recipe_meta(recipe, JobMetaKey.JOB_LAUNCHER_SPEC, launcher_spec)
        set_recipe_meta(recipe, JobMetaKey.SCOPE, "private")
        set_recipe_meta(recipe, JobMetaKey.CUSTOM_PROPS, {"team": "research"})

        job_config = recipe.job.job
        # Dedicated FedJobConfig fields are untouched by the helper.
        assert job_config.min_clients == 1
        assert job_config.mandatory_clients is None
        assert job_config.resource_specs == {}
        assert job_config.meta_props == {
            "owner": "alice",
            "resource_spec": resource_spec,
            "launcher_spec": launcher_spec,
            "scope": "private",
            "custom_props": {"team": "research"},
        }

        exported_meta = self._export_meta(recipe, tmp_path)
        assert exported_meta["min_clients"] == 1
        assert exported_meta["resource_spec"] == resource_spec
        assert exported_meta["launcher_spec"] == launcher_spec
        assert exported_meta["scope"] == "private"
        assert exported_meta["custom_props"] == {"team": "research"}
        assert exported_meta["owner"] == "alice"

    def test_set_recipe_meta_warns_and_overrides_registered_resource_specs(self, tmp_path):
        recipe = self._make_recipe("test_recipe_meta_resource_conflict", min_clients=2)
        recipe.job.job.add_resource_spec("base-site", {"num_of_gpus": 0})
        recipe.job.to_server({"server_arg": True})

        resource_spec = {"site-1": {"num_of_gpus": 1}}
        with pytest.warns(UserWarning, match="overrides the per-site resource specs"):
            set_recipe_meta(recipe, JobMetaKey.RESOURCE_SPEC, resource_spec)

        assert recipe.job.job.meta_props["resource_spec"] == resource_spec
        # The dedicated FedJobConfig field is never mutated by the helper.
        assert recipe.job.job.resource_specs == {"base-site": {"num_of_gpus": 0}}

        # meta_props is merged last, so the registered per-site spec is overridden.
        exported_meta = self._export_meta(recipe, tmp_path)
        assert exported_meta["resource_spec"] == resource_spec

    def test_set_recipe_meta_replaces_different_value_from_existing_meta_props(self):
        recipe = self._make_recipe("test_recipe_meta_props_conflict", min_clients=1, meta_props={"scope": "a"})
        original_meta_props = recipe.job.job.meta_props
        set_recipe_meta(recipe, JobMetaKey.SCOPE, "b")

        assert recipe.job.job.meta_props is original_meta_props
        assert recipe.job.job.meta_props["scope"] == "b"

    def test_set_recipe_meta_allows_existing_meta_props_that_differ_from_generated_values(self):
        recipe = self._make_recipe("test_recipe_meta_generated_conflict", min_clients=2, meta_props={"min_clients": 5})
        set_recipe_meta(recipe, JobMetaKey.SCOPE, "private")

        assert recipe.job.job.meta_props["min_clients"] == 5
        assert recipe.job.job.meta_props["scope"] == "private"

    @pytest.mark.parametrize(
        "key",
        [
            JobMetaKey.DEPLOY_MAP,
            JobMetaKey.JOB_NAME,
            # Keys with dedicated FedJob constructor fields are rejected to keep a single
            # source of truth (set them via FedJob(min_clients=..., mandatory_clients=...)).
            JobMetaKey.MIN_CLIENTS,
            JobMetaKey.MANDATORY_CLIENTS,
            # STUDY is rejected because the server assigns it from the admin session's
            # active study at job submission, silently overwriting a recipe-set value.
            JobMetaKey.STUDY,
        ],
    )
    def test_set_recipe_meta_rejects_non_user_settable_keys(self, key):
        recipe = self._make_recipe("test_recipe_meta_restricted_key", min_clients=1)
        with pytest.raises(ValueError, match=rf"{key.value}.*cannot be set through set_recipe_meta"):
            set_recipe_meta(recipe, key, "value")

    def test_set_recipe_meta_stores_caller_independent_copy(self):
        recipe = self._make_recipe("test_recipe_meta_copy", min_clients=1)
        launcher_spec = {"site-1": {"docker": {"image": "nvflare:latest"}}}
        custom_props = {"clients": ["site-1"]}

        set_recipe_meta(recipe, JobMetaKey.JOB_LAUNCHER_SPEC, launcher_spec)
        set_recipe_meta(recipe, JobMetaKey.CUSTOM_PROPS, custom_props)
        launcher_spec["site-1"]["docker"]["image"] = "changed"
        custom_props["clients"].append("site-2")

        assert recipe.job.job.meta_props["launcher_spec"] == {"site-1": {"docker": {"image": "nvflare:latest"}}}
        assert recipe.job.job.meta_props["custom_props"] == {"clients": ["site-1"]}

    def test_set_recipe_meta_accepts_nested_booleans(self):
        recipe = self._make_recipe("test_recipe_meta_nested_bool", min_clients=1)
        set_recipe_meta(recipe, JobMetaKey.CUSTOM_PROPS, {"enabled": True, "flags": [False, True]})

        assert recipe.job.job.meta_props["custom_props"] == {"enabled": True, "flags": [False, True]}

    def test_set_recipe_meta_normalizes_non_string_dict_keys(self):
        recipe = self._make_recipe("test_recipe_meta_key_coercion", min_clients=1)
        set_recipe_meta(recipe, JobMetaKey.CUSTOM_PROPS, {1: "a"})

        # Stored value matches what meta.json will contain (keys coerced to strings),
        # so in-process consumers and reloaded-meta consumers agree.
        assert recipe.job.job.meta_props["custom_props"] == {"1": "a"}

    def test_set_recipe_meta_requires_recipe_with_fed_job(self):
        with pytest.raises(TypeError, match="recipe must provide a FedJob through recipe.job"):
            set_recipe_meta(object(), JobMetaKey.SCOPE, "private")

    def test_set_recipe_meta_rejects_wrong_typed_job_config(self):
        from types import SimpleNamespace

        recipe = SimpleNamespace(job=SimpleNamespace(job=object()))
        with pytest.raises(TypeError, match="recipe must provide a FedJob through recipe.job"):
            set_recipe_meta(recipe, JobMetaKey.SCOPE, "private")

    @pytest.mark.parametrize(
        "key, value, error_type, match",
        [
            (1, "value", TypeError, "key must be a JobMetaKey"),
            ("scope", "private", TypeError, "key must be a JobMetaKey"),
            (JobMetaKey.SCOPE, True, TypeError, "must be a str"),
            (JobMetaKey.SCOPE, None, TypeError, "must be a str"),
            (JobMetaKey.SCOPE, {"site-1": "public"}, TypeError, "must be a str"),
            (JobMetaKey.CUSTOM_PROPS, "not-a-dict", TypeError, "must be a dict"),
            (JobMetaKey.CUSTOM_PROPS, [1, 2], TypeError, "must be a dict"),
            (JobMetaKey.CUSTOM_PROPS, Decimal("1.5"), TypeError, "must be a dict"),
            (JobMetaKey.RESOURCE_SPEC, "2gpus", TypeError, "resource_spec.*must be a dict"),
            (JobMetaKey.RESOURCE_SPEC, {"site-1": "2gpus"}, TypeError, "site.*must be a dict"),
            (JobMetaKey.RESOURCE_SPEC, {1: {"num_of_gpus": 1}}, TypeError, "must be a str"),
            (JobMetaKey.JOB_LAUNCHER_SPEC, {"site-1": ["docker"]}, TypeError, "site.*must be a dict"),
        ],
    )
    def test_set_recipe_meta_validates_key_and_value_shapes(self, key, value, error_type, match):
        recipe = self._make_recipe("test_recipe_meta_validation", min_clients=1)
        with pytest.raises(error_type, match=match):
            set_recipe_meta(recipe, key, value)

    @pytest.mark.parametrize(
        "value, error_type",
        [
            ({"score": float("inf")}, ValueError),
            ({"score": float("-inf")}, ValueError),
            ({"score": float("nan")}, ValueError),
            ({"key": object()}, TypeError),
            ({"when": [datetime.now()]}, TypeError),
        ],
    )
    def test_set_recipe_meta_rejects_non_json_serializable_values(self, value, error_type):
        recipe = self._make_recipe("test_recipe_meta_json_validation", min_clients=1)
        with pytest.raises(error_type, match="must be JSON-serializable"):
            set_recipe_meta(recipe, JobMetaKey.CUSTOM_PROPS, value)


class TestCrossSiteEvalIdempotency:
    """Tests for resilient idempotency in add_cross_site_evaluation."""

    def test_idempotency_survives_missing_flag(self):
        from nvflare.fuel.utils.constants import FrameworkType
        from nvflare.recipe import FedAvgRecipe
        from nvflare.recipe.utils import add_cross_site_evaluation

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# dummy train script\n")
            train_script = f.name

        try:
            recipe = FedAvgRecipe(
                name="test_cse_idempotency",
                model=[1.0, 2.0],
                min_clients=2,
                num_rounds=2,
                train_script=train_script,
                framework=FrameworkType.NUMPY,
            )

            add_cross_site_evaluation(recipe)
            assert getattr(recipe, "_cse_added", False) is True

            # Simulate transient attribute loss (e.g. serialization boundary).
            del recipe._cse_added
            assert not hasattr(recipe, "_cse_added")

            with pytest.raises(RuntimeError, match="already been added"):
                add_cross_site_evaluation(recipe)
        finally:
            os.unlink(train_script)

    def test_unified_numpy_recipe_supports_cross_site_evaluation(self):
        from nvflare.fuel.utils.constants import FrameworkType
        from nvflare.recipe import FedAvgRecipe
        from nvflare.recipe.utils import add_cross_site_evaluation

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# dummy train script\n")
            train_script = f.name

        try:
            recipe = FedAvgRecipe(
                name="test_unified_numpy_cse",
                model=[1.0, 2.0],
                min_clients=2,
                num_rounds=2,
                train_script=train_script,
                framework=FrameworkType.NUMPY,
            )

            add_cross_site_evaluation(recipe)

            assert getattr(recipe, "_cse_added", False) is True
        finally:
            os.unlink(train_script)

    def test_participating_clients_passed_to_cross_site_eval_controller(self, tmp_path, monkeypatch):
        from nvflare.app_common.workflows import cross_site_model_eval
        from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
        from nvflare.fuel.utils.constants import FrameworkType
        from nvflare.recipe import FedAvgRecipe
        from nvflare.recipe.utils import add_cross_site_evaluation

        train_script = tmp_path / "client.py"
        train_script.write_text("# dummy train script\n")
        participating_clients = ["site-1", "site-3"]
        captured_kwargs = {}

        class RecordingCrossSiteModelEval(CrossSiteModelEval):
            def __init__(self, *args, **kwargs):
                captured_kwargs.update(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(cross_site_model_eval, "CrossSiteModelEval", RecordingCrossSiteModelEval)

        recipe = FedAvgRecipe(
            name="test_cse_participating_clients",
            model=[1.0, 2.0],
            min_clients=2,
            num_rounds=2,
            train_script=str(train_script),
            framework=FrameworkType.NUMPY,
        )

        add_cross_site_evaluation(recipe, participating_clients=participating_clients)

        assert captured_kwargs["participating_clients"] == participating_clients


class TestFinalGlobalEvaluation:
    def _make_recipe(self, comp_ids=None, framework=None):
        from nvflare.job_config.script_runner import FrameworkType

        job = MagicMock()
        job._deploy_map = {}
        job.comp_ids = comp_ids
        return SimpleNamespace(job=job, framework=framework or FrameworkType.PYTORCH)

    def test_adds_locator_generator_and_final_eval_controller(self):
        from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
        from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
        from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
        from nvflare.recipe import add_final_global_evaluation

        recipe = self._make_recipe({"persistor_id": "persistor"})
        recipe.job.to_server.side_effect = ["final_model_locator", None, None]

        add_final_global_evaluation(recipe, participating_clients=["site-1"], validation_timeout=42)

        calls = recipe.job.to_server.call_args_list
        assert isinstance(calls[0].args[0], PTFileModelLocator)
        assert calls[0].kwargs == {"id": "final_model_locator"}
        assert isinstance(calls[1].args[0], ValidationJsonGenerator)
        controller = calls[2].args[0]
        assert isinstance(controller, CrossSiteModelEval)
        assert controller._model_locator_id == "final_model_locator"
        assert controller._submit_model_task_name == ""
        assert controller._participating_clients == ["site-1"]
        assert controller._validation_timeout == 42
        assert recipe.job.comp_ids["locator_id"] == "final_model_locator"
        assert recipe._cse_added is True

    def test_reuses_existing_model_locator(self):
        from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
        from nvflare.recipe import add_final_global_evaluation

        recipe = self._make_recipe({"persistor_id": "persistor", "locator_id": "existing_locator"})

        add_final_global_evaluation(recipe)

        assert recipe.job.to_server.call_count == 2
        controller = recipe.job.to_server.call_args_list[-1].args[0]
        assert isinstance(controller, CrossSiteModelEval)
        assert controller._model_locator_id == "existing_locator"
        assert controller._participating_clients is None

    @pytest.mark.parametrize(
        "participating_clients, error_type",
        [
            ("site-1", TypeError),
            (("site-1",), TypeError),
            ([1], TypeError),
            ([], ValueError),
        ],
    )
    def test_validates_participating_clients(self, participating_clients, error_type):
        from nvflare.recipe import add_final_global_evaluation

        recipe = self._make_recipe({"persistor_id": "persistor"})

        with pytest.raises(error_type, match="participating_clients must"):
            add_final_global_evaluation(recipe, participating_clients=participating_clients)

        recipe.job.to_server.assert_not_called()

    def test_rejects_duplicate_configuration(self):
        from nvflare.recipe import add_final_global_evaluation

        recipe = self._make_recipe({"persistor_id": "persistor"})
        recipe._cse_added = True

        with pytest.raises(RuntimeError, match="already configured"):
            add_final_global_evaluation(recipe)

    def test_requires_pytorch_recipe(self):
        from nvflare.job_config.script_runner import FrameworkType
        from nvflare.recipe import add_final_global_evaluation

        recipe = self._make_recipe({"persistor_id": "persistor"}, framework=FrameworkType.NUMPY)

        with pytest.raises(ValueError, match="supports PyTorch"):
            add_final_global_evaluation(recipe)

    @pytest.mark.parametrize("comp_ids", [None, [], "persistor"])
    def test_requires_component_id_mapping(self, comp_ids):
        from nvflare.recipe import add_final_global_evaluation

        recipe = self._make_recipe(comp_ids)

        with pytest.raises(ValueError, match="tracks component IDs"):
            add_final_global_evaluation(recipe)

    def test_requires_model_persistor(self):
        from nvflare.recipe import add_final_global_evaluation

        recipe = self._make_recipe({})

        with pytest.raises(ValueError, match="requires a PyTorch model persistor"):
            add_final_global_evaluation(recipe)

    def test_rejects_failed_model_locator_registration(self):
        from nvflare.recipe import add_final_global_evaluation

        recipe = self._make_recipe({"persistor_id": "persistor"})
        recipe.job.to_server.return_value = None

        with pytest.raises(RuntimeError, match="failed to register"):
            add_final_global_evaluation(recipe)


class TestAddExperimentTrackingClients:
    """Test client targeting and per-site configs in add_experiment_tracking."""

    @pytest.fixture
    def dummy_tracking(self, monkeypatch):
        """Register a dependency-free tracking type backed by types.SimpleNamespace."""
        import nvflare.recipe.utils as utils_mod

        monkeypatch.setitem(
            utils_mod.TRACKING_REGISTRY,
            "dummy",
            # json is always importable; argparse.Namespace(**config) acts as the receiver
            # (unlike types.SimpleNamespace it exposes __module__, so it also survives export).
            {"package": "json", "receiver_module": "argparse", "receiver_class": "Namespace"},
        )
        return "dummy"

    def _make_recipe(self, name="test_tracking_clients"):
        return Recipe(FedJob(name=name, min_clients=1))

    @pytest.fixture
    def dummy_mlflow(self, monkeypatch):
        """Replace MLflow with a dependency-free receiver while retaining MLflow defaults."""
        import nvflare.recipe.utils as utils_mod

        monkeypatch.setitem(
            utils_mod.TRACKING_REGISTRY,
            "mlflow",
            {"package": "json", "receiver_module": "argparse", "receiver_class": "Namespace"},
        )

    def test_mlflow_defaults_derive_from_recipe_name(self, dummy_mlflow):
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe("named_job")

        add_experiment_tracking(recipe, "mlflow")

        receiver = recipe.job._deploy_map["server"].app_config.components["receiver"]
        assert receiver.kw_args == {
            "experiment_name": "named_job-experiment",
            "run_name": "named_job-Client",
        }

    def test_mlflow_client_tracking_can_omit_config(self, dummy_mlflow):
        from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
        from nvflare.apis.job_def import ALL_SITES
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe("client_job")

        add_experiment_tracking(recipe, "mlflow", client_side=True, server_side=False)

        receiver = recipe.job._deploy_map[ALL_SITES].app_config.components["client_receiver"]
        assert receiver.kw_args == {
            "experiment_name": "client_job-experiment",
            "run_name": "client_job-Client",
        }
        assert receiver.events == [ANALYTIC_EVENT_TYPE]

    def test_mlflow_defaults_preserve_explicit_values_and_input(self, dummy_mlflow):
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe("named_job")
        config = {"kw_args": {"experiment_name": "custom-experiment"}}

        add_experiment_tracking(recipe, "mlflow", config)

        receiver = recipe.job._deploy_map["server"].app_config.components["receiver"]
        assert receiver.kw_args == {
            "experiment_name": "custom-experiment",
            "run_name": "named_job-Client",
        }
        assert config == {"kw_args": {"experiment_name": "custom-experiment"}}

    def test_client_side_tracking_specific_clients(self, dummy_tracking):
        from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
        from nvflare.apis.job_def import ALL_SITES
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe()
        add_experiment_tracking(
            recipe,
            dummy_tracking,
            {"tracking_uri": "file:///tmp/site-1/mlruns"},
            client_side=True,
            server_side=False,
            clients=["site-1"],
        )

        receiver = recipe.job._deploy_map["site-1"].app_config.components["client_receiver"]
        assert receiver.tracking_uri == "file:///tmp/site-1/mlruns"
        # Local (non-federated) analytics events are configured by default.
        assert receiver.events == [ANALYTIC_EVENT_TYPE]
        assert ALL_SITES not in recipe.job._deploy_map

    def test_client_side_tracking_per_site_configs(self, dummy_tracking):
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe()
        # Real recipes have per-site client apps (e.g. from per_site_config) before
        # tracking is added; targeting only existing sites is enforced.
        for site in ("site-1", "site-2"):
            recipe.job.to({"site_arg": site}, site)
        for site in ("site-1", "site-2"):
            add_experiment_tracking(
                recipe,
                dummy_tracking,
                {"tracking_uri": f"file:///tmp/{site}/mlruns"},
                client_side=True,
                server_side=False,
                clients=[site],
            )

        for site in ("site-1", "site-2"):
            receiver = recipe.job._deploy_map[site].app_config.components["client_receiver"]
            assert receiver.tracking_uri == f"file:///tmp/{site}/mlruns"

    def test_client_side_tracking_all_clients_by_default(self, dummy_tracking):
        from nvflare.apis.job_def import ALL_SITES
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe()
        add_experiment_tracking(recipe, dummy_tracking, {"tracking_uri": "u"}, client_side=True, server_side=False)

        receiver = recipe.job._deploy_map[ALL_SITES].app_config.components["client_receiver"]
        assert receiver.tracking_uri == "u"

    def test_server_side_tracking_unaffected_by_clients_feature(self, dummy_tracking):
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe()
        add_experiment_tracking(recipe, dummy_tracking, {"tracking_uri": "u"})

        components = recipe.job._deploy_map["server"].app_config.components
        assert components["receiver"].tracking_uri == "u"
        # Server receiver keeps federated (default) events.
        assert not hasattr(components["receiver"], "events")

    def test_clients_without_client_side_raises(self, dummy_tracking):
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe()
        with pytest.raises(ValueError, match="client_side=True"):
            add_experiment_tracking(recipe, dummy_tracking, {"tracking_uri": "u"}, clients=["site-1"])

    @pytest.mark.parametrize("bad_clients", ["site-1", [1, 2], [None]])
    def test_clients_must_be_list_of_str(self, dummy_tracking, bad_clients):
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe()
        with pytest.raises(TypeError, match="clients must be a list of str"):
            add_experiment_tracking(
                recipe, dummy_tracking, {"tracking_uri": "u"}, client_side=True, clients=bad_clients
            )

    def test_clients_empty_list_raises(self, dummy_tracking):
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe()
        with pytest.raises(ValueError, match="must not be empty"):
            add_experiment_tracking(recipe, dummy_tracking, {"tracking_uri": "u"}, client_side=True, clients=[])

    def test_clients_targeting_rejects_unknown_site(self, dummy_tracking):
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe()
        recipe.job.to({"site_arg": 1}, "site-1")
        recipe.job.to({"site_arg": 2}, "site-2")

        with pytest.raises(ValueError, match="unknown client site"):
            add_experiment_tracking(
                recipe,
                dummy_tracking,
                {"tracking_uri": "u"},
                client_side=True,
                server_side=False,
                clients=["site-3"],
            )

    def test_clients_targeting_rejects_all_sites_topology(self, dummy_tracking):
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe()
        # Default recipe topology: one client app for all clients.
        recipe.job.to_clients({"executor_standin": True})

        with pytest.raises(ValueError, match="applies to all clients"):
            add_experiment_tracking(
                recipe,
                dummy_tracking,
                {"tracking_uri": "u"},
                client_side=True,
                server_side=False,
                clients=["site-1"],
            )

    def test_per_site_client_receiver_survives_export(self, dummy_tracking, tmp_path):
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = self._make_recipe("test_tracking_export")
        recipe.job.to_server({"server_arg": True})
        add_experiment_tracking(
            recipe,
            dummy_tracking,
            {"tracking_uri": "file:///tmp/site-1/mlruns"},
            client_side=True,
            server_side=False,
            clients=["site-1"],
        )

        recipe.job.export_job(str(tmp_path))
        with open(tmp_path / "test_tracking_export" / "app_site-1" / "config" / "config_fed_client.json") as f:
            client_cfg = json.load(f)

        entry = next(c for c in client_cfg["components"] if c["id"] == "client_receiver")
        assert entry["path"].endswith("Namespace")
