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

import pytest
import torch.nn as nn

from nvflare.apis.job_def import JobMetaKey
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.statistics.json_stats_file_persistor import JsonStatsFileWriter
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.recipe import add_cross_site_evaluation, set_recipe_meta
from nvflare.recipe._job_outputs import outputs_of


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)


@pytest.fixture
def train_script(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('train')\n")
    return str(script)


def _fedavg(train_script):
    return FedAvgRecipe(name="hello-pt", min_clients=2, train_script=train_script, model=_Net())


def _exported_outputs(recipe, tmp_path):
    export_dir = tmp_path / "export"
    recipe.export(str(export_dir))
    with open(export_dir / recipe._job.name / "meta.json") as f:
        return json.load(f).get(JobMetaKey.JOB_OUTPUTS.value)


class TestRecipeDeclarations:
    def test_pytorch_fedavg_declares_its_models_and_metrics(self, tmp_path, train_script):
        outputs = _exported_outputs(_fedavg(train_script), tmp_path)

        assert sorted(outputs) == ["app_server/FL_global_model.pt", "app_server/best_FL_global_model.pt", "metrics/"]

    def test_cross_site_evaluation_adds_its_folder(self, tmp_path, train_script):
        recipe = _fedavg(train_script)
        add_cross_site_evaluation(recipe)

        assert "cross_site_val/" in _exported_outputs(recipe, tmp_path)

    def test_a_component_added_after_an_earlier_export_is_declared(self, tmp_path, train_script):
        recipe = _fedavg(train_script)
        assert "cross_site_val/" not in _exported_outputs(recipe, tmp_path / "first")
        add_cross_site_evaluation(recipe)

        assert "cross_site_val/" in _exported_outputs(recipe, tmp_path / "second")

    @pytest.mark.parametrize("dest_dir", [None, ".", "./"])
    @pytest.mark.parametrize("as_folder", [False, True])
    def test_a_checkpoint_folder_in_the_shipped_environment_is_declared(
        self, tmp_path, train_script, as_folder, dest_dir
    ):
        config = tmp_path / "server_config"
        config.mkdir()
        (config / "environment.json").write_text(json.dumps({"APP_CKPT_DIR": "model"}))
        recipe = _fedavg(train_script)
        source = config if as_folder else config / "environment.json"
        recipe._job.add_file_to_server(str(source), dest_dir=dest_dir, app_folder_type="config")

        outputs = _exported_outputs(recipe, tmp_path)

        assert "app_server/model/FL_global_model.pt" in outputs
        assert "app_server/model/best_FL_global_model.pt" in outputs
        assert "app_server/FL_global_model.pt" not in outputs

    @pytest.mark.parametrize("environment", ["not json", json.dumps({"APP_CKPT_DIR": "/abs/ckpt"})])
    def test_an_unresolvable_checkpoint_folder_leaves_the_job_undeclared(self, tmp_path, train_script, environment):
        env_file = tmp_path / "environment.json"
        env_file.write_text(environment)
        recipe = _fedavg(train_script)
        recipe._job.add_file_to_server(str(env_file), app_folder_type="config")

        assert _exported_outputs(recipe, tmp_path / "out") is None

    def test_set_recipe_meta_replaces_the_declaration(self, tmp_path, train_script):
        recipe = _fedavg(train_script)
        set_recipe_meta(recipe, JobMetaKey.JOB_OUTPUTS, ["app_server/FL_global_model.pt"])
        add_cross_site_evaluation(recipe)

        assert _exported_outputs(recipe, tmp_path) == ["app_server/FL_global_model.pt"]


class TestComponentOutputs:
    @pytest.mark.parametrize(
        "component, expected",
        [
            (
                PTFileModelPersistor(global_model_file_name="m.pt", best_global_model_file_name="best.pt"),
                ["app_server/m.pt", "app_server/best.pt"],
            ),
            (JoblibModelParamPersistor(save_name="model.joblib"), ["app_server/model.joblib"]),
            (NPModelPersistor(), ["models/server.npy"]),
            (NPModelPersistor(model_dir="out", model_name="final"), ["out/final.npy"]),  # np.save adds .npy
            (MetricsArtifactWriter(results_dir="my_metrics"), ["my_metrics/"]),
            (MetricsArtifactWriter(results_dir="./my_metrics/"), ["my_metrics/"]),
            # metrics written straight into the run folder are named one by one
            (MetricsArtifactWriter(results_dir="."), ["metrics_summary.json", "round_metrics.jsonl"]),
            (MetricsArtifactWriter(results_dir=""), ["metrics_summary.json", "round_metrics.jsonl"]),
            (JsonStatsFileWriter(output_path="./output"), ["output"]),
            (JsonStatsFileWriter(output_path="statistics/adult_stats.json"), ["statistics/adult_stats.json"]),
            (FedAvg(persistor_id="", save_filename="final.pt"), ["final.pt"]),
        ],
    )
    def test_each_component_declares_the_files_its_arguments_name(self, component, expected):
        assert outputs_of({"c": component}) == expected

    def test_xgboost_persistor_declares_its_model(self):
        model_persistor = pytest.importorskip("nvflare.app_opt.xgboost.tree_based.model_persistor")

        assert outputs_of({"p": model_persistor.XGBModelPersistor(save_name="xgb.json")}) == ["app_server/xgb.json"]

    @pytest.mark.parametrize("output_path", ["../outside.json", ".", "/abs/stats.json"])
    def test_an_undeclarable_path_is_skipped_not_raised(self, output_path):
        assert outputs_of({"w": JsonStatsFileWriter(output_path=output_path)}) == []

    def test_unknown_components_and_duplicates_add_nothing(self):
        components = {"other": object(), "m1": MetricsArtifactWriter(), "m2": MetricsArtifactWriter()}

        assert outputs_of(components) == ["metrics/"]

    def test_a_file_inside_a_declared_folder_is_not_repeated(self):
        components = {"w": JsonStatsFileWriter(output_path="metrics/stats.json"), "m": MetricsArtifactWriter()}

        assert outputs_of(components) == ["metrics/"]

    def test_fedavg_declares_its_own_save_only_when_its_persistor_is_not_configured(self):
        controller = FedAvg(persistor_id="persistor", save_filename="final.pt")

        assert outputs_of({"controller": controller}) == ["final.pt"]
        assert outputs_of({"controller": controller, "persistor": PTFileModelPersistor()}) == [
            "app_server/FL_global_model.pt",
            "app_server/best_FL_global_model.pt",
        ]

    def test_an_unknown_model_persistor_leaves_the_job_undeclared(self):
        class CustomPersistor(ModelPersistor):
            def load_model(self, fl_ctx):
                return None

            def save_model(self, model, fl_ctx):
                pass

        components = {"persistor": CustomPersistor(), "metrics_artifact_writer": MetricsArtifactWriter()}

        assert outputs_of(components) == []

    def test_a_component_that_cannot_tell_its_files_leaves_the_job_undeclared(self):
        class Unsure:
            def get_job_outputs(self, app_env, component_ids):
                return None

        assert outputs_of({"unsure": Unsure(), "metrics_artifact_writer": MetricsArtifactWriter()}) == []

    def test_a_component_without_the_method_writes_no_outputs(self):
        assert outputs_of({"plain": object(), "metrics_artifact_writer": MetricsArtifactWriter()}) == ["metrics/"]

    @pytest.mark.parametrize(
        "app_env, expected",
        [
            (
                {"APP_CKPT_DIR": "model"},
                ["app_server/model/FL_global_model.pt", "app_server/model/best_FL_global_model.pt"],
            ),
            ({"APP_CKPT_DIR": "/abs/model"}, []),
            ({"APP_CKPT_DIR": "../model"}, []),
            ({"APP_CKPT_DIR": 3}, []),
            ({}, ["app_server/FL_global_model.pt", "app_server/best_FL_global_model.pt"]),
        ],
    )
    def test_pytorch_persistor_follows_the_checkpoint_folder(self, app_env, expected):
        assert outputs_of({"persistor": PTFileModelPersistor()}, app_env) == expected

    def test_tensorflow_persistor_declares_its_model(self):
        model_persistor = pytest.importorskip("nvflare.app_opt.tf.model_persistor")

        assert outputs_of({"p": model_persistor.TFModelPersistor(save_name="tf.weights.h5")}) == [
            "app_server/tf.weights.h5"
        ]

    def test_metrics_in_the_run_folder_are_kept_beside_the_model(self):
        components = {
            "persistor": PTFileModelPersistor(),
            "metrics_artifact_writer": MetricsArtifactWriter(results_dir="."),
        }

        outputs = outputs_of(components)

        assert "metrics_summary.json" in outputs and "round_metrics.jsonl" in outputs
        assert "app_server/FL_global_model.pt" in outputs

    def test_an_environment_file_in_a_config_subfolder_is_not_the_apps(self, tmp_path, train_script):
        env_file = tmp_path / "environment.json"
        env_file.write_text(json.dumps({"APP_CKPT_DIR": "model"}))
        recipe = _fedavg(train_script)
        recipe._job.add_file_to_server(str(env_file), dest_dir="extra", app_folder_type="config")

        assert "app_server/FL_global_model.pt" in _exported_outputs(recipe, tmp_path / "out")
