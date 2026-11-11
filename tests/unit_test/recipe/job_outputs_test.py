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

from nvflare.apis.job_def import MAX_JOB_OUTPUTS, JobMetaKey
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.job_config.model import declare_pt_persistor_outputs
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe
from nvflare.job_config.api import FedJob
from nvflare.job_config.job_outputs import (
    as_folder,
    declare_metrics_outputs,
    declare_np_persistor_outputs,
    recorded_job_outputs,
    tag_component_outputs,
)
from nvflare.recipe import add_cross_site_evaluation, set_recipe_meta
from nvflare.recipe._job_outputs import outputs_from
from nvflare.recipe.fedstats import FedStatsRecipe
from nvflare.recipe.utils import setup_custom_persistor


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


def _fedavg(train_script, **kwargs):
    kwargs.setdefault("model", _Net())
    return FedAvgRecipe(name="hello-pt", min_clients=2, train_script=train_script, **kwargs)


def _exported_outputs(recipe, tmp_path):
    export_dir = tmp_path / "export"
    recipe.export(str(export_dir))
    with open(export_dir / recipe._job.name / "meta.json") as f:
        return json.load(f).get(JobMetaKey.JOB_OUTPUTS.value)


class TestRecipeDeclarations:
    def test_pytorch_fedavg_declares_its_models_and_metrics(self, tmp_path, train_script):
        outputs = _exported_outputs(_fedavg(train_script), tmp_path)

        assert sorted(outputs) == ["app_server/FL_global_model.pt", "app_server/best_FL_global_model.pt", "metrics/"]

    def test_sklearn_fedavg_declares_the_persistor_it_creates(self, tmp_path, train_script):
        recipe = SklearnFedAvgRecipe(name="sk", min_clients=2, train_script=train_script)

        assert sorted(_exported_outputs(recipe, tmp_path)) == ["app_server/model_param.joblib", "metrics/"]

    @pytest.mark.parametrize(
        "stats_output_path, expected",
        [("./output", ["output"]), ("statistics/adults.json", ["statistics/adults.json"])],
    )
    def test_fedstats_declares_its_normalized_output_path(self, tmp_path, stats_output_path, expected):
        from nvflare.app_common.abstract.statistics_spec import Statistics

        class _Generator(Statistics):
            def initialize(self, fl_ctx):
                pass

            def features(self):
                return {}

            def count(self, dataset_name, feature_name):
                return 0

        recipe = FedStatsRecipe(
            name="stats",
            stats_output_path=stats_output_path,
            sites=["site-1"],
            statistic_configs={"count": {}},
            stats_generator=_Generator(),
        )

        assert _exported_outputs(recipe, tmp_path) == expected

    def test_cross_site_evaluation_adds_its_folder(self, tmp_path, train_script):
        recipe = _fedavg(train_script)
        add_cross_site_evaluation(recipe)

        assert "cross_site_val/" in _exported_outputs(recipe, tmp_path)

    def test_a_component_added_after_an_earlier_export_is_declared(self, tmp_path, train_script):
        recipe = _fedavg(train_script)
        assert "cross_site_val/" not in _exported_outputs(recipe, tmp_path / "first")
        add_cross_site_evaluation(recipe)

        assert "cross_site_val/" in _exported_outputs(recipe, tmp_path / "second")

    def test_set_recipe_meta_replaces_the_declaration(self, tmp_path, train_script):
        recipe = _fedavg(train_script)
        set_recipe_meta(recipe, JobMetaKey.JOB_OUTPUTS, ["app_server/FL_global_model.pt"])
        add_cross_site_evaluation(recipe)

        assert _exported_outputs(recipe, tmp_path) == ["app_server/FL_global_model.pt"]

    def test_a_users_own_persistor_leaves_the_job_undeclared(self, tmp_path, train_script):
        recipe = _fedavg(train_script, model=None, model_persistor=PTFileModelPersistor(model=_Net()))

        assert _exported_outputs(recipe, tmp_path) is None

    @pytest.mark.parametrize("dest_dir", [None, ".", "./"])
    @pytest.mark.parametrize("as_folder_source", [False, True])
    def test_a_shipped_server_environment_leaves_the_job_undeclared(
        self, tmp_path, train_script, as_folder_source, dest_dir
    ):
        # environment.json can move where components write, such as PTFileModelPersistor's checkpoint folder
        config = tmp_path / "server_config"
        config.mkdir()
        (config / "environment.json").write_text(json.dumps({"APP_CKPT_DIR": "model"}))
        recipe = _fedavg(train_script)
        source = config if as_folder_source else config / "environment.json"
        recipe._job.add_file_to_server(str(source), dest_dir=dest_dir, app_folder_type="config")

        assert _exported_outputs(recipe, tmp_path / "out") is None

    def test_an_environment_file_in_a_config_subfolder_is_not_the_apps(self, tmp_path, train_script):
        env_file = tmp_path / "environment.json"
        env_file.write_text(json.dumps({"APP_CKPT_DIR": "model"}))
        recipe = _fedavg(train_script)
        recipe._job.add_file_to_server(str(env_file), dest_dir="extra", app_folder_type="config")

        assert "app_server/FL_global_model.pt" in _exported_outputs(recipe, tmp_path / "out")


class TestFailingSafe:
    def test_a_fedavg_recipe_without_a_persistor_declares_the_model_the_controller_saves(self, tmp_path, train_script):
        from nvflare.recipe.fedavg import FedAvgRecipe as BaseFedAvgRecipe

        model = {"class_path": "torch.nn.Linear", "args": {"in_features": 2, "out_features": 1}}
        recipe = BaseFedAvgRecipe(name="dict-model", min_clients=2, train_script=train_script, model=model)

        # FedAvg saves the model itself, relative to the run folder
        assert sorted(_exported_outputs(recipe, tmp_path)) == ["FL_global_model.pt", "metrics/"]

    def test_only_metrics_recorded_means_the_main_result_is_unknown(self, tmp_path):
        # a custom job: metrics recorded by BaseFedJob, a model saved by a controller nobody recorded
        from nvflare.app_common.workflows.fedavg import FedAvg
        from nvflare.job_config.base_fed_job import BaseFedJob
        from nvflare.recipe.spec import Recipe

        job = BaseFedJob(name="custom", min_clients=1)
        job.to_server(FedAvg(num_clients=1, num_rounds=1, persistor_id="", save_filename="final.pt"))

        assert _exported_outputs(Recipe(job), tmp_path) is None

    def test_a_persistor_added_without_a_declaration_leaves_the_job_undeclared(self, tmp_path, train_script):
        recipe = _fedavg(train_script)
        recipe._job.to_server(NPModelPersistor(), id="extra_persistor")

        assert _exported_outputs(recipe, tmp_path) is None

    def test_a_rejected_recorded_path_leaves_the_job_undeclared(self):
        job = FedJob(name="j")
        declare_np_persistor_outputs(job, NPModelPersistor(model_dir="/data/out"))
        declare_metrics_outputs(job, MetricsArtifactWriter())
        paths, unknown = recorded_job_outputs(job)

        assert not unknown and outputs_from(paths) == []


class TestDeclarationHelpers:
    def test_pytorch_persistor_declares_its_file_names(self):
        job = FedJob(name="j")
        declare_pt_persistor_outputs(job, PTFileModelPersistor(global_model_file_name="m.pt"))

        assert recorded_job_outputs(job) == (["app_server/m.pt", "app_server/best_FL_global_model.pt"], False)

    def test_a_pytorch_persistor_subclass_is_unknown(self):
        class _Elsewhere(PTFileModelPersistor):
            pass

        job = FedJob(name="j")
        declare_pt_persistor_outputs(job, _Elsewhere())

        assert recorded_job_outputs(job) == ([], True)

    def test_tensorflow_wrapper_declares_its_persistor(self):
        model = pytest.importorskip("nvflare.app_opt.tf.job_config.model")
        tf = pytest.importorskip("tensorflow")
        job = FedJob(name="j")
        job.to_server(model.TFModel(model=tf.keras.Sequential([tf.keras.layers.Dense(1)])))

        assert recorded_job_outputs(job) == (["app_server/tf_model.weights.h5"], False)

    @pytest.mark.parametrize(
        "writer, expected",
        [
            (MetricsArtifactWriter(), ["metrics/"]),
            (MetricsArtifactWriter(results_dir="./my_metrics/"), ["my_metrics/"]),
            (MetricsArtifactWriter(results_dir="."), ["metrics_summary.json", "round_metrics.jsonl"]),
            (MetricsArtifactWriter(results_dir=""), ["metrics_summary.json", "round_metrics.jsonl"]),
        ],
    )
    def test_metrics_writer_declares_its_folder_or_its_files(self, writer, expected):
        job = FedJob(name="j")
        declare_np_persistor_outputs(job, NPModelPersistor())  # a main result beside the metrics
        declare_metrics_outputs(job, writer)

        assert recorded_job_outputs(job) == (["models/server.npy"] + expected, False)

    def test_another_metrics_component_is_unknown(self):
        class _OwnWriter(MetricsArtifactWriter):
            pass

        job = FedJob(name="j")
        declare_metrics_outputs(job, _OwnWriter())

        assert recorded_job_outputs(job) == ([], True)

    @pytest.mark.parametrize(
        "persistor, expected",
        [
            (NPModelPersistor(), ["models/server.npy"]),
            (NPModelPersistor(model_dir="out", model_name="final"), ["out/final.npy"]),  # np.save adds .npy
        ],
    )
    def test_numpy_persistor_declares_its_npy_file(self, persistor, expected):
        job = FedJob(name="j")
        declare_np_persistor_outputs(job, persistor)

        assert recorded_job_outputs(job) == (expected, False)

    def test_a_persistor_the_recipe_tagged_is_declared_from_its_settings_when_added(self):
        persistor = PTFileModelPersistor()
        tag_component_outputs(persistor, lambda p: [f"app_server/{p.global_model_file_name}"])
        persistor.global_model_file_name = "renamed.pt"  # changed after tagging
        job = FedJob(name="j")

        setup_custom_persistor(job=job, model_persistor=persistor)

        assert recorded_job_outputs(job) == (["app_server/renamed.pt"], False)

    def test_an_untagged_persistor_is_unknown_when_added(self):
        job = FedJob(name="j")

        setup_custom_persistor(job=job, model_persistor=PTFileModelPersistor())

        assert recorded_job_outputs(job) == ([], True)

    @pytest.mark.parametrize("path, expected", [("metrics", "metrics/"), ("./m/", "m/"), (".", None), ("", None)])
    def test_as_folder(self, path, expected):
        assert as_folder(path) == expected


class TestOutputsFrom:
    def test_normalizes_and_removes_duplicates_and_covered_files(self):
        paths = ["./output", "metrics/", "metrics/metrics_summary.json", "output", "app_server/FL_global_model.pt"]

        assert outputs_from(paths) == ["output", "metrics/", "app_server/FL_global_model.pt"]

    @pytest.mark.parametrize("path", ["../outside.json", ".", "/abs/stats.json", ""])
    def test_an_undeclarable_path_declares_nothing(self, path):
        assert outputs_from([path, "metrics/"]) == []

    def test_more_outputs_than_allowed_declare_nothing(self):
        assert outputs_from([f"stats/{i}.json" for i in range(MAX_JOB_OUTPUTS + 1)]) == []
