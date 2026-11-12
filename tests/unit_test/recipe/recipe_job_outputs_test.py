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
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe
from nvflare.recipe import add_cross_site_evaluation, set_recipe_meta
from nvflare.recipe.fedstats import FedStatsRecipe


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

    def test_a_builtin_persistor_the_user_passes_is_declared_from_its_settings(self, tmp_path, train_script):
        persistor = PTFileModelPersistor(model=_Net(), global_model_file_name="final.pt")
        recipe = _fedavg(train_script, model=None, model_persistor=persistor)

        assert "app_server/final.pt" in _exported_outputs(recipe, tmp_path)

    def test_a_users_own_persistor_leaves_the_job_undeclared(self, tmp_path, train_script):
        class _OwnPersistor(PTFileModelPersistor):  # may save elsewhere than its base class
            pass

        recipe = _fedavg(train_script, model=None, model_persistor=_OwnPersistor(model=_Net()))

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

    def test_a_plain_job_is_declared_from_its_components(self, tmp_path):
        from nvflare.app_common.workflows.fedavg import FedAvg
        from nvflare.job_config.base_fed_job import BaseFedJob
        from nvflare.recipe.spec import Recipe

        job = BaseFedJob(name="custom", min_clients=1)
        job.to_server(FedAvg(num_clients=1, num_rounds=1, persistor_id="", save_filename="final.pt"))

        assert sorted(_exported_outputs(Recipe(job), tmp_path)) == ["final.pt", "metrics/"]

    def test_a_persistor_added_without_a_resolver_leaves_the_job_undeclared(self, tmp_path, train_script):
        class _Extra(NPModelPersistor):
            pass

        recipe = _fedavg(train_script)
        recipe._job.to_server(_Extra(), id="extra_persistor")

        assert _exported_outputs(recipe, tmp_path) is None
