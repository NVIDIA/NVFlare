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

import pytest

from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter
from nvflare.recipe import set_per_site_config

pytest.importorskip("xgboost")

xgb_recipes = pytest.importorskip("nvflare.app_opt.xgboost.recipes")
XGBBaggingRecipe = xgb_recipes.XGBBaggingRecipe
XGBHorizontalRecipe = xgb_recipes.XGBHorizontalRecipe
XGBVerticalRecipe = xgb_recipes.XGBVerticalRecipe


class DummyDataLoader:
    pass


def _per_site_config():
    return {
        "site-1": {"data_loader": DummyDataLoader()},
        "site-2": {"data_loader": DummyDataLoader()},
    }


def _get_metrics_writer(recipe):
    server_app = recipe._job._deploy_map[SERVER_SITE_NAME]
    return server_app.app_config.components.get("metrics_artifact_writer")


def _get_server_controller(recipe, controller_id):
    server_app = recipe._job._deploy_map[SERVER_SITE_NAME]
    return next(
        (workflow.controller for workflow in server_app.app_config.workflows if workflow.id == controller_id),
        None,
    )


@pytest.mark.parametrize(
    "recipe",
    [
        pytest.param(XGBBaggingRecipe(name="xgb_bagging", min_clients=2), id="bagging"),
        pytest.param(XGBHorizontalRecipe(name="xgb_horizontal", min_clients=2, num_rounds=1), id="horizontal"),
        pytest.param(
            XGBVerticalRecipe(name="xgb_vertical", min_clients=2, num_rounds=1, label_owner="site-1"),
            id="vertical",
        ),
    ],
)
def test_xgb_recipes_apply_helper_config(recipe, tmp_path):
    config = _per_site_config()

    assert isinstance(_get_metrics_writer(recipe), MetricsArtifactWriter)
    with pytest.raises(RuntimeError, match="requires set_per_site_config"):
        recipe.export(str(tmp_path))

    set_per_site_config(recipe, config)

    assert recipe.configured_sites() == ["site-1", "site-2"]
    assert recipe._job.clients == []

    recipe.add_client_config({"configured": True})

    assert recipe._job.clients == ["site-1", "site-2"]
    for site_name, site_config in config.items():
        components = recipe._job._deploy_map[site_name].app_config.components
        assert components[recipe.data_loader_id] is site_config["data_loader"]
    assert _get_server_controller(recipe, "xgb_controller") is not None
    recipe._validate_before_use()


def test_xgb_legacy_constructor_config_delegates_to_helper():
    config = _per_site_config()

    with pytest.warns(FutureWarning, match="set_per_site_config"):
        recipe = XGBBaggingRecipe(name="xgb_legacy", min_clients=2, per_site_config=config)

    assert recipe.configured_sites() == ["site-1", "site-2"]
    assert recipe._job.clients == []
    recipe._ensure_client_apps_prepared()
    assert recipe._job.clients == ["site-1", "site-2"]


def test_xgb_per_site_config_validates_client_count_and_data_loader():
    recipe = XGBHorizontalRecipe(name="xgb_validation", min_clients=2, num_rounds=1)

    with pytest.raises(ValueError, match=r"defines 1 site.*min_clients=2"):
        set_per_site_config(recipe, {"site-1": {"data_loader": DummyDataLoader()}})
    with pytest.raises(ValueError, match="site-2.*data_loader"):
        set_per_site_config(
            recipe,
            {
                "site-1": {"data_loader": DummyDataLoader()},
                "site-2": {},
            },
        )

    assert recipe._job.clients == []
    assert recipe.configured_sites() == []


def test_xgb_vertical_resolves_label_owner_rank_when_helper_is_applied():
    recipe = XGBVerticalRecipe(
        name="xgb_vertical_ranks",
        min_clients=2,
        num_rounds=1,
        label_owner="site-2",
    )

    set_per_site_config(recipe, _per_site_config())

    assert recipe.client_ranks == {"site-2": 0, "site-1": 1}
    assert _get_server_controller(recipe, "xgb_controller") is None

    recipe._ensure_client_apps_prepared()

    controller = _get_server_controller(recipe, "xgb_controller")
    assert controller is not None
    assert controller.client_ranks == recipe.client_ranks
