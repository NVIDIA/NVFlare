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

pytest.importorskip("xgboost")


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


def test_xgb_bagging_recipe_configures_metrics_artifact_writer():
    from nvflare.app_opt.xgboost.recipes import XGBBaggingRecipe

    recipe = XGBBaggingRecipe(name="xgb_bagging", min_clients=2, per_site_config=_per_site_config())

    assert isinstance(_get_metrics_writer(recipe), MetricsArtifactWriter)


def test_xgb_horizontal_recipe_configures_metrics_artifact_writer():
    from nvflare.app_opt.xgboost.recipes import XGBHorizontalRecipe

    recipe = XGBHorizontalRecipe(name="xgb_horizontal", min_clients=2, num_rounds=1, per_site_config=_per_site_config())

    assert isinstance(_get_metrics_writer(recipe), MetricsArtifactWriter)


def test_xgb_vertical_recipe_configures_metrics_artifact_writer():
    from nvflare.app_opt.xgboost.recipes import XGBVerticalRecipe

    recipe = XGBVerticalRecipe(
        name="xgb_vertical",
        min_clients=2,
        num_rounds=1,
        label_owner="site-1",
        per_site_config=_per_site_config(),
    )

    assert isinstance(_get_metrics_writer(recipe), MetricsArtifactWriter)
