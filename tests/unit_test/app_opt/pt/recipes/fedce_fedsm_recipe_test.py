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

torch = pytest.importorskip("torch")
from torch import nn

from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.app_opt.pt.recipes.fedce import FedCERecipe
from nvflare.app_opt.pt.recipes.fedsm import FedSMRecipe
from nvflare.client.config import ExchangeFormat


def _models():
    return nn.Linear(2, 1), nn.Linear(2, 2)


def test_fedce_recipe_uses_fedavg_with_contribution_aggregator():
    model, _ = _models()
    recipe = FedCERecipe(model=model, min_clients=2, train_script=__file__)

    assert recipe.params_transfer_type.value == ParamsType.DIFF.value
    assert recipe.server_expected_format == ExchangeFormat.PYTORCH


def test_fedce_recipe_rejects_non_pytorch_exchange():
    model, _ = _models()
    with pytest.raises(ValueError, match="requires server_expected_format"):
        FedCERecipe(
            model=model,
            min_clients=2,
            train_script=__file__,
            server_expected_format=ExchangeFormat.NUMPY,
        )


def test_fedsm_recipe_builds_dedicated_components():
    model, selector = _models()
    recipe = FedSMRecipe(
        model=model,
        selector_model=selector,
        sites=["site-1", "site-2"],
        min_clients=2,
        train_script=__file__,
    )

    assert recipe.site_label_mapping == {"site-1": 0, "site-2": 1}
    assert recipe.configured_sites() == ["site-1", "site-2"]


def test_fedsm_recipe_requires_every_configured_client_each_round():
    model, selector = _models()
    with pytest.raises(ValueError, match=r"min_clients to equal len\(sites\)"):
        FedSMRecipe(
            model=model,
            selector_model=selector,
            sites=["site-1", "site-2"],
            min_clients=1,
            train_script=__file__,
        )


def test_fedsm_recipe_requires_exact_label_mapping():
    model, selector = _models()
    with pytest.raises(ValueError, match="keys must exactly match"):
        FedSMRecipe(
            model=model,
            selector_model=selector,
            sites=["site-1", "site-2"],
            site_label_mapping={"site-1": 0},
            min_clients=2,
            train_script=__file__,
        )


def test_fedsm_recipe_supports_pr_4862_targeted_client_helpers():
    model, selector = _models()
    recipe = FedSMRecipe(
        model=model,
        selector_model=selector,
        sites=["site-1", "site-2"],
        min_clients=2,
        train_script=__file__,
    )

    recipe.add_client_config({"site_setting": 1}, clients=["site-1"])
    with pytest.raises(ValueError, match="unknown client site"):
        recipe.add_client_config({"site_setting": 3}, clients=["site-3"])


def test_fedce_and_fedsm_recipes_export_server_components(tmp_path):
    model, selector = _models()
    FedCERecipe(
        name="test-fedce-export",
        model=model,
        min_clients=2,
        train_script=__file__,
    ).export(str(tmp_path / "fedce"))
    fedsm_recipe = FedSMRecipe(
        name="test-fedsm-export",
        model=model,
        selector_model=selector,
        sites=["site-1", "site-2"],
        min_clients=2,
        train_script=__file__,
    )
    fedsm_recipe.add_client_config({"site_setting": 1}, clients=["site-1"])
    fedsm_recipe.export(str(tmp_path / "fedsm"))

    fedce_config = json.loads((tmp_path / "fedce/test-fedce-export/app/config/config_fed_server.json").read_text())
    fedsm_config = json.loads(
        (tmp_path / "fedsm/test-fedsm-export/app_server/config/config_fed_server.json").read_text()
    )

    assert fedce_config["workflows"][0]["path"].endswith("workflows.fedavg.FedAvg")
    assert fedce_config["workflows"][0]["args"]["aggregator"]["path"].endswith("FedCEModelAggregator")
    assert fedsm_config["workflows"][0]["path"].endswith("pt.fedsm.FedSM")
    persistor = next(component for component in fedsm_config["components"] if component["id"] == "persistor")
    assert persistor["path"].endswith("PTFedSMModelPersistor")

    fedsm_meta = json.loads((tmp_path / "fedsm/test-fedsm-export/meta.json").read_text())
    assert fedsm_meta["deploy_map"] == {
        "app_server": ["server"],
        "app_site-1": ["site-1"],
        "app_site-2": ["site-2"],
    }
    site_1_config = json.loads(
        (tmp_path / "fedsm/test-fedsm-export/app_site-1/config/config_fed_client.json").read_text()
    )
    site_2_config = json.loads(
        (tmp_path / "fedsm/test-fedsm-export/app_site-2/config/config_fed_client.json").read_text()
    )
    assert site_1_config["site_setting"] == 1
    assert "site_setting" not in site_2_config
