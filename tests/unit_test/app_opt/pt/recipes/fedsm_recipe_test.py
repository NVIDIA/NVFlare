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

from nvflare.app_opt.pt.recipes.fedsm import FedSMRecipe
from nvflare.client.config import ExchangeFormat


def _models():
    return nn.Linear(2, 1), nn.Linear(2, 2)


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


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model": None}, "requires model"),
        ({"selector_model": None}, "requires selector_model"),
        ({"sites": []}, "non-empty list"),
        ({"sites": [""], "min_clients": 1}, "non-empty strings"),
        ({"sites": ["site-1", "site-1"]}, "unique site names"),
        ({"server_expected_format": ExchangeFormat.NUMPY}, "requires server_expected_format"),
        ({"site_label_mapping": {"site-1": 1, "site-2": 2}}, "contiguous selector labels"),
    ],
)
def test_fedsm_recipe_validates_contract(overrides, match):
    model, selector = _models()
    kwargs = {
        "model": model,
        "selector_model": selector,
        "sites": ["site-1", "site-2"],
        "min_clients": 2,
        "train_script": __file__,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        FedSMRecipe(**kwargs)


def test_fedsm_recipe_accepts_dict_model_configs():
    recipe = FedSMRecipe(
        model={"class_path": "torch.nn.Linear", "args": {"in_features": 2, "out_features": 1}},
        selector_model={"path": "torch.nn.Linear", "args": {"in_features": 2, "out_features": 2}},
        sites=["site-1"],
        min_clients=1,
        train_script=__file__,
    )

    assert recipe.model["path"] == "torch.nn.Linear"
    assert recipe.selector_model["path"] == "torch.nn.Linear"


def test_fedsm_recipe_supports_targeted_client_helpers():
    model, selector = _models()
    recipe = FedSMRecipe(
        model=model,
        selector_model=selector,
        sites=["site-1", "site-2"],
        min_clients=2,
        train_script=__file__,
    )

    recipe.add_client_config({"site_setting": 1}, clients=["site-1"])
    assert recipe.configured_sites() == ["site-1", "site-2"]
    recipe.set_per_site_config({"site-1": {}, "site-2": {}})
    assert recipe.configured_sites() == ["site-1", "site-2"]
    with pytest.raises(ValueError, match="unknown client site"):
        recipe.add_client_config({"site_setting": 3}, clients=["site-3"])


def test_pt_recipe_package_lazy_loads_fedsm():
    import nvflare.app_opt.pt.recipes as recipes

    assert recipes.__getattr__("FedSMRecipe") is FedSMRecipe


def test_fedsm_recipe_exports_server_components(tmp_path):
    model, selector = _models()
    recipe = FedSMRecipe(
        name="test-fedsm-export",
        model=model,
        selector_model=selector,
        sites=["site-1", "site-2"],
        min_clients=2,
        train_script=__file__,
        load_weights_only=False,
    )
    recipe.add_client_config({"site_setting": 1}, clients=["site-1"])
    recipe.export(str(tmp_path / "fedsm"))

    config = json.loads((tmp_path / "fedsm/test-fedsm-export/app_server/config/config_fed_server.json").read_text())
    assert config["workflows"][0]["path"].endswith("pt.fedsm.FedSM")
    persistor = next(component for component in config["components"] if component["id"] == "persistor")
    assert persistor["path"].endswith("PTFedSMModelPersistor")
    assert persistor["args"]["load_weights_only"] is False

    meta = json.loads((tmp_path / "fedsm/test-fedsm-export/meta.json").read_text())
    assert meta["mandatory_clients"] == ["site-1", "site-2"]
    assert meta["deploy_map"] == {
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
