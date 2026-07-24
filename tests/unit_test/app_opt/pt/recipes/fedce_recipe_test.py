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

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_opt.pt.fedce import FedCEConstants
from nvflare.app_opt.pt.recipes.fedce import FedCERecipe
from nvflare.client.config import ExchangeFormat


def test_fedce_recipe_uses_fedavg_with_contribution_aggregator():
    recipe = FedCERecipe(model=nn.Linear(2, 1), min_clients=2, train_script=__file__)

    assert recipe.params_transfer_type.value == ParamsType.DIFF.value
    assert recipe.server_expected_format == ExchangeFormat.PYTORCH


def test_fedce_recipe_rejects_non_pytorch_exchange():
    with pytest.raises(ValueError, match="requires server_expected_format"):
        FedCERecipe(
            model=nn.Linear(2, 1),
            min_clients=2,
            train_script=__file__,
            server_expected_format=ExchangeFormat.NUMPY,
        )


def test_fedce_dict_model_requires_explicit_trainable_names_and_filters_buffers():
    model = {"path": "torch.nn.BatchNorm1d", "args": {"num_features": 1}}
    with pytest.raises(ValueError, match="trainable_param_names is required"):
        FedCERecipe(model=model, min_clients=2, train_script=__file__)

    recipe = FedCERecipe(
        model=model,
        min_clients=2,
        train_script=__file__,
        trainable_param_names=["weight", "bias"],
    )
    for client in ("site-1", "site-2"):
        recipe.fedce_aggregator.accept_model(
            FLModel(
                params={
                    "weight": torch.tensor([1.0]),
                    "bias": torch.tensor([0.0]),
                    "running_mean": torch.tensor([10.0]),
                    "running_var": torch.tensor([2.0]),
                    "num_batches_tracked": torch.tensor(3),
                },
                params_type=ParamsType.DIFF,
                meta={"client_name": client, FedCEConstants.MINUS_MODEL_SCORE: 0.5},
            )
        )

    assert recipe.fedce_aggregator._get_cosine_param_names(["site-1", "site-2"]) == ["bias", "weight"]


@pytest.mark.parametrize("names", [[], [""], ["weight", "weight"], "weight"])
def test_fedce_recipe_rejects_invalid_explicit_trainable_names(names):
    with pytest.raises(ValueError, match="non-empty list of unique"):
        FedCERecipe(
            model={"path": "torch.nn.Linear", "args": {"in_features": 1, "out_features": 1}},
            min_clients=2,
            train_script=__file__,
            trainable_param_names=names,
        )


def test_fedce_recipe_requires_at_least_one_trainable_parameter():
    with pytest.raises(ValueError, match="at least one trainable parameter"):
        FedCERecipe(model=nn.Identity(), min_clients=2, train_script=__file__)


def test_pt_recipe_package_lazy_loads_fedce():
    import nvflare.app_opt.pt.recipes as recipes

    assert recipes.__getattr__("FedCERecipe") is FedCERecipe


def test_fedce_recipe_exports_server_components(tmp_path):
    FedCERecipe(
        name="test-fedce-export",
        model=nn.Linear(2, 1),
        min_clients=2,
        train_script=__file__,
    ).export(str(tmp_path / "fedce"))

    config = json.loads((tmp_path / "fedce/test-fedce-export/app/config/config_fed_server.json").read_text())
    assert config["workflows"][0]["path"].endswith("workflows.fedavg.FedAvg")
    assert config["workflows"][0]["args"]["aggregator"]["path"].endswith("FedCEModelAggregator")
