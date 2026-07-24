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

import inspect
from unittest.mock import patch

import pytest
import torch.nn as nn

from nvflare.apis.dxo import DataKind
from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.app_opt.pt.recipes import FedProxRecipe as ExportedFedProxRecipe
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.app_opt.pt.recipes.fedprox import FedProxRecipe
from nvflare.client.config import ExchangeFormat, TransferType


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)


@pytest.fixture
def mock_file_system():
    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        yield


def _get_controller(recipe):
    server_app = recipe._job._deploy_map[SERVER_SITE_NAME]
    return server_app.app_config.workflows[0].controller


def _make_recipe(**kwargs):
    return FedProxRecipe(
        model=SimpleTestModel(),
        min_clients=2,
        train_script="client.py",
        **kwargs,
    )


class TestFedProxRecipe:
    def test_package_export_and_defaults(self, mock_file_system):
        recipe = _make_recipe()

        assert ExportedFedProxRecipe is FedProxRecipe
        assert recipe.name == "fedprox"
        assert recipe.fedprox_mu == 0.01
        assert inspect.signature(FedProxRecipe).parameters["fedprox_mu"].default == 0.01
        assert _get_controller(recipe).fedprox_mu == 0.01

    def test_signature_mirrors_fedavg(self):
        fedavg_parameters = inspect.signature(FedAvgRecipe).parameters
        fedprox_parameters = inspect.signature(FedProxRecipe).parameters

        assert set(fedprox_parameters) == set(fedavg_parameters) | {"fedprox_mu"}
        for name, fedavg_parameter in fedavg_parameters.items():
            fedprox_parameter = fedprox_parameters[name]
            assert fedprox_parameter.kind == fedavg_parameter.kind
            if name == "name":
                assert fedprox_parameter.default == "fedprox"
            else:
                assert fedprox_parameter.default == fedavg_parameter.default

    def test_custom_mu_configures_controller(self, mock_file_system):
        recipe = _make_recipe(fedprox_mu=0.2)

        assert recipe.fedprox_mu == 0.2
        assert _get_controller(recipe).fedprox_mu == 0.2

    @pytest.mark.parametrize("fedprox_mu", [None, 0.0, -0.1, float("inf"), float("nan"), True, "0.1"])
    def test_invalid_mu_is_rejected(self, mock_file_system, fedprox_mu):
        with pytest.raises((TypeError, ValueError), match="finite positive number"):
            _make_recipe(fedprox_mu=fedprox_mu)

    def test_inherits_fedavg_options(self, mock_file_system):
        recipe = _make_recipe(
            name="custom-fedprox",
            num_rounds=7,
            aggregator_data_kind=DataKind.WEIGHT_DIFF,
            server_expected_format=ExchangeFormat.PYTORCH,
            params_transfer_type=TransferType.DIFF,
            aggregation_weights={"site-1": 2.0},
            server_memory_gc_rounds=3,
            enable_tensor_disk_offload=True,
            client_memory_gc_rounds=4,
            cuda_empty_cache=True,
        )
        controller = _get_controller(recipe)

        assert recipe.name == "custom-fedprox"
        assert recipe.num_rounds == 7
        assert recipe.aggregator_data_kind == DataKind.WEIGHT_DIFF
        assert recipe.params_transfer_type == TransferType.DIFF
        assert recipe.aggregation_weights == {"site-1": 2.0}
        assert recipe.server_memory_gc_rounds == 3
        assert recipe.enable_tensor_disk_offload is True
        assert recipe.client_memory_gc_rounds == 4
        assert recipe.cuda_empty_cache is True
        assert controller.memory_gc_rounds == 3
        assert controller.enable_tensor_disk_offload is True
