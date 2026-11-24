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

torch = pytest.importorskip("torch")

from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.fedce import FedCEModelAggregator
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.app_opt.pt.recipes.fedce import FedCERecipe
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.recipe import set_per_site_config


@pytest.mark.parametrize("selection", ["common", "per_site", "export_override"])
def test_fedce_rejects_task_lifetime_at_export(tmp_path, selection):
    script = tmp_path / "client.py"
    script.write_text("pass\n")
    recipe = FedCERecipe(model=torch.nn.Linear(2, 1), min_clients=2, train_script=str(script))
    if selection == "export_override":
        kwargs = {"client_exec_params": {"execution_lifetime": "task"}}
    else:
        kwargs = {}
        if selection == "per_site":
            set_per_site_config(recipe, {"site-1": {}, "site-2": {}})
            recipe.add_client_config({"execution_lifetime": "task"}, clients=["site-1"])
        else:
            recipe.add_client_config({"execution_lifetime": "task"})
    with pytest.raises(ValueError, match="FedCE previous-model and score-history state.*use job lifetime"):
        recipe.export(str(tmp_path / "export"), **kwargs)


@pytest.mark.parametrize("subclass", [False, True])
def test_fedce_aggregator_cannot_bypass_task_guard_through_fedavg(tmp_path, subclass):
    class CustomFedCEAggregator(FedCEModelAggregator):
        pass

    script = tmp_path / "client.py"
    script.write_text("pass\n")
    aggregator_type = CustomFedCEAggregator if subclass else FedCEModelAggregator
    recipe = FedAvgRecipe(
        model=torch.nn.Linear(2, 1),
        min_clients=2,
        train_script=str(script),
        aggregator=aggregator_type(trainable_param_names=["weight", "bias"]),
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
        server_expected_format=ExchangeFormat.PYTORCH,
        params_transfer_type=TransferType.DIFF,
        execution_lifetime="task",
    )
    with pytest.raises(ValueError, match="FedCE previous-model and score-history state.*use job lifetime"):
        recipe.export(str(tmp_path / "export"))
