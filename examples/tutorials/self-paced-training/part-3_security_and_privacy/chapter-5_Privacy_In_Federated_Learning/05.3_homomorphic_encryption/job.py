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

import sys
from pathlib import Path

from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.recipes import FedAvgRecipeWithHE
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.recipe import PocEnv, add_experiment_tracking

TUTORIAL_DIR = Path(__file__).resolve().parent
CLIENT_DIR = TUTORIAL_DIR / "code" / "fl"


def create_recipe() -> FedAvgRecipeWithHE:
    client_dir = str(CLIENT_DIR)
    if client_dir not in sys.path:
        sys.path.insert(0, client_dir)

    from net import Net

    recipe = FedAvgRecipeWithHE(
        name="cifar10_sag_pt_he",
        model=Net(),
        min_clients=2,
        num_rounds=2,
        train_script=str(CLIENT_DIR / "train.py"),
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
        launch_external_process=True,
        server_expected_format=ExchangeFormat.NUMPY,
        params_transfer_type=TransferType.DIFF,
    )
    recipe.add_client_file(str(CLIENT_DIR / "net.py"))
    add_experiment_tracking(recipe, tracking_type="tensorboard")
    return recipe


def main():
    recipe = create_recipe()
    run = recipe.execute(PocEnv(num_clients=2, use_he=True))
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result(clean_up=False))


if __name__ == "__main__":
    main()
