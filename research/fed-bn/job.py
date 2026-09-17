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
from nvflare.app_opt.pt.recipes import FedAvgRecipe
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.recipe import SimEnv, add_experiment_tracking

FEDBN_DIR = Path(__file__).resolve().parent
SOURCE_DIR = FEDBN_DIR / "src"


def create_recipe() -> FedAvgRecipe:
    source_dir = str(SOURCE_DIR)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    from net import Net

    recipe = FedAvgRecipe(
        name="job",
        model=Net(),
        min_clients=2,
        num_rounds=2,
        train_script=str(SOURCE_DIR / "fedbn_cifar10.py"),
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
        launch_external_process=True,
        server_expected_format=ExchangeFormat.NUMPY,
        params_transfer_type=TransferType.DIFF,
        key_metric="accuracy",
    )
    recipe.add_client_file(str(SOURCE_DIR / "net.py"))
    add_experiment_tracking(recipe, tracking_type="tensorboard")
    return recipe


def main():
    recipe = create_recipe()
    env = SimEnv(num_clients=2, num_threads=2, workspace_root="/tmp/nvflare/fed_bn/workspace")
    run = recipe.execute(env)
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result(clean_up=False))


if __name__ == "__main__":
    main()
