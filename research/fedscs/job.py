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

"""FedSCS CIFAR-10 job."""

import os

from src.fedscs_aggregator import FedSCSAggregator
from src.model import SimpleCNN

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.recipe import SimEnv


def main():
    """Create and execute the FedSCS job."""
    job_dir = os.path.dirname(os.path.abspath(__file__))

    recipe = FedAvgRecipe(
        name="fedscs",
        min_clients=5,
        num_rounds=10,
        model=SimpleCNN(),
        train_script=os.path.join(job_dir, "client.py"),
        aggregator=FedSCSAggregator(),
        params_transfer_type=TransferType.DIFF,
    )

    recipe.execute(SimEnv(num_clients=5))


if __name__ == "__main__":
    main()
