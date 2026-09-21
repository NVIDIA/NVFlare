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
    data_dir = os.path.join(job_dir, "data")

    # Create the server-side model used by FedAvgRecipe.
    model = SimpleCNN()

    # Derive the authoritative parameter schema and dtypes from the
    # server-side model. Dtypes are stored as strings because the
    # recipe configuration must be JSON serializable.
    expected_schema = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    expected_dtypes = {name: str(value.dtype).replace("torch.", "") for name, value in model.state_dict().items()}

    # Defense-in-depth bound for received client DIFF updates.
    # This is separate from the published FedSCS scoring formulation.
    max_update_norm = 10.0

    aggregator = FedSCSAggregator(
        expected_schema=expected_schema,
        expected_dtypes=expected_dtypes,
        max_update_norm=max_update_norm,
    )

    # Five clients are used in this example to provide a peer group
    # for the FedSCS consensus calculation.
    recipe = FedAvgRecipe(
        name="fedscs",
        min_clients=5,
        num_rounds=10,
        model=model,
        train_script=os.path.join(job_dir, "client.py"),
        train_args=f"--data_dir {data_dir}",
        aggregator=aggregator,
        params_transfer_type=TransferType.DIFF,
        key_metric="global_accuracy",
    )

    recipe.execute(SimEnv(num_clients=5))


if __name__ == "__main__":
    main()
