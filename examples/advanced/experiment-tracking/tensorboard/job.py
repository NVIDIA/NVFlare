# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from model import SimpleNetwork

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.utils import add_experiment_tracking

if __name__ == "__main__":
    # Create FedAvg recipe
    recipe = FedAvgRecipe(
        name="fedavg_tensorboard",
        min_clients=2,
        num_rounds=5,
        initial_model=SimpleNetwork(),
        train_script="client.py",
    )

    # Add TensorBoard tracking
    add_experiment_tracking(recipe, "tensorboard", tracking_config={"tb_folder": "tb_events"})

    # Run in simulator
    recipe.run(workspace="/tmp/nvflare/jobs/workdir")
