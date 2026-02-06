# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from model import Net

from nvflare.app_opt.tf.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 3
    train_script = "client.py"

    recipe = FedAvgRecipe(
        name="hello-tf_fedavg",
        num_rounds=num_rounds,
        # Model can be specified as class instance or dict config:
        model=Net(),
        # Alternative: model={"path": "model.Net", "args": {}},
        # For pre-trained weights: initial_ckpt="/server/path/to/model.h5",
        min_clients=n_clients,
        train_script=train_script,
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env=env)
    print()
    print("Result can be found in :", run.get_result())
    print("Job Status is:", run.get_status())
    print()
