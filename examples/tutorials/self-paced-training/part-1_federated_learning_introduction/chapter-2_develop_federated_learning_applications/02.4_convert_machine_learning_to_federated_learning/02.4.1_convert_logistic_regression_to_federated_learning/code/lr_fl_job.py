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

from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

if __name__ == "__main__":
    n_clients = 4
    num_rounds = 5
    data_root = "/tmp/flare/dataset/heart_disease_data"

    # Create FedAvgLrRecipe for Logistic Regression with Newton-Raphson
    recipe = FedAvgLrRecipe(
        name="newton_raphson_fedavg",
        num_rounds=num_rounds,
        damping_factor=0.8,
        num_features=13,
        train_script="src/newton_raphson_train.py",
        train_args=f"--data_root {data_root}",
        launch_external_process=True,
    )

    # Add experiment tracking
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    # Execute the recipe in simulation environment
    env = SimEnv(num_clients=n_clients, num_threads=n_clients)
    run = recipe.execute(env)
    result_location = run.get_result()
    print(f"Result location: {result_location}")
