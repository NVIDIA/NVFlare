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
"""
This script demonstrates how to run the cyclic script runner for federated learning.
"""

from model import Net

from nvflare.app_opt.tf.job_config.model import TFModel
from nvflare.job_config.cyclic_recipe import CyclicRecipe
from nvflare.job_config.script_runner import FrameworkType

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 3
    train_script = "client.py"

    recipe = CyclicRecipe(
        framework=FrameworkType.TENSORFLOW,
        min_clients=1,
        num_rounds=num_rounds,
        model=TFModel(Net()),
        client_script=train_script,
        client_script_args="",
    )

    recipe.execute(clients=n_clients)
