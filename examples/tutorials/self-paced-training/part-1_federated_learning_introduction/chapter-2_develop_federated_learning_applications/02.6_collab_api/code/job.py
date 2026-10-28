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

from client import Trainer
from server import FedAvg

from nvflare.collab import CollabRecipe
from nvflare.recipe import SimEnv

if __name__ == "__main__":
    recipe = CollabRecipe(job_name="collab_fedavg", server=FedAvg(), client=Trainer(), min_clients=2)
    recipe.set_per_site_config(
        {
            "site-1": {"seed": 1, "samples": 80},
            "site-2": {"seed": 2, "samples": 120},
        }
    )
    # Include the training helper alongside the client in exported jobs.
    recipe.add_client_file("model.py")
    run = recipe.execute(
        SimEnv(clients=recipe.configured_sites(), workspace_root="/tmp/nvflare/chapter2/collab", log_config="progress")
    )
    print("Results at:", run.get_result())
