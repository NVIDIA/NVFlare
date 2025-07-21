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
This code show to use NVIDIA FLARE Job Recipe to connect both Federated learning client and server algorithm
and run it under different environments
"""
from nvflare.app_opt.pt.job_config.Job_recipe import FedAvgRecipe
from model import SimpleNetwork

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "client.py"
    client_script_args = ""

    recipe = FedAvgRecipe(clients=n_clients,
                          num_rounds=num_rounds,
                          model= SimpleNetwork(),
                          client_script=train_script,
                          client_script_args= client_script_args)
    recipe.execute()

