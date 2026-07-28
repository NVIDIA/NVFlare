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

"""In-time aggregation with Collab response callbacks.

The server aggregates each client result as it arrives instead of waiting for
the complete group call to finish.

Run from the ``examples`` directory:

    python -m collab.async_aggregation.async_aggregation
"""

import argparse

from collab.async_aggregation.avg_intime import NPFedAvgInTime
from collab.async_aggregation.trainer import NPTrainer

from nvflare.collab import CollabRecipe, simple_logging
from nvflare.recipe import SimEnv


def make_recipe(args):
    recipe = CollabRecipe(
        job_name="collab_fedavg_intime",
        server=NPFedAvgInTime(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], num_rounds=args.num_rounds),
        client=NPTrainer(delta=1.0),
        min_clients=args.num_clients,
        sync_task_timeout=60,
    )
    recipe.set_server_prop("default_timeout", 8.0)
    recipe.set_client_prop("default_timeout", 5.0)
    return recipe


def main():
    parser = argparse.ArgumentParser(description="In-time aggregation with response callbacks")
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=2)
    args = parser.parse_args()
    simple_logging()
    run = make_recipe(args).execute(SimEnv(num_clients=args.num_clients))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
