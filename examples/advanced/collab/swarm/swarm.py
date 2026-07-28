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

"""Decentralized (swarm) learning.

After a server kick-off, control moves around the clients themselves: each
round one client aggregates its peers' results and hands the model to the
next, exercising client-to-client Collab calls.

Run:
    python -m collab.swarm.swarm --num-clients 3
"""

import argparse

from collab.swarm.swarm_algo import NPSwarm, NPSwarmClient

from nvflare.collab import CollabRecipe
from nvflare.recipe import SimEnv


def make_recipe(args):
    return CollabRecipe(
        job_name="collab_swarm",
        server=NPSwarm(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], num_rounds=args.num_rounds),
        client=NPSwarmClient(delta=1.0),
        min_clients=args.num_clients,
        sync_task_timeout=60,
    )


def main():
    parser = argparse.ArgumentParser(description="Decentralized swarm learning")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    args = parser.parse_args()
    run = make_recipe(args).execute(SimEnv(num_clients=args.num_clients))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
