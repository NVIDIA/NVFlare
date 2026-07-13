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

"""Job recipe for DDP with dist.broadcast sync.

This example uses dist.broadcast to sync model parameters across ranks.

Usage:
    python job_db.py --clients 2 --rounds 5 --gpus 2
"""

import argparse

from nvflare.client.in_process.collab_api import CollabClientAPI
from collab.pt.client_api.sub_process.server import FedAvg
from nvflare.collab.sim.in_process_env import InProcessEnv
from nvflare.collab.sys.recipe import CollabRecipe


def main():
    parser = argparse.ArgumentParser(description="DDP FedAvg with dist.broadcast")
    parser.add_argument("--clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--gpus", type=int, default=2, help="GPUs per client")
    args = parser.parse_args()

    print("=" * 60)
    print("DDP FedAvg with dist.broadcast sync")
    print("=" * 60)
    print(f"  Clients: {args.clients}")
    print(f"  Rounds:  {args.rounds}")
    print(f"  GPUs:    {args.gpus} per client")
    print("=" * 60)

    # Server-side algorithm
    server = FedAvg(num_rounds=args.rounds)

    # Client uses CollabClientAPI
    client = CollabClientAPI()

    # Build run command for torchrun
    run_cmd = f"torchrun --nproc_per_node={args.gpus}"

    # Create recipe
    recipe = CollabRecipe(
        job_name="fedavg_ddp_broadcast",
        server=server,
        client=client,
        min_clients=args.clients,
        inprocess=False,
        run_cmd=run_cmd,
        training_module="collab.pt.client_api.sub_process.client_db",
    )

    # Create simulation environment
    env = InProcessEnv(num_clients=args.clients, workspace_root="/tmp/nvflare/collab_simulation")

    # Execute
    result = recipe.execute(env)

    print("=" * 60)
    print(f"Job completed! Status: {result.get_status()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
