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

"""Job recipe for Client API FedAvg example.

Usage:
    python job.py
    python job.py --gpus 2  # Multi-GPU with torchrun
"""

import argparse

from nvflare.client.in_process.collab_api import CollabClientAPI
from collab.pt.client_api.sub_process.server import FedAvg
from nvflare.collab.sim import InProcessEnv
from nvflare.collab.sys.recipe import CollabRecipe


def main():
    parser = argparse.ArgumentParser(description="Run FedAvg with Client API")
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Client API FedAvg Example")
    print("=" * 60)
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds:  {args.num_rounds}")
    if args.gpus:
        print(f"  GPUs:    {args.gpus} per client")
    print("=" * 60)

    # Server and client
    server = FedAvg(num_rounds=args.num_rounds)
    client = CollabClientAPI()

    # Build run_cmd for torchrun
    run_cmd = f"torchrun --nproc_per_node={args.gpus}" if args.gpus else None

    # Create recipe
    recipe = CollabRecipe(
        job_name="fedavg_client_api",
        server=server,
        client=client,
        min_clients=args.num_clients,
        inprocess=False,
        run_cmd=run_cmd,
        training_module="collab.pt.client_api.sub_process.client",
    )

    # Execute
    env = InProcessEnv(num_clients=args.num_clients)
    result = recipe.execute(env)

    print()
    print("=" * 60)
    print(f"Job completed! Status: {result.get_status()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
