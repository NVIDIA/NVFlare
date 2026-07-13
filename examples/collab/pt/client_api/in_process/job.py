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

"""Job recipe for in-process Client API FedAvg example.

This example demonstrates the standard Client API pattern (receive/send)
running in-process (no subprocess, no torchrun). The client training code
uses the SAME imports as standard NVFlare:

    import nvflare.client as flare

    flare.init()
    while flare.is_running():
        model = flare.receive()
        # ... training ...
        flare.send(result)

The framework sets CLIENT_API_TYPE=COLLAB_IN_PROCESS_API to route the calls
to CollabClientAPI instead of the standard InProcessClientAPI.

Usage:
    python -m collab.pt.client_api.in_process.job
    python -m collab.pt.client_api.in_process.job --num-clients 5 --num-rounds 10

Architecture:
    Server (Collab API)              Client (Client API)
    ─────────────────                ───────────────────
    collab.clients.execute(model) ────▶ CollabClientAPI.execute()
                                       │
                                       ├─ set_api(self)  # Set global instance
                                       ├─ training_loop()
                                       │    │
                                       │    ├─ flare.receive()  ← returns stored model
                                       │    ├─ (train...)
                                       │    └─ flare.send(result) → stores result
                                       │
    result = execute() ◀───────────────┘

    collab.clients.stop() ─────────────▶ sets is_running() = False
"""

import argparse

from nvflare.client.in_process.collab_api import CollabClientAPI
from collab.pt.client_api.in_process.client import training_loop
from collab.pt.client_api.in_process.server import FedAvg
from nvflare.collab.sim import InProcessEnv
from nvflare.collab.sys.recipe import CollabRecipe


def main():
    parser = argparse.ArgumentParser(description="In-process Client API FedAvg Example")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--num-rounds", type=int, default=3, help="Number of FL rounds")
    args = parser.parse_args()

    print("=" * 60)
    print("In-Process Client API FedAvg Example")
    print("=" * 60)
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds:  {args.num_rounds}")
    print("  Mode:    In-process (no subprocess)")
    print("=" * 60)

    # Create server with FedAvg algorithm
    server = FedAvg(num_rounds=args.num_rounds)

    # Create client API adapter and register training function
    # The training function uses module-level flare.init(), flare.receive(), etc.
    client = CollabClientAPI()
    client.set_training_func(training_loop)

    # Create recipe for in-process execution
    recipe = CollabRecipe(
        job_name="fedavg_client_api_inprocess",
        server=server,
        client=client,
        min_clients=args.num_clients,
        inprocess=True,  # In-process mode - no subprocess
    )

    # Execute with InProcessEnv
    env = InProcessEnv(num_clients=args.num_clients)
    result = recipe.execute(env)

    print()
    print("=" * 60)
    print(f"Job completed! Status: {result.get_status()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
