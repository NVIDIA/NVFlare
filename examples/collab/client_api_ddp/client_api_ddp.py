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

"""Distributed data parallel (DDP) training under collab.

Each client launches its training script with torchrun; the script joins the
FL round via the Client API. Model sync across ranks is selected with --sync:

    checkpoint  rank 0 receives the model and shares it via a checkpoint file
                (simple, works everywhere, disk I/O per round)
    broadcast   rank 0 receives the model and dist.broadcast()s the tensors
                (no disk I/O)

Run:
    python -m collab.client_api_ddp.client_api_ddp --sync checkpoint --procs 2
    python -m collab.client_api_ddp.client_api_ddp --sync broadcast --procs 2
"""

from collab.client_api.server import FedAvg
from collab.common.runner import make_parser, run_recipe

from nvflare.collab import CollabClientAPI, CollabRecipe

TRAIN_MODULES = {
    "checkpoint": "collab.client_api_ddp.train_checkpoint",
    "broadcast": "collab.client_api_ddp.train_broadcast",
}


def make_recipe(args):
    return CollabRecipe(
        job_name=f"collab_client_api_ddp_{args.sync}",
        server=FedAvg(num_rounds=args.num_rounds),
        client=CollabClientAPI(),
        min_clients=args.num_clients,
        inprocess=False,
        run_cmd=f"torchrun --nproc_per_node={args.procs}",
        training_module=TRAIN_MODULES[args.sync],
        sync_task_timeout=60,
    )


def main():
    parser = make_parser("DDP under collab: --sync checkpoint | broadcast")
    parser.add_argument("--sync", choices=sorted(TRAIN_MODULES), default="checkpoint")
    parser.add_argument("--procs", type=int, default=2, help="torchrun processes per client")
    parser.add_argument("--num-rounds", type=int, default=3)
    args = parser.parse_args()
    run_recipe(make_recipe(args), args)


if __name__ == "__main__":
    main()
