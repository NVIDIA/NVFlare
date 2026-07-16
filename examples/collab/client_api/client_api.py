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

"""Client API under collab: the client side is standard Client API code
(``import nvflare.client as flare``; receive/train/send) while the server is
a collab ``@collab.main`` workflow. The framework routes the flare calls to
``CollabClientAPI``. Select the client execution style with --mode:

    in_process  the training loop runs as a function in the client process
    subprocess  a self-contained training script (train_script.py) runs in a
                separate process managed by the collab worker

Run:
    python -m collab.client_api.client_api --mode in_process
    python -m collab.client_api.client_api --mode subprocess
"""

from collab.client_api.in_process_client import training_loop
from collab.client_api.server import FedAvg
from collab.common.runner import make_parser, run_recipe

from nvflare.collab import CollabClientAPI, CollabRecipe


def make_recipe(args):
    server = FedAvg(num_rounds=args.num_rounds)
    client = CollabClientAPI()

    if args.mode == "in_process":
        client.set_training_func(training_loop)
        return CollabRecipe(
            job_name="collab_client_api_inprocess",
            server=server,
            client=client,
            min_clients=args.num_clients,
            inprocess=True,
            sync_task_timeout=60,
        )

    return CollabRecipe(
        job_name="collab_client_api_subprocess",
        server=server,
        client=client,
        min_clients=args.num_clients,
        inprocess=False,
        training_module="collab.client_api.train_script",
        sync_task_timeout=60,
    )


def main():
    parser = make_parser("Client API under collab: in_process | subprocess")
    parser.add_argument("--mode", choices=("in_process", "subprocess"), default="in_process")
    parser.add_argument("--num-rounds", type=int, default=3)
    args = parser.parse_args()
    run_recipe(make_recipe(args), args)


if __name__ == "__main__":
    main()
