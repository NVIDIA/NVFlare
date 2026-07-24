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

"""Collab rewrite of ``examples/hello-world/hello-numpy`` using the same Recipe API.

The client publishes an ordinary Python function and the server calls it on
all clients as if it were local. NumPy arrays and floats are passed directly;
there are no Shareable, DXO, or FLModel transport objects.

Run from the ``examples`` directory:

    python -m collab.hello_numpy_collab.hello_numpy_collab
"""

import argparse

import numpy as np

from nvflare.collab import CollabRecipe, collab
from nvflare.recipe import SimEnv

INITIAL_MODEL = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)


@collab.publish
def train(model: np.ndarray, update_type: str):
    """Simulate local training by adding one to every model weight."""
    updated_model = model + 1
    model_update = updated_model - model if update_type == "diff" else updated_model
    return model_update, float(np.mean(updated_model))


@collab.main
def run():
    """Run the federated averaging loop on the server."""
    model = INITIAL_MODEL.copy()
    num_rounds = collab.get_app_prop("num_rounds", 3)
    update_type = collab.get_app_prop("update_type", "full")

    for current_round in range(num_rounds):
        print(f"Round {current_round + 1}: sending weights\n{model}")
        client_results = collab.clients.train(model, update_type)
        model_updates = []
        for client_name, (model_update, weight_mean) in client_results:
            print(f"  {client_name}: weight_mean={weight_mean:.1f}")
            model_updates.append(model_update)
        averaged_update = np.mean(model_updates, axis=0)
        model = model + averaged_update if update_type == "diff" else averaged_update
        print(f"Round {current_round + 1}: averaged weights\n{model}")

    print(f"Final model after {num_rounds} rounds:\n{model}")
    return model


def define_parser():
    """Use the core experiment options from ``hello-world/hello-numpy``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--update_type", choices=["full", "diff"], default="full")
    parser.add_argument("--export_config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--log_config",
        type=str,
        default=None,
        help="Log config mode ('concise', 'full', 'verbose'), filepath, or logging level",
    )
    return parser.parse_args()


def make_recipe(args):
    # No server or client objects are needed: CollabRecipe discovers the
    # decorated functions in this module and uses them for both sides.
    recipe = CollabRecipe(
        job_name="hello_numpy_collab",
        min_clients=args.n_clients,
        sync_task_timeout=60,
    )
    recipe.set_server_prop("num_rounds", args.num_rounds)
    recipe.set_server_prop("update_type", args.update_type)
    return recipe


def main():
    args = define_parser()
    recipe = make_recipe(args)

    if args.export_config:
        job_dir = "/tmp/nvflare/jobs/job_config"
        recipe.export(job_dir)
        print(f"Job config exported to {job_dir}")
        return

    env = SimEnv(num_clients=args.n_clients, log_config=args.log_config)
    run = recipe.execute(env)
    print()
    print("Result can be found in:", run.get_result())
    print("Job Status is:", run.get_status())
    print()


if __name__ == "__main__":
    main()
