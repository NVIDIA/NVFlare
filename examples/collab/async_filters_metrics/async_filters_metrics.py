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

"""In-time aggregation, filter chains, and metrics.

Two flavors, selected with --flavor:

    np  in-time (asynchronous) aggregation: the server aggregates each client
        result as it arrives instead of waiting for the round to complete.
        Call/result filter chains (AddNoiseToModel, Print) are attached on
        both sides, and a MetricReceiver collects tracked metrics.

    pt  typed CallFilter/ResultFilter classes transforming a PyTorch model on
        its way out of the server and into the clients (and back). These
        filters move the model by reference (fuel downloader), which needs a
        FLARE backend: run this flavor with --runtime multi_process or prod.

Run:
    python -m collab.async_filters_metrics.async_filters_metrics --flavor np
    python -m collab.async_filters_metrics.async_filters_metrics --flavor pt --runtime multi_process
"""

from collab.async_filters_metrics.avg_intime import NPFedAvgInTime
from collab.async_filters_metrics.np_filters import AddNoiseToModel, Print
from collab.common.np_trainer import NPTrainer
from collab.common.runner import make_parser, run_recipe
from collab.common.widgets import MetricReceiver

from nvflare.collab import CollabRecipe


def make_np_recipe(args):
    recipe = CollabRecipe(
        job_name="collab_fedavg_intime",
        server=NPFedAvgInTime(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], num_rounds=args.num_rounds),
        client=NPTrainer(delta=1.0),
        server_objects={"metric_receiver": MetricReceiver()},
        min_clients=args.num_clients,
        sync_task_timeout=60,
    )

    print_filter = Print()
    recipe.add_server_outgoing_call_filters("*.train", [AddNoiseToModel()])
    recipe.add_server_incoming_result_filters("*.train", [print_filter])
    recipe.set_server_prop("default_timeout", 8.0)

    recipe.add_client_incoming_call_filters("*.train", [print_filter])
    recipe.add_client_outgoing_result_filters("*.train", [print_filter])
    recipe.set_client_prop("default_timeout", 5.0)
    return recipe


def make_pt_recipe(args):
    # Keep the default NumPy flavor usable in a base installation without the
    # optional PyTorch dependency.
    from collab.async_filters_metrics.pt_algo import PTFedAvg, PTTrainer
    from collab.async_filters_metrics.pt_filters import (
        IncomingModelCallFilter,
        IncomingModelResultFilter,
        OutgoingModelCallFilter,
        OutgoingModelResultFilter,
    )

    recipe = CollabRecipe(
        job_name="collab_pt_fedavg_filter",
        server=PTFedAvg(
            initial_model={
                "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            },
            num_rounds=args.num_rounds,
        ),
        client=PTTrainer(delta=1.0),
        min_clients=args.num_clients,
        sync_task_timeout=60,
    )
    recipe.add_server_outgoing_call_filters(pattern="*.train", filters=[OutgoingModelCallFilter("weights")])
    recipe.add_server_incoming_result_filters(pattern="*.train", filters=[IncomingModelResultFilter()])
    recipe.add_client_incoming_call_filters(pattern="*.train", filters=[IncomingModelCallFilter("weights")])
    recipe.add_client_outgoing_result_filters(pattern="*.train", filters=[OutgoingModelResultFilter()])
    return recipe


def main():
    parser = make_parser("In-time aggregation, filters, and metrics")
    parser.add_argument("--flavor", choices=("np", "pt"), default="np")
    parser.add_argument("--num-rounds", type=int, default=2)
    args = parser.parse_args()
    if args.flavor == "pt" and args.runtime == "in_process":
        raise SystemExit("--flavor pt uses ref-based filters, which need a FLARE backend: add --runtime multi_process")
    recipe = make_np_recipe(args) if args.flavor == "np" else make_pt_recipe(args)
    run_recipe(recipe, args)


if __name__ == "__main__":
    main()
