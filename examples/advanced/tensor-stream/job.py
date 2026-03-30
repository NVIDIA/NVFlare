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

"""Example of using FedAvgRecipe with Tensor Streamers.
This example demonstrates how to set up and execute a federated learning job using
the FedAvgRecipe in NVFlare with Tensor Streamers for efficient tensor communication."""
import argparse

from model import get_model

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.app_opt.tensor_stream.client import TensorClientStreamer
from nvflare.app_opt.tensor_stream.server import TensorServerStreamer
from nvflare.client.config import ExchangeFormat
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds

    model, _ = get_model("gpt2")

    recipe = FedAvgRecipe(
        name="hello-tensor-stream",
        min_clients=n_clients,
        num_rounds=num_rounds,
        model=model,
        server_expected_format=ExchangeFormat.PYTORCH,
        train_script="client.py",
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    recipe.job.to_server(TensorServerStreamer(), "tensor_server_streamer")
    recipe.job.to_clients(TensorClientStreamer(), "tensor_client_streamer")

    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()


if __name__ == "__main__":
    main()
