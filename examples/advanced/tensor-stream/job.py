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
    parser.add_argument("--model-name", type=str, default="gpt2", help="Hugging Face model name.")
    parser.add_argument("--n_clients", type=int, default=2, help="Number of clients to simulate.")
    parser.add_argument("--num_rounds", type=int, default=2, help="Number of federated learning rounds.")
    parser.add_argument("--export-only", action="store_true", help="If set, export the job without executing it.")
    parser.add_argument(
        "--export-path", type=str, default="jobs/tensor-example-job-gpt2", help="Path to export the job."
    )
    parser.add_argument("--disable-tensorstream", action="store_true", help="If set, disable tensor streamers.")
    parser.add_argument(
        "--exchange-model-only", action="store_true", help="If set, only exchange the model without training."
    )

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds

    model, _ = get_model(args.model_name)

    if args.disable_tensorstream:
        server_expected_format = ExchangeFormat.NUMPY
    else:
        server_expected_format = ExchangeFormat.PYTORCH

    train_args = f"--model-name {args.model_name}"
    if args.exchange_model_only:
        train_args += " --exchange-model-only"

    recipe = FedAvgRecipe(
        name="hello-tensor-stream",
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=model,
        server_expected_format=server_expected_format,
        train_script="client.py",
        train_args=train_args,
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    if not args.disable_tensorstream:
        recipe.job.to_server(TensorServerStreamer(), "tensor_server_streamer")
        recipe.job.to_clients(TensorClientStreamer(), "tensor_client_streamer")

    if args.export_only:
        recipe.job.export_job(args.export_path)
        print(f"Job exported to {args.export_path}")
    else:
        env = SimEnv(num_clients=n_clients)
        run = recipe.execute(env)

        print()
        print("Job Status is:", run.get_status())
        print("Result can be found in :", run.get_result())
        print()


if __name__ == "__main__":
    main()
