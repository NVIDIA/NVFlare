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

from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.app_opt.tensor_stream.client import TensorClientStreamer
from nvflare.app_opt.tensor_stream.server import TensorServerStreamer
from nvflare.client.config import ExchangeFormat
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner


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

    job = FedJob(name="hello-tensor-stream")
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )

    job.to_server(controller)
    job.to_server(PTModel(model))
    job.to_server(IntimeModelSelector(key_metric="accuracy"))

    executor = ScriptRunner(
        script="client.py",
        server_expected_format=ExchangeFormat.PYTORCH,
    )
    job.to_clients(executor, tasks=["train"])

    job.to_server(TensorServerStreamer(), "tensor_server_streamer")
    job.to_clients(TensorClientStreamer(), "tensor_client_streamer")

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0", n_clients=n_clients)


if __name__ == "__main__":
    main()
