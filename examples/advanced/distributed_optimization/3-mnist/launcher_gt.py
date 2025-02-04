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
import torch
from config import ITERATIONS, NUM_CLIENTS, STEPSIZE
from utils import NeuralNetwork, get_dataloaders, plot_results

from nvflare.app_opt.p2p.controllers import DistOptController
from nvflare.app_opt.p2p.executors import GTExecutor
from nvflare.app_opt.p2p.types import Config
from nvflare.app_opt.p2p.utils.config_generator import generate_random_network
from nvflare.job_config.api import FedJob


class CustomGTExecutor(GTExecutor):
    def __init__(self, data_chunk: int | None = None):
        self._data_chunk = data_chunk
        train_dataloader, test_dataloader = get_dataloaders(data_chunk)
        super().__init__(
            model=NeuralNetwork(),
            loss=torch.nn.CrossEntropyLoss(),
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        )


if __name__ == "__main__":
    # Create job
    job_name = "gt"
    job = FedJob(name=job_name)

    # generate random config
    network, _ = generate_random_network(num_clients=NUM_CLIENTS)
    config = Config(network=network, extra={"iterations": ITERATIONS, "stepsize": STEPSIZE})

    # send controller to server
    controller = DistOptController(config=config)
    job.to_server(controller)

    # Add clients
    for i in range(NUM_CLIENTS):
        executor = CustomGTExecutor(data_chunk=i)
        job.to(executor, f"site-{i + 1}")

    # run
    job.export_job("./tmp/job_configs")
    job.simulator_run(f"./tmp/runs/{job_name}")

    plot_results(job_name, NUM_CLIENTS)
