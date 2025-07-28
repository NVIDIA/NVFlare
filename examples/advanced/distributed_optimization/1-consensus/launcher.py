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
import random

import matplotlib.pyplot as plt
import torch

from nvflare.app_opt.p2p.controllers import DistOptController
from nvflare.app_opt.p2p.executors import ConsensusExecutor
from nvflare.app_opt.p2p.types import Config
from nvflare.app_opt.p2p.utils.config_generator import generate_random_network
from nvflare.job_config.api import FedJob

if __name__ == "__main__":
    # Create job
    job = FedJob(name="consensus")

    # generate random config
    num_clients = 6
    network, _ = generate_random_network(num_clients=num_clients)
    config = Config(network=network, extra={"iterations": 50})

    # send controller to server
    controller = DistOptController(config=config)
    job.to_server(controller)

    # Add clients
    for i in range(num_clients):
        executor = ConsensusExecutor(random.randint(0, 10))
        job.to(executor, f"site-{i + 1}")

    # run
    job.export_job("./tmp/job_configs")
    job.simulator_run("./tmp/runs/consensus")

    history = {
        f"site-{i + 1}": torch.load(f"tmp/runs/consensus/site-{i + 1}/value_sequence.pt") for i in range(num_clients)
    }
    plt.figure()
    for i in range(num_clients):
        plt.plot(history[f"site-{i + 1}"], label=f"site-{i + 1}")
    plt.legend()
    plt.title("Evolution of local values")
    plt.show()
