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
import logging

from nvflare.fox.api.app import ClientApp, ServerApp
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.np.algos.client import NPTrainer
from nvflare.fox.examples.np.algos.strategies.avg_seq import NPFedAvgSequential
from nvflare.fox.examples.np.algos.widgets import MetricReceiver
from nvflare.fox.sim.simulator import Simulator


def main():
    simple_logging(logging.DEBUG)

    server_app = ServerApp(
        NPFedAvgSequential(
            num_rounds=2,
            initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ),
    )
    server_app.add_collab_object("metric_receiver", MetricReceiver())
    server_app.set_prop(
        "client_weight_config",
        {
            "site-1": 70,
            "site-2": 100,
        },
    )

    client_app = ClientApp(NPTrainer(delta=1.0))
    client_app.set_prop(
        "client_delta",
        {
            "site-1": 1.0,
            "site-2": 2.0,
        },
    )
    simulator = Simulator(
        root_dir="/tmp/fox",
        experiment_name="fedavg_seq",
        server_app=server_app,
        client_app=client_app,
        num_clients=2,
    )

    result = simulator.run()
    print(f"Final result: {result}")


if __name__ == "__main__":
    main()
