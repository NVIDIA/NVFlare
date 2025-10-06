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

from nvflare.fox.api.app import ServerApp
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.np.algos.swarm import NPSwarm, NPSwarmClient
from nvflare.fox.sim.simulator import Simulator


def main():
    simple_logging(logging.DEBUG)

    server_app = ServerApp(
        strategy_name="strategy", strategy=NPSwarm(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], num_rounds=5)
    )
    client_app = NPSwarmClient(delta=1.0)

    simulator = Simulator(
        server_app=server_app,
        client_app=client_app,
        num_clients=3,
    )

    simulator.run()


if __name__ == "__main__":
    main()
