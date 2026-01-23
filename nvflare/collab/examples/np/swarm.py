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

from nvflare.collab.api.utils import simple_logging
from nvflare.collab.examples import get_experiment_root
from nvflare.collab.examples.np.mains.swarm import NPSwarm, NPSwarmClient
from nvflare.collab.sim.simulator import Simulator


def main():
    simple_logging(logging.DEBUG)

    simulator = Simulator(
        root_dir=get_experiment_root(),
        experiment_name="swarm",
        server=NPSwarm(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], num_rounds=5),
        client=NPSwarmClient(delta=1.0),
        num_clients=3,
    )

    result = simulator.run()
    print(f"Final result: {result}")


if __name__ == "__main__":
    main()
