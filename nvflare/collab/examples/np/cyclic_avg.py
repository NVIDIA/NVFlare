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

from nvflare.collab import fox
from nvflare.collab.api.utils import simple_logging
from nvflare.collab.examples import get_experiment_root
from nvflare.collab.examples.np.mains.client import NPTrainer
from nvflare.collab.examples.np.mains.strategies.avg_para import NPFedAvgParallel
from nvflare.collab.examples.np.mains.strategies.cyclic import NPCyclic
from nvflare.collab.sim.simulator import Simulator
from nvflare.fuel.utils.log_utils import get_obj_logger


class Controller:

    def __init__(
        self,
        initial_model,
        cyclic_rounds,
        avg_rounds,
    ):
        self.initial_model = initial_model
        self.cyclic_rounds = cyclic_rounds
        self.avg_rounds = avg_rounds
        self.logger = get_obj_logger(self)

    @fox.main
    def run(self):
        self.logger.info("running cyclic ...")
        ctl = NPCyclic(self.initial_model, num_rounds=self.cyclic_rounds)
        result = ctl.execute()
        self.logger.info(f"final cyclic model: {result}")

        self.logger.info("running fed-avg ...")
        ctl = NPFedAvgParallel(initial_model=result, num_rounds=self.avg_rounds)
        result = ctl.execute()
        self.logger.info(f"final model: {result}")
        return result


def main():
    simple_logging(logging.DEBUG)
    exp_name = "cyclic_avg"

    server = Controller(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cyclic_rounds=2, avg_rounds=3)

    simulator = Simulator(
        root_dir=get_experiment_root(),
        experiment_name=exp_name,
        server=server,
        client=NPTrainer(delta=1.0),
        num_clients=3,
    )

    final_result = simulator.run()
    print(f"final model: {final_result}")


if __name__ == "__main__":
    main()
