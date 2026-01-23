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
from nvflare.collab.examples.pt.utils import add as add_pt
from nvflare.collab.examples.pt.utils import div as div_pt
from nvflare.collab.examples.pt.utils import parse_state_dict
from nvflare.collab.sim.simulator import Simulator
from nvflare.fuel.utils.log_utils import get_obj_logger


class PTFedAvg:

    def __init__(self, initial_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.timeout = timeout
        self.name = "PTFedAvg"
        self.logger = get_obj_logger(self)
        self._init_model = parse_state_dict(initial_model)

    @fox.algo
    def execute(self):
        self.logger.info(f"[{fox.call_info}] Start training for {self.num_rounds} rounds")
        current_model = self._init_model
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model)
            if current_model is None:
                self.logger.error(f"training failed at round {i}")
                break
        self.logger.info(f"FINAL MODEL: {current_model}")
        return current_model

    def _do_one_round(self, r, current_model):
        aggr_result = {}

        results = fox.clients(timeout=self.timeout).train(r, current_model)
        for n, v in results:
            add_pt(v, aggr_result)

        num_results = len(results)
        aggr_result = div_pt(aggr_result, num_results) if num_results > 0 else None
        self.logger.info(f"[{fox.call_info}] round {r}: aggr result from {num_results} clients: {aggr_result}")
        return aggr_result


class PTTrainer:

    def __init__(self, delta: float):
        self.delta = delta
        self.logger = get_obj_logger(self)

    @fox.collab
    def train(self, current_round, weights):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return None

        self.logger.debug(f"[{fox.call_info}] training round {current_round}: {weights=}")
        result = {}
        for k, v in weights.items():
            result[k] = v + self.delta
        return result


def main():
    simple_logging(logging.DEBUG)

    server = PTFedAvg(
        initial_model={
            "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        },
        num_rounds=4,
    )

    client = PTTrainer(delta=1.0)

    simulator = Simulator(
        root_dir=get_experiment_root(),
        experiment_name="pt_fedavg_intime",
        server=server,
        client=client,
        num_clients=2,
    )

    result = simulator.run()
    print(f"final result: {result}")


if __name__ == "__main__":
    main()
