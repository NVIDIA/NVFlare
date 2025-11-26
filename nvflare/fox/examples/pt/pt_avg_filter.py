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
import threading

import torch

from nvflare.fox import fox
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.pt.utils import parse_state_dict
from nvflare.fox.sim.simulator import Simulator
from nvflare.fuel.utils.log_utils import get_obj_logger


class _AggrResult:

    def __init__(self):
        self.total = {}
        self.count = 0
        self.lock = threading.Lock()  # ensure update integrity


class PTFedAvg:

    def __init__(self, initial_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.timeout = timeout
        self.name = "PTFedAvgStream"
        self.logger = get_obj_logger(self)
        self._init_model = parse_state_dict(initial_model)

    @fox.algo
    def execute(self):
        self.logger.info(f"[{fox.call_info}] Start training for {self.num_rounds} rounds")
        current_model = fox.get_input(self._init_model)
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model)
        self.logger.info(f"FINAL MODEL: {current_model}")
        return current_model

    def _do_one_round(self, r, current_model):
        aggr_result = _AggrResult()

        fox.clients(
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        ).train(r, current_model)

        if aggr_result.count == 0:
            return None
        else:
            result = {}
            for k, v in aggr_result.total.items():
                result[k] = torch.div(v, aggr_result.count)
            self.logger.info(f"[{fox.call_info}] round {r}: aggr result from {aggr_result.count} clients: {result}")
            return result

    def _accept_train_result(self, result, aggr_result: _AggrResult):
        self.logger.info(f"[{fox.call_info}] got train result from {fox.caller}: {result}")

        for k, v in result.items():
            if k not in aggr_result.total:
                aggr_result.total[k] = v
            else:
                aggr_result.total[k] += v

        aggr_result.count += 1
        return None


class PTTrainer:

    def __init__(self, delta: float):
        self.delta = delta
        self.logger = get_obj_logger(self)

    @fox.collab
    def train(self, current_round, weights):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return 0
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
        root_dir="/tmp/fox",
        experiment_name="pt_fedavg_intime",
        server=server,
        client=client,
        num_clients=2,
    )

    result = simulator.run()
    print(f"final result: {result}")


if __name__ == "__main__":
    main()
