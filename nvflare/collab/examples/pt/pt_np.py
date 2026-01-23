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

from nvflare.collab import fox
from nvflare.collab.api.utils import simple_logging
from nvflare.collab.examples import get_experiment_root
from nvflare.collab.examples.np.algos.utils import add as add_np
from nvflare.collab.examples.np.algos.utils import div as div_np
from nvflare.collab.examples.np.algos.utils import parse_state_dict as parse_np
from nvflare.collab.examples.pt.utils import add as add_pt
from nvflare.collab.examples.pt.utils import div as div_pt
from nvflare.collab.examples.pt.utils import parse_state_dict as parse_pt
from nvflare.collab.sim.simulator import Simulator
from nvflare.fuel.utils.log_utils import get_obj_logger


class _AggrResult:

    def __init__(self):
        self.pt_total = {}
        self.np_total = {}
        self.count = 0
        self.lock = threading.Lock()  # ensure update integrity


class PTFedAvgMixed:

    def __init__(self, pt_model, np_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.pt_model = pt_model
        self.np_model = np_model
        self.timeout = timeout
        self.name = "PTFedAvg"
        self.logger = get_obj_logger(self)
        self._pt_model = parse_pt(pt_model)
        self._np_model = parse_np(np_model)

    @fox.algo
    def execute(self):
        self.logger.info(f"[{fox.call_info}] Start training for {self.num_rounds} rounds")
        pt_model, np_model = self._pt_model, self._np_model
        for i in range(self.num_rounds):
            pt_model, np_model = self._do_one_round(i, pt_model, np_model)
            if pt_model is None or np_model is None:
                self.logger.error(f"training failed at round {i}")
                break
        self.logger.info(f"FINAL MODEL: {pt_model=} {np_model=}")
        return pt_model, np_model

    def _do_one_round(self, r, pt_model, np_model):
        aggr_result = _AggrResult()

        fox.clients(
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        ).train(r, pt_model, np_model)

        if aggr_result.count == 0:
            return None, None
        else:
            pt_result = aggr_result.pt_total
            div_pt(pt_result, aggr_result.count)
            self.logger.info(
                f"[{fox.call_info}] round {r}: aggr PT result from {aggr_result.count} clients: {pt_result}"
            )

            np_result = aggr_result.np_total
            div_np(np_result, aggr_result.count)
            self.logger.info(
                f"[{fox.call_info}] round {r}: aggr NP result from {aggr_result.count} clients: {np_result}"
            )
            return pt_result, np_result

    def _accept_train_result(self, gcc, result, aggr_result: _AggrResult):
        self.logger.info(f"[{fox.call_info}] got train result from {fox.caller}: {result}")

        pt_result, np_result = result
        with aggr_result.lock:
            add_pt(pt_result, aggr_result.pt_total)
            add_np(np_result, aggr_result.np_total)
            aggr_result.count += 1
        return None


class PTTrainer:

    def __init__(self, delta: float):
        self.delta = delta
        self.logger = get_obj_logger(self)

    @fox.collab
    def train(self, current_round, pt_model, np_model):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return None, None

        self.logger.debug(f"[{fox.call_info}] training round {current_round}: {pt_model=} {np_model=}")

        pt_result = {}
        for k, v in pt_model.items():
            pt_result[k] = v + self.delta

        np_result = {}
        for k, v in np_model.items():
            np_result[k] = v + self.delta

        return pt_result, np_result


def main():
    simple_logging(logging.DEBUG)

    init_model = {
        "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }

    server = PTFedAvgMixed(
        pt_model=init_model,
        np_model=init_model,
        num_rounds=4,
    )

    simulator = Simulator(
        root_dir=get_experiment_root(),
        experiment_name="pt_np",
        server=server,
        client=PTTrainer(delta=1.0),
        num_clients=2,
    )

    result = simulator.run()
    print(f"Final result: {result}")


if __name__ == "__main__":
    main()
