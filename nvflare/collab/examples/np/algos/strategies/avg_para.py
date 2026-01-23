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
from nvflare.collab import fox
from nvflare.collab.examples.np.mains.utils import parse_array_def
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPFedAvgParallel:

    def __init__(self, initial_model, num_rounds=10):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self._initial_model = parse_array_def(initial_model)
        self.name = "NPFedAvgParallel"
        self.logger = get_obj_logger(self)

    @fox.main
    def execute(self):
        self.logger.info(f"[{fox.call_info}] Start training for {self.num_rounds} rounds")
        current_model = self._initial_model
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model)
            if current_model is None:
                self.logger.error(f"training failed at round {i}")
                break
            score = self._do_eval(current_model)
            self.logger.info(f"[{fox.call_info}]: eval score in round {i}: {score}")
        return current_model

    def _do_eval(self, model):
        results = fox.clients.evaluate(model)
        total = 0.0
        for n, v in results:
            self.logger.info(f"[{fox.call_info}]: got eval result from client {n}: {v}")
            total += v

        num_results = len(results)
        return total / len(results) if num_results > 0 else 0.0

    def _do_one_round(self, r, current_model):
        total = 0
        results = fox.clients(timeout=4, blocking=False, target="client").train(r, current_model)
        for n, v in results:
            # the value 'v' could be an exception!
            if isinstance(v, Exception):
                # this site encountered problem
                self.logger.error(f"[{fox.call_info}] round {r}: got exception from client {n}: {v}")
                raise v

            self.logger.info(f"[{fox.call_info}] round {r}: got group result from client {n}: {v}")
            total += v
        num_results = len(results)
        return total / len(results) if num_results > 0 else None
