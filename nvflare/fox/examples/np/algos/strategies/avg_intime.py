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
from nvflare.fox import fox
from nvflare.fox.api.constants import ContextKey
from nvflare.fox.examples.np.algos.utils import parse_array_def
from nvflare.fuel.utils.log_utils import get_obj_logger


class _AggrResult:

    def __init__(self):
        self.total = 0
        self.count = 0


class NPFedAvgInTime:

    def __init__(self, initial_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.timeout = timeout
        self.name = "NPFedAvgInTime"
        self.logger = get_obj_logger(self)
        self._init_model = parse_array_def(initial_model)

    @fox.algo
    def execute(self):
        self.logger.info(f"[{fox.call_info}] Start training for {self.num_rounds} rounds")
        current_model = fox.get_prop(ContextKey.INPUT, self._init_model)
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model)
            # current_model = self._do_one_round_non_blocking(i, current_model)
            score = self._do_eval(current_model)
            self.logger.info(f"[{fox.call_info}]: eval score in round {i}: {score}")
        self.logger.info(f"FINAL MODEL: {current_model}")
        return current_model

    def _do_eval(self, model):
        results = fox.clients.evaluate(model)
        total = 0.0
        for n, v in results:
            self.logger.info(f"[{fox.call_info}]: got eval result from client {n}: {v}")
            total += v
        return total / len(results)

    def _do_one_round(self, r, current_model):
        aggr_result = _AggrResult()
        timeout = fox.get_app_prop("default_timeout", 10)
        self.logger.info(f"got timeout: {timeout}")
        fox.clients(
            timeout=timeout,
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        ).train(r, current_model)

        if aggr_result.count == 0:
            return None
        else:
            result = aggr_result.total / aggr_result.count
            self.logger.info(f"[{fox.call_info}] round {r}: aggr result from {aggr_result.count} clients: {result}")
            return result

    def _do_one_round_non_blocking(self, r, current_model):
        timeout = fox.get_app_prop("default_timeout", 10)
        self.logger.info(f"got timeout: {timeout}")
        results = fox.clients(
            timeout=timeout,
            blocking=False,
        ).train(r, current_model)

        total = 0
        for n, v in results:
            self.logger.info(f"[{fox.call_info}] round {r}: got group result from client {n}: {v}")
            total += v
        result = total / len(results)
        self.logger.info(f"[{fox.call_info}] round {r}: aggr result from {len(results)} clients: {result}")
        return result

    def _accept_train_result(self, result, aggr_result: _AggrResult):
        self.logger.info(f"[{fox.call_info}] got train result from {fox.caller} {result}")
        aggr_result.total += result
        aggr_result.count += 1
        return None
