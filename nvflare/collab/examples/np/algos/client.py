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
import time

from nvflare.collab import fox
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPTrainer:

    def __init__(self, delta: float, delay=0):
        self.delta = delta
        self.delay = delay
        self.logger = get_obj_logger(self)

    @fox.init
    def init_trainer(self):
        delta_config = fox.get_app_prop("client_delta", {})
        self.delta = delta_config.get(fox.site_name, self.delta)
        self.logger.info(f"init_trainer: client {fox.site_name}: delta={self.delta}")

    @fox.init
    def init_trainer2(self):
        self.logger.info(f"init_trainer2: client {fox.site_name}: init again")

    @fox.publish
    def train(self, current_round, weights):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return 0
        self.logger.info(f"[{fox.call_info}] training round {current_round=} {weights=}")
        # result = fox.server(expect_result=True).fire_event("metrics", {"round": current_round, "y": 10})
        # self.logger.info(f"[{fox.call_info}] got event result: {result}")

        if self.delay > 0:
            time.sleep(self.delay)
        return weights + self.delta

    @fox.publish
    def evaluate(self, model):
        self.logger.debug(f"[{fox.call_info}] evaluate")
        return random.random()


class NPHierarchicalTrainer:

    def __init__(self, delta: float):
        self.delta = delta
        self.logger = get_obj_logger(self)

    @fox.publish
    def train(self, current_round, weights):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return None

        self.logger.debug(f"[{fox.call_info}] training round {current_round}")
        if fox.has_children:
            total = 0
            results = fox.child_clients.train(current_round, weights)
            for n, v in results:
                total += v
            result = total / len(results)
            self.logger.debug(f"[{fox.call_info}]: aggr result from children of round {current_round}: {result}")
        else:
            result = self._local_train(current_round, weights)
            self.logger.debug(f"[{fox.call_info}]: local train result of round {current_round}: {result}")
            fox.server(expect_result=False).fire_event("metrics", {"round": current_round, "y": 10})
        return result

    def _local_train(self, current_round, weights):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return None
        self.logger.info(f"[{fox.call_info}] local trained round {current_round} {weights} {type(weights)}")
        return weights + self.delta

    @fox.publish
    def evaluate(self, model):
        self.logger.debug(f"[{fox.call_info}] evaluate")
        return random.random()
