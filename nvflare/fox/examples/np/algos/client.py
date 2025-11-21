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

from nvflare.fox import fox
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPTrainer:

    def __init__(self, delta: float):
        self.delta = delta
        self.logger = get_obj_logger(self)

    @fox.init
    def init_trainer(self):
        delta_config = fox.get_app_prop("client_delta", {})
        self.delta = delta_config.get(fox.site_name, self.delta)
        self.logger.info(f"init_trainer: client {fox.site_name}: delta={self.delta}")

    @fox.init
    def init_trainer2(self):
        self.logger.info(f"init_trainer2: client {fox.site_name}: init again")

    @fox.collab
    def train(self, current_round, weights):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return 0
        self.logger.info(f"[{fox.call_info}] EZ trained round {current_round=} {weights=}")

        # metric_receiver = self.server.get_target("metric_receiver")
        # if metric_receiver:
        #     self.server.accept_metric({"round": r, "y": 2})
        #     self.server.metric_receiver.accept_metric({"round": r, "y": 2})
        #
        self.logger.info(f"before fire_event: fox ctx={id(fox.context)}")
        fox.server(blocking=False).fire_event("metrics", {"round": current_round, "y": 10})
        self.logger.info(f"after fire_event: fox ctx={id(fox.context)}")
        return weights + self.delta

    @fox.collab
    def evaluate(self, model):
        self.logger.debug(f"[{fox.call_info}] evaluate")
        return random.random()


class NPHierarchicalTrainer:

    def __init__(self, delta: float):
        self.delta = delta
        self.logger = get_obj_logger(self)

    @fox.collab
    def train(self, current_round, weights):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return 0

        self.logger.debug(f"[{fox.call_info}] training round {current_round}")
        if fox.has_children:
            total = 0
            results = fox.child_clients.train(current_round, weights)
            for n, v in results.items():
                total += v
            result = total / len(results)
            self.logger.debug(f"[{fox.call_info}]: aggr result from children of round {current_round}: {result}")
        else:
            result = self._local_train(current_round, weights)
            self.logger.debug(f"[{fox.call_info}]: local train result of round {current_round}: {result}")
            fox.server.fire_event("metrics", {"round": current_round, "y": 10}, _blocking=False)
        return result

    def _local_train(self, current_round, weights):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return 0
        self.logger.info(f"[{fox.call_info}] local trained round {current_round} {weights} {type(weights)}")
        return weights + self.delta

    @fox.collab
    def evaluate(self, model):
        self.logger.debug(f"[{fox.call_info}] evaluate")
        return random.random()
