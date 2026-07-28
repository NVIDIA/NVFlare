# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPTrainer:

    def __init__(self, delta: float, delay=0):
        self.delta = delta
        self.delay = delay
        self.logger = get_obj_logger(self)

    @collab.init
    def init_trainer(self):
        delta_config = collab.get_app_prop("client_delta", {})
        self.delta = delta_config.get(collab.site_name, self.delta)
        self.logger.info(f"init_trainer: client {collab.site_name}: delta={self.delta}")

    @collab.init
    def init_trainer2(self):
        self.logger.info(f"init_trainer2: client {collab.site_name}: init again")

    @collab.publish
    def train(self, current_round, weights):
        if collab.is_aborted:
            self.logger.debug("training aborted")
            return 0

        self.logger.info(f"[{collab.call_info}] training round {current_round=} {weights=}")
        if self.delay > 0:
            time.sleep(self.delay)
        return weights + self.delta

    @collab.publish
    def evaluate(self, model):
        self.logger.debug(f"[{collab.call_info}] evaluate")
        return random.random()
