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

from collab.workflow_composition.np_utils import parse_array_def

from nvflare.collab import CollabCallError, collab
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPCyclic:

    def __init__(self, initial_model, num_rounds=2):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self._initial_model = parse_array_def(initial_model)
        self.final_model = None
        self.logger = get_obj_logger(self)

    @collab.main
    def execute(self):
        current_model = self._initial_model
        for current_round in range(self.num_rounds):
            try:
                current_model = self._do_one_round(current_round, current_model)
            except CollabCallError as error:
                self.logger.error(f"round {current_round} failed on {error.site}: {error.cause}")
                return None
        self.logger.info(f"[{collab.call_info}] final result: {current_model}")
        self.final_model = current_model
        return current_model

    def _do_one_round(self, current_round, current_model):
        clients = collab.clients
        random.shuffle(clients)
        for client in clients:
            current_model = client.train(current_round, current_model)
            self.logger.info(f"[{collab.call_info}] result from {client.name}: {current_model}")
        return current_model
