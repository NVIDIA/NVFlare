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
import os
import random

from nvflare.collab import collab
from nvflare.collab.examples.np.mains.utils import load_np_model, parse_array_def, save_np_model
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPCyclic:

    def __init__(self, initial_model, num_rounds=2):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self._initial_model = parse_array_def(initial_model)
        self.final_model = None
        self.logger = get_obj_logger(self)

    @collab.init
    def check_initial_model(self):
        if isinstance(self._initial_model, str):
            # this is name of the file that contains model data
            # load the model.
            resource_dir = collab.workspace.get_resource_dir("data")
            file_name = os.path.join(resource_dir, self._initial_model)
            self._initial_model = load_np_model(file_name)
            self.logger.info(f"loaded initial model from {file_name}: {self._initial_model}")

    @collab.main
    def execute(self):
        current_model = self._initial_model
        for current_round in range(self.num_rounds):
            current_model = self._do_one_round(current_round, current_model)
            if current_model is None:
                self.logger.error(f"training failed at round {current_round}")
                break
        self.logger.info(f"[{collab.call_info}] final result: {current_model}")
        self.final_model = current_model
        return current_model

    @collab.final
    def save_result(self):
        final_result = collab.get_result()
        file_name = os.path.join(collab.workspace.get_work_dir(), "final_model.npy")
        save_np_model(final_result, file_name)
        self.logger.info(f"[{collab.call_info}]: saved final model {final_result} to {file_name}")

    def _do_one_round(self, current_round, current_model):
        # Note: collab.clients always returns a new copy of all clients!
        clients = collab.clients
        random.shuffle(clients)
        for c in clients:
            current_model = c.train(current_round, current_model)
            if current_model is None:
                self.logger.error(f"training failed on client {c.name} at round {current_round}")
                return None
            self.logger.info(f"[{collab.call_info}] result from {c.name}: {current_model}")
        return current_model
