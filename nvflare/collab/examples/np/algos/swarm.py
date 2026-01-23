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
import threading
import traceback

from nvflare.collab import fox
from nvflare.collab.examples.np.mains.utils import parse_array_def, save_np_model
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPSwarm:

    def __init__(self, initial_model, num_rounds=10):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self._initial_model = parse_array_def(initial_model)
        self.waiter = threading.Event()
        self.logger = get_obj_logger(self)

    @fox.main
    def execute(self):
        fox.register_event_handler("all_done", self._all_done)

        # randomly pick a client to start
        start_client_idx = random.randint(0, len(fox.clients) - 1)
        start_client = fox.clients[start_client_idx]
        start_client(target="client").start(self.num_rounds, self._initial_model)
        while not fox.is_aborted:
            if self.waiter.wait(timeout=0.5):
                break

    def _all_done(self, event_type: str, data):
        self.logger.info(f"[{fox.call_info}]: received {event_type} from client: {fox.caller}: {data}")
        self.all_done(data)

    @fox.publish
    def all_done(self, reason: str):
        self.logger.info(f"[{fox.call_info}]: all done from client: {fox.caller}: {reason}")
        self.waiter.set()


class NPSwarmClient:

    def __init__(self, delta: float):
        self.delta = delta
        self.logger = get_obj_logger(self)

    @fox.init
    def init(self):
        # This example shows that there could be multiple listeners for the same event
        fox.register_event_handler("final_model", self._accept_final_model)
        fox.register_event_handler("final_model", self._save_final_model)

    @fox.publish
    def train(self, weights, current_round):
        self.logger.info(f"[{fox.call_info}]: train asked by {fox.caller}: {current_round=}")
        return weights + self.delta

    def sag(self, model, current_round):
        # results = fox.clients.train(model, current_round)
        results = fox.other_clients.train(model, current_round)
        total = 0
        for n, v in results:
            total += v
        return total / len(results)

    @fox.publish
    def swarm_learn(self, num_rounds, model, current_round):
        self.logger.info(f"[{fox.call_info}]: swarm learn asked: {num_rounds=} {current_round=} {model=}")
        new_model = self.sag(model, current_round)

        self.logger.info(f"[{fox.call_info}]: trained model {new_model=}")
        if current_round == num_rounds - 1:
            # all done
            result = fox.clients(expect_result=True).fire_event("final_model", new_model)
            for n, v in result:
                self.logger.info(f"[{fox.call_info}] final_model reply from {n}: {v}")
            self.logger.info("notify server all done!")
            try:
                fox.server(expect_result=False).all_done("OK")
            except Exception as ex:
                traceback.print_exc()
                self.logger.error(f"exception occurred in learning: {type(ex)}")
            self.logger.info("Swarm Training is DONE!")
            return

        # determine next client
        next_round = current_round + 1
        next_client_idx = random.randint(0, len(fox.clients) - 1)
        self.logger.debug(f"chose aggr client for round {next_round}: {next_client_idx}")
        next_client = fox.clients[next_client_idx]
        next_client(expect_result=False).swarm_learn(num_rounds, new_model, next_round)

    @fox.publish
    def start(self, num_rounds, initial_model):
        self.logger.info(f"[{fox.call_info}]: starting swarm learning")
        self.swarm_learn(num_rounds, initial_model, 0)

    def _accept_final_model(self, event_type: str, model):
        # accept the final model
        # write model to disk
        self.logger.info(f"[{fox.call_info}]: received event '{event_type}' from {fox.caller}: {model}")
        return "received"

    def _save_final_model(self, event_type: str, model):
        # accept the final model
        # write model to disk
        file_name = os.path.join(fox.workspace.get_work_dir(), "final_model.npy")
        save_np_model(model, file_name)
        self.logger.info(f"[{fox.call_info}]: saved model {model} to {file_name}")
        return "saved"
