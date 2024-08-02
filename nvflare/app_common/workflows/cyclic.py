# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from .model_controller import ModelController


class Cyclic(ModelController):
    def __init__(
        self,
        *args,
        num_clients: int = 2,
        num_rounds: int = 5,
        start_round: int = 0,
        **kwargs,
    ):
        """The Cyclic ModelController to implement the Cyclic Weight Transfer (CWT) algorithm.

        Args:
            num_clients (int, optional): The number of clients. Defaults to 2.
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            start_round (int, optional): The starting round number. Defaults to 0
        """
        super().__init__(*args, **kwargs)

        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.start_round = start_round
        self.current_round = None

    def run(self) -> None:
        self.info("Start Cyclic.")

        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            for client in clients:
                result = self.send_model_and_wait(targets=[client], data=model)[0]
                model.params, model.meta = result.params, result.meta

            self.save_model(model)

        self.info("Finished Cyclic.")
