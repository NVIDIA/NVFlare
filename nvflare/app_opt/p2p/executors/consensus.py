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

import torch

from nvflare.apis.dxo import from_shareable
from nvflare.app_opt.p2p.executors.sync_executor import SyncAlgorithmExecutor


class ConsensusExecutor(SyncAlgorithmExecutor):
    """An executor that implements a consensus algorithm in a peer-to-peer (P2P) setup.

    This executor extends the SyncAlgorithmExecutor to implement a simple consensus algorithm.
    The client starts with an initial value and iteratively exchanges values with its neighbors.
    At each iteration, the client updates its current value based on its own value and the weighted sum
    of its neighbors' values. The process continues for a specified number of iterations, and the history
    of values is saved at the end of the run.

    The number of iterations must be provided by the controller when asing to run the algorithm. It can
    be set in the extra parameters of the controller's config with the "iterations" key.

    Args:
        initial_value (float, optional): The initial value for the consensus algorithm.
            If not provided, a random value between 0 and 1 is used.

    Attributes:
        current_value (float): The current value of the client in the consensus algorithm.
        value_history (list[float]): A list storing the history of values over iterations.
    """

    def __init__(
        self,
        initial_value: float | None = None,
    ):
        super().__init__()
        if initial_value is None:
            initial_value = random.random()
        self.initial_value = initial_value
        self.current_value = initial_value
        self.value_history = [self.current_value]

    def run_algorithm(self, fl_ctx, shareable, abort_signal):
        iterations = from_shareable(shareable).data["iterations"]

        for iteration in range(iterations):
            if abort_signal.triggered:
                break

            # run algorithm step
            # 1. exchange values
            self._exchange_values(fl_ctx, value=self.current_value, iteration=iteration)

            # 2. compute new value
            current_value = self.current_value * self._weight
            for neighbor in self.neighbors:
                current_value += self.neighbors_values[iteration][neighbor.id] * neighbor.weight

            # 3. store current value
            self.current_value = current_value
            self.value_history.append(current_value)

            # free memory that's no longer needed
            del self.neighbors_values[iteration]

    def _post_algorithm_run(self, *args, **kwargs):
        torch.save(torch.tensor(self.value_history), "value_sequence.pt")
