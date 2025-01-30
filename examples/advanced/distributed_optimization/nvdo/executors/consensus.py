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
import pickle
import random

from nvflare.apis.dxo import from_shareable

from .base import SynchronousAlgorithmExecutor


class ConsensusExecutor(SynchronousAlgorithmExecutor):
    """An executor that implements the consensus algorithm"""

    def __init__(
        self,
        initial_value: float | None = None,
    ):
        super().__init__()
        if initial_value is None:
            initial_value = random.random()
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
                current_value += (
                    self.neighbors_values[iteration][neighbor.id] * neighbor.weight
                )

            # 3. store current value
            self.current_value = current_value
            self.value_history.append(current_value)

            # free memory that's no longer needed
            del self.neighbors_values[iteration]

    def _post_algorithm_run(self, *args, **kwargs):
        pickle.dump(self.value_history, open("results.pkl", "wb"))
