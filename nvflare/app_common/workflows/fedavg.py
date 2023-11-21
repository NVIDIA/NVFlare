# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


class FedAvg(ModelController):
    def run(self) -> None:
        self.info("Start FedAvg.")

        for self._current_round in range(self._num_rounds):
            self.info(f"Round {self._current_round} started.")

            clients = self.sample_clients(self._min_clients)

            results = self.send_model_and_wait(targets=clients, data=self.model)

            aggregate_results = self.aggregate(
                results, aggregate_fn=None
            )  # if no `aggregate_fn` provided, default `WeightedAggregationHelper` is used

            self.update_model(aggregate_results)

            self.save_model()

        self.info("Finished FedAvg.")

    def finalize(self):
        pass
