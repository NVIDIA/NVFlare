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


from typing import Dict, Optional

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.math_utils import parse_compare_criteria
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs


class FedAvgV1(BaseFedAvg):
    def __init__(
        self,
        *args,
        stop_cond: str = None,
        initial_model=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.stop_cond = stop_cond
        if stop_cond:
            self.stop_condition = parse_compare_criteria(stop_cond)
        else:
            self.stop_condition = None

        self.initial_model = initial_model
        fobs.register(TensorDecomposer)

    def run(self) -> None:
        if self.initial_model:
            initial_weights = self.initial_model.state_dict()
        else:
            initial_weights = {}

        model = FLModel(params=initial_weights)

        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):

            self.info(f"Round {self.current_round} started.")

            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            results = self.send_model_and_wait(targets=clients, data=model)

            # using default aggregate_fn with `WeightedAggregationHelper`.
            # Can overwrite self.aggregate_fn with signature Callable[List[FLModel], FLModel]
            aggregate_results = self.aggregate(results, aggregate_fn=self.aggregate_fn)
            model = self.update_model(model, aggregate_results)

            self.info(f"Round {self.current_round} global metrics: {model.metrics}")

            if self.should_stop(model.metrics, self.stop_condition):
                self.info(
                    f"Stopping at round={self.current_round} out of total_rounds={self.num_rounds}. Early stop condition satisfied: {self.stop_condition}"
                )
                break

        self.info("Finished FedAvg.")

    def should_stop(self, metrics: Optional[Dict] = None, stop_condition: Optional[str] = None):
        if stop_condition is None or metrics is None:
            return False

        key, target, op_fn = stop_condition
        value = metrics.get(key, None)

        if value is None:
            raise RuntimeError(f"stop criteria key '{key}' doesn't exists in metrics")

        return op_fn(value, target)
