# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


from nvflare.app_common.utils.math_utils import parse_compare_criteria
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg


class FedAvgV1(BaseFedAvg):
    """FedAvg with Early Stopping

    Args:
        num_clients (int, optional): The number of clients. Defaults to 3.
        num_rounds (int, optional): The total number of training rounds. Defaults to 5.
        stop_cond (str, optional): early stopping condition based on metric.
            string literal in the format of "<key> <op> <value>" (e.g. "accuracy >= 80")
    """

    def __init__(
        self,
        *args,
        stop_cond: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.stop_cond = stop_cond
        if stop_cond:
            self.stop_condition = parse_compare_criteria(stop_cond)
        else:
            self.stop_condition = None

    def run(self) -> None:

        # self.info("Start FedAvg v1.")

        # if self.initial_model:
        #     # Use FOBS for serializing/deserializing PyTorch tensors (self.initial_model)
        #     fobs.register(TensorDecomposer)
        #     # PyTorch weights
        #     initial_weights = self.initial_model.state_dict()
        # else:
        #     initial_weights = {}

        # model = FLModel(params=initial_weights)

        # model.start_round = self.start_round
        # model.total_rounds = self.num_rounds

        # for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
        #     self.info(f"Round {self.current_round} started.")
        #     model.current_round = self.current_round

        #     clients = self.sample_clients(self.num_clients)

        #     results = self.send_model_and_wait(targets=clients, data=model)

        #     aggregate_results = self.aggregate(
        #         results, aggregate_fn=self.aggregate_fn
        #     )  # using default aggregate_fn with `WeightedAggregationHelper`. Can overwrite self.aggregate_fn with signature Callable[List[FLModel], FLModel]

        #     model = self.update_model(model, aggregate_results)

        #     self.info(f"Round {self.current_round} global metrics: {model.metrics}")

        #     self.select_best_model(model)

        #     self.save_model(self.best_model, os.path.join(os.getcwd(), self.save_filename))

        #     if self.should_stop(model.metrics, self.stop_condition):
        #         self.info(
        #             f"Stopping at round={self.current_round} out of total_rounds={self.num_rounds}. Early stop condition satisfied: {self.stop_condition}"
        #         )
        #         break

        self.info("Finished FedAvg.")
