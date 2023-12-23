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
import copy
from typing import Any

from nvflare.app_common.app_defined.aggregator import AppDefinedAggregator

from .av_model import META_IS_DIFF, AVModel


class AVAggregator(AppDefinedAggregator):
    def __init__(self):
        AppDefinedAggregator.__init__(self)
        self.accumulated_diff = {}
        self.current_meta = {}
        self.num_clients = 0

    def processing_training_result(self, client_name: str, trained_weights: Any, trained_meta: dict) -> bool:
        if not isinstance(trained_weights, AVModel):
            self.error(f"invalid result from client {client_name}: expect AVModel but got {type(trained_weights)}")
            return False

        trained_weights = trained_weights.free_layers
        free_layers = self.base_model_obj.free_layers
        self.info(f"Round {self.current_round}: received result from client {client_name}: {trained_weights}")
        self.info(f"Round {self.current_round}: current base: {free_layers}")
        self.info(f"Round {self.current_round}: current diff: {self.accumulated_diff}")

        if not trained_meta.get(META_IS_DIFF):
            # compute weight diff from base model
            assert isinstance(self.base_model_obj, AVModel)
            for k, v in trained_weights.items():
                for i, w in enumerate(v):
                    v[i] -= free_layers[k][i]

        for k, v in trained_weights.items():
            if k in self.accumulated_diff:
                for i, w in enumerate(v):
                    self.accumulated_diff[k][i] += w
            else:
                self.accumulated_diff[k] = copy.deepcopy(trained_weights[k])

        self.current_meta.update(trained_meta)
        self.num_clients += 1
        self.info(f"accumulated diffs after client {self.num_clients}: {self.accumulated_diff}")
        return True

    def reset(self):
        self.accumulated_diff = {}
        self.current_meta = {}
        self.num_clients = 0

    def aggregate_training_result(self) -> (Any, dict):
        result = self.accumulated_diff
        meta = self.current_meta
        if self.num_clients > 0:
            for k, v in result.items():
                for i, w in enumerate(v):
                    v[i] = w / self.num_clients
        self.info(f"Round {self.current_round} aggregated diff: {result}")
        meta[META_IS_DIFF] = True
        return AVModel({}, {}, result), meta
