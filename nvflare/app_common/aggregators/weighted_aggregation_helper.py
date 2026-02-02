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

import re
import threading
from typing import Optional


class WeightedAggregationHelper(object):
    def __init__(self, exclude_vars: Optional[str] = None, weigh_by_local_iter: bool = True):
        """Perform weighted aggregation.

        Args:
            exclude_vars (str, optional): regex string to match excluded vars during aggregation. Defaults to None.
            weigh_by_local_iter (bool, optional): Whether to weight the contributions by the number of iterations
                performed in local training in the current round. Defaults to `True`.
                Setting it to `False` can be useful in applications such as homomorphic encryption to reduce
                the number of computations on encrypted ciphertext.
                The aggregated sum will still be divided by the provided weights and `aggregation_weights` for the
                resulting weighted sum to be valid.
        """
        super().__init__()
        self.lock = threading.Lock()
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.weigh_by_local_iter = weigh_by_local_iter
        self.reset_stats()
        self.total = dict()
        self.counts = dict()
        self.history = list()

    def reset_stats(self):
        self.total = dict()
        self.counts = dict()
        self.history = list()

    @staticmethod
    def _is_pytorch_tensor(tensor):
        """Check if tensor is a PyTorch tensor with in-place operation support."""
        return hasattr(tensor, "add_") and hasattr(tensor, "mul_") and hasattr(tensor, "clone")

    def add(self, data, weight, contributor_name, contribution_round):
        """Compute weighted sum and sum of weights."""
        with self.lock:
            for k, v in data.items():
                if self.exclude_vars is not None and self.exclude_vars.search(k):
                    continue

                current_total = self.total.get(k, None)

                if current_total is None:
                    # First contribution: initialize accumulator
                    # Note: When weigh_by_local_iter=False, we store a reference (not a copy) to save memory.
                    # This is safe because no controller accesses client tensor data after aggregation.
                    # Controllers only access metadata (.params_type, .meta, .metrics), never .params.
                    # When weigh_by_local_iter=True, multiplication creates a new tensor/array anyway.
                    self.total[k] = v * weight if self.weigh_by_local_iter else v
                    self.counts[k] = weight
                else:
                    # Subsequent contributions: use in-place operations for PyTorch tensors
                    if self.weigh_by_local_iter:
                        # Weighted accumulation
                        if self._is_pytorch_tensor(v) and self._is_pytorch_tensor(current_total):
                            self.total[k].add_(v, alpha=weight)
                        else:
                            self.total[k] = current_total + v * weight
                    else:
                        # Unweighted accumulation
                        if self._is_pytorch_tensor(v) and self._is_pytorch_tensor(current_total):
                            self.total[k].add_(v)
                        else:
                            self.total[k] = current_total + v
                    self.counts[k] += weight

            self.history.append(
                {
                    "contributor_name": contributor_name,
                    "round": contribution_round,
                    "weight": weight,
                }
            )

    def get_result(self):
        """Divide weighted sum by sum of weights."""
        with self.lock:
            aggregated_dict = {}
            for k, v in self.total.items():
                if self._is_pytorch_tensor(v):
                    aggregated_dict[k] = v.div_(self.counts[k])
                else:
                    aggregated_dict[k] = v * (1.0 / self.counts[k])

            self.reset_stats()
            return aggregated_dict

    def get_history(self):
        return self.history

    def get_len(self):
        return len(self.get_history())
