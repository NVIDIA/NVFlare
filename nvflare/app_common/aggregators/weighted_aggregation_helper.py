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
from typing import Any, Callable, Dict, Optional, Set


def _is_aggregatable_metric_value(v: Any) -> bool:
    """Return True if the metric value supports weighted aggregation (v * weight and addition)."""
    if v is None:
        return False
    if isinstance(v, (dict, list, set, tuple, str)):
        return False
    if isinstance(v, (int, float, bool)):
        return True
    try:
        _ = v * 1.0
        _ = v + v
        return True
    except Exception:
        return False


def filter_aggregatable_metrics(
    metrics: Optional[Dict[str, Any]],
    warn_skipped: Optional[Callable[[str, str], None]] = None,
    warned_metric_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Return metric entries that support weighted aggregation.

    Args:
        metrics: Dict of metric name -> value.
        warn_skipped: Optional callback invoked as warn_skipped(key, type_name) for skipped metrics.
        warned_metric_keys: Optional set of keys already warned about. If provided, warnings are emitted
            at most once per key and newly warned keys are added to this set.
    """
    if not metrics:
        return {}

    filtered = {}
    for key, value in metrics.items():
        if _is_aggregatable_metric_value(value):
            filtered[key] = value
            continue
        if warn_skipped is None:
            continue
        if warned_metric_keys is None or key not in warned_metric_keys:
            warn_skipped(key, type(value).__name__)
            if warned_metric_keys is not None:
                warned_metric_keys.add(key)
    return filtered


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
                    # We must create a copy to avoid mutating caller's input tensors
                    if self._is_pytorch_tensor(v):
                        if self.weigh_by_local_iter:
                            # Weigh by local iter: create weighted copy (multiply by weight)
                            self.total[k] = v.mul(weight)
                        else:
                            self.total[k] = v.clone()
                    else:
                        # Fallback for non-PyTorch tensors
                        if self.weigh_by_local_iter:
                            # Multiply creates a new array/tensor, no aliasing issue
                            self.total[k] = v * weight
                        else:
                            # For HE mode: try to copy to avoid aliasing
                            # But encrypted tensors can't be copied (requires secret key)
                            try:
                                self.total[k] = v.copy() if hasattr(v, "copy") else v
                            except (ValueError, RuntimeError):
                                # Encrypted tensor copy failed, use reference (safe, immutable)
                                self.total[k] = v
                    self.counts[k] = weight
                else:
                    # Subsequent contributions: use in-place operations
                    if self._is_pytorch_tensor(v) and self._is_pytorch_tensor(current_total):
                        if self.weigh_by_local_iter:
                            # Weigh by local iter: weighted accumulation
                            self.total[k].add_(v, alpha=weight)
                        else:
                            self.total[k].add_(v)
                    else:
                        # Fallback for non-PyTorch tensors
                        if self.weigh_by_local_iter:
                            self.total[k] = current_total + v * weight
                        else:
                            self.total[k] = current_total + v
                    self.counts[k] = self.counts[k] + weight

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
                    # For PyTorch tensors, use in-place division to avoid creating a copy
                    aggregated_dict[k] = v.div_(self.counts[k])
                else:
                    # Fallback for non-PyTorch tensors (including encrypted tensors)
                    aggregated_dict[k] = v * (1.0 / self.counts[k])

            self.reset_stats()
            return aggregated_dict

    def get_history(self):
        return self.history

    def get_len(self):
        return len(self.get_history())
