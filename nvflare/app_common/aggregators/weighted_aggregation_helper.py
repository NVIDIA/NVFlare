# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional


class WeightedAggregationHelper(object):
    def __init__(self, exclude_vars: Optional[str] = None):
        """Perform weighted aggregation.

        Args:
            exclude_vars (str, optional): regex string to match excluded vars during aggregation. Defaults to None.
        """
        super().__init__()
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.reset_stats()
        self.total = dict()
        self.counts = dict()
        self.history = list()

    def reset_stats(self):
        self.total = {}
        self.counts = {}
        self.history = []

    def add(self, data, weight, contributor_name, contribution_round):
        """Compute weighted sum and sum of weights."""
        for k, v in data.items():
            if self.exclude_vars is not None and self.exclude_vars.search(k):
                continue
            weighted_value = v * weight
            current_total = self.total.get(k, None)
            if current_total is None:
                self.total[k] = weighted_value
                self.counts[k] = weight
            else:
                self.total[k] = current_total + weighted_value
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
        aggregated_dict = {k: v / self.counts[k] for k, v in self.total.items()}
        self.reset_stats()
        return aggregated_dict

    def get_history(self):
        return self.history

    def get_len(self):
        return len(self.get_history())
