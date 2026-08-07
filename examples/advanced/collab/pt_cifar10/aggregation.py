# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Aggregation shared by the synchronous Collab examples."""

from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper


def aggregate_result(client_results, result_key: str, round_number: int):
    """Aggregate one tensor dictionary from each successful client result."""

    helper = WeightedAggregationHelper()
    for site_name, result in dict(client_results).items():
        helper.add(
            data=result[result_key],
            weight=result["num_steps"],
            contributor_name=site_name,
            contribution_round=round_number,
        )
    return helper.get_result()
