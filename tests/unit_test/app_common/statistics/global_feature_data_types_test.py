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

from copy import deepcopy
from itertools import permutations

import pytest

from nvflare.app_common.abstract.statistics_spec import DataType, Feature
from nvflare.app_common.statistics.numeric_stats import get_global_feature_data_types


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_union_is_preserved_across_clients(order):
    reports = [
        {"train": [Feature("age", DataType.INT)]},
        {"train": [Feature("weight", DataType.FLOAT), Feature("age", DataType.INT)]},
        {"train": [Feature("name", DataType.STRING)], "validation": [Feature("age", DataType.INT)]},
    ]
    clients = {f"site-{i}": reports[i] for i in order}
    before = deepcopy(clients)
    assert get_global_feature_data_types(clients) == {
        "train": {"age": DataType.INT, "weight": DataType.FLOAT, "name": DataType.STRING},
        "validation": {"age": DataType.INT},
    }
    assert clients == before


def test_empty_client_report_does_not_erase_feature_types():
    clients = {"site-1": {"train": [Feature("x", DataType.FLOAT)]}, "site-2": {"train": []}}
    assert get_global_feature_data_types(clients) == {"train": {"x": DataType.FLOAT}}


@pytest.mark.parametrize("dataset_name", ["train", "validation"])
def test_feature_names_may_match_dataset_names(dataset_name):
    clients = {"site": {"train": [Feature(dataset_name, DataType.FLOAT)], "validation": []}}
    assert get_global_feature_data_types(clients) == {"train": {dataset_name: DataType.FLOAT}, "validation": {}}


def test_repeated_feature_in_consistent_schema_remains_unchanged():
    features = [Feature("age", DataType.INT), Feature("weight", DataType.FLOAT)]
    clients = {name: {"train": features} for name in ["a", "b", "c"]}
    assert get_global_feature_data_types(clients) == {"train": {"age": DataType.INT, "weight": DataType.FLOAT}}


def test_empty_reports_remain_empty():
    assert get_global_feature_data_types({}) == {}
    assert get_global_feature_data_types({"site": {"train": []}}) == {"train": {}}
