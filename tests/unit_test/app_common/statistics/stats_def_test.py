# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import random

import pandas as pd

from nvflare.app_common.abstract.statistics_spec import Bin, DataType, Histogram, HistogramType
from nvflare.app_common.statistics.numpy_utils import dtype_to_data_type
from nvflare.app_common.utils.json_utils import ObjectEncoder


class TestStatsDef:
    def test_dtype_to_data_type(self):
        train_data = [
            ["tom", 10, 15.5],
            ["nick", 15, 10.2],
            ["juli", 14],
            ["tom2", 10, 13.0],
            ["nick1", 25],
            ["juli1", 24, 10.5],
        ]
        train = pd.DataFrame(train_data, columns=["Name", "Age", "Edu"])

        assert DataType.STRING == dtype_to_data_type(train["Name"].dtype)
        assert DataType.INT == dtype_to_data_type(train["Age"].dtype)
        assert DataType.FLOAT == dtype_to_data_type(train["Edu"].dtype)

    def test_feature_histogram_to_json(self):
        even = [1, 3, 5, 7, 9]
        odd = [2, 4, 6, 8, 10]
        buckets = zip(even, odd)
        bins = [Bin(low_value=b[0], high_value=b[1], sample_count=random.randint(10, 100)) for b in buckets]
        hist = Histogram(HistogramType.STANDARD, bins)
        statistics = {"histogram": {"site-1": {"train": {"feat": hist}}}}
        x = json.dumps(statistics, cls=ObjectEncoder)
        assert x.__eq__(
            {
                "histogram": {
                    "site-1": {
                        "train": {"feat": [0, [[1, 2, 83], [3, 4, 79], [5, 6, 69], [7, 8, 72], [9, 10, 20]], "null"]}
                    }
                }
            }
        )
