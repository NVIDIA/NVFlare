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

import numpy as np
import pytest

from custom_aggregators import MedianAggregator

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.np.constants import NPConstants


def _model(weights: np.ndarray, params_type=ParamsType.FULL) -> FLModel:
    return FLModel(params={NPConstants.NUMPY_KEY: weights}, params_type=params_type)


def test_median_aggregation_reduces_outlier_impact():
    agg = MedianAggregator()

    benign_1 = np.array([2.0, 2.0, 2.0])
    benign_2 = np.array([2.1, 1.9, 2.0])
    benign_3 = np.array([1.9, 2.1, 2.0])
    poisoned = np.array([1000.0, 1000.0, 1000.0])

    agg.accept_model(_model(benign_1))
    agg.accept_model(_model(benign_2))
    agg.accept_model(_model(benign_3))
    agg.accept_model(_model(poisoned))

    result = agg.aggregate_model()
    median_weights = result.params[NPConstants.NUMPY_KEY]

    benign_mean = np.mean(np.stack([benign_1, benign_2, benign_3]), axis=0)
    naive_mean = np.mean(np.stack([benign_1, benign_2, benign_3, poisoned]), axis=0)

    assert np.linalg.norm(median_weights - benign_mean) < np.linalg.norm(naive_mean - benign_mean)


def test_params_type_mismatch_raises():
    agg = MedianAggregator()
    agg.accept_model(_model(np.array([1.0, 2.0]), params_type=ParamsType.FULL))

    with pytest.raises(ValueError, match="ParamsType mismatch"):
        agg.accept_model(_model(np.array([1.0, 2.0]), params_type=ParamsType.DIFF))


def test_reset_stats_clears_state():
    agg = MedianAggregator()
    agg.accept_model(_model(np.array([1.0, 2.0, 3.0])))
    agg.reset_stats()
    result = agg.aggregate_model()
    assert result.params == {}
