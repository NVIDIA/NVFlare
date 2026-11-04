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

from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.executors.model_learner_executor import ModelLearnerExecutor
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


@pytest.mark.parametrize("params_type", [ParamsType.FULL, ParamsType.DIFF])
@pytest.mark.parametrize("validation_metrics", [{"accuracy": 0.4}, None])
def test_train_preserves_before_train_metrics_for_model_selection(params_type, validation_metrics):
    executor = ModelLearnerExecutor("learner")
    executor.learner = Mock()
    executor.learner.validate.return_value = FLModel(metrics=validation_metrics) if validation_metrics else None
    executor.learner.train.return_value = FLModel(params={"w": 1}, params_type=params_type, metrics={"accuracy": 0.9})
    incoming = FLModelUtils.to_shareable(FLModel(params={"w": 0}))

    result = from_shareable(executor.train(incoming, FLContext()))

    assert result.data_kind == (DataKind.WEIGHTS if params_type == ParamsType.FULL else DataKind.WEIGHT_DIFF)
    assert result.data == {"w": 1}
    assert result.meta[MetaKey.INITIAL_METRICS] == (validation_metrics or {"accuracy": 0.9})


def test_metrics_only_train_result_preserves_payload_and_validation_metadata():
    executor = ModelLearnerExecutor("learner")
    executor.learner = Mock()
    executor.learner.validate.return_value = FLModel(metrics={"accuracy": 0.4})
    executor.learner.train.return_value = FLModel(metrics={"accuracy": 0.9})
    incoming = FLModelUtils.to_shareable(FLModel(params={"w": 0}))

    result = from_shareable(executor.train(incoming, FLContext()))

    assert result.data_kind == DataKind.METRICS
    assert result.data == {"accuracy": 0.9}
    assert result.meta[MetaKey.INITIAL_METRICS] == {"accuracy": 0.4}
