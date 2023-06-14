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

from unittest.mock import Mock, patch

import numpy
from pt.learner_with_tb import PTLearner

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class TestPTLearner:
    @patch.object(PTLearner, "save_local_model")
    def test_train_empty_input(self, mock_save_local_model):
        fl_ctx = FLContext()
        learner = PTLearner(epochs=1)
        learner.initialize(parts={}, fl_ctx=fl_ctx)

        data = Shareable()
        result = learner.train(data, fl_ctx=FLContext(), abort_signal=Signal())
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    @patch.object(PTLearner, "save_local_model")
    def test_train_with_empty_input(self, mock_save_local_model):
        fl_ctx = FLContext()
        learner = PTLearner(epochs=1)
        learner.initialize(parts={}, fl_ctx=fl_ctx)

        data = Shareable()
        result = learner.train(data, fl_ctx=FLContext(), abort_signal=Signal())
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    @patch.object(PTLearner, "save_local_model")
    def test_train_with_invalid_data_kind(self, mock_save_local_model):
        fl_ctx = FLContext()
        learner = PTLearner(epochs=1)
        learner.initialize(parts={}, fl_ctx=fl_ctx)

        dxo = DXO(DataKind.WEIGHT_DIFF, data={"x": numpy.array([1, 2, 3])})
        result = learner.train(dxo.to_shareable(), fl_ctx=FLContext(), abort_signal=Signal())
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    @patch.object(PTLearner, "save_local_model")
    def test_train(self, mock_save_local_model):
        fl_ctx = FLContext()
        learner = PTLearner(epochs=1)
        learner.initialize(parts={}, fl_ctx=fl_ctx)

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=learner.model.state_dict())
        result = learner.train(dxo.to_shareable(), fl_ctx=FLContext(), abort_signal=Signal())
        assert result.get_return_code() == ReturnCode.OK

    @patch.object(FLContext, "get_engine")
    def test_validate_with_empty_input(self, mock_get_engine):
        mock_get_engine.get_workspace = Mock()
        fl_ctx = FLContext()
        fl_ctx.set_prop(ReservedKey.RUN_NUM, 100)

        learner = PTLearner(epochs=1)
        learner.initialize(parts={}, fl_ctx=fl_ctx)

        data = Shareable()
        result = learner.validate(data, fl_ctx=fl_ctx, abort_signal=Signal())
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    @patch.object(FLContext, "get_engine")
    def test_validate_with_invalid_data_kind(self, mock_get_engine):
        mock_get_engine.get_workspace = Mock()
        fl_ctx = FLContext()
        fl_ctx.set_prop(ReservedKey.RUN_NUM, 100)

        learner = PTLearner(epochs=1)
        learner.initialize(parts={}, fl_ctx=fl_ctx)

        dxo = DXO(DataKind.WEIGHT_DIFF, data={"x": numpy.array([1, 2, 3])})
        result = learner.validate(dxo.to_shareable(), fl_ctx=fl_ctx, abort_signal=Signal())
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    @patch.object(FLContext, "get_engine")
    def test_validate(self, mock_get_engine):
        mock_get_engine.get_workspace = Mock()
        fl_ctx = FLContext()
        fl_ctx.set_prop(ReservedKey.RUN_NUM, 100)

        learner = PTLearner(epochs=1)
        learner.initialize(parts={}, fl_ctx=fl_ctx)

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=learner.model.state_dict())
        result = learner.train(dxo.to_shareable(), fl_ctx=FLContext(), abort_signal=Signal())
        assert result.get_return_code() == ReturnCode.OK
