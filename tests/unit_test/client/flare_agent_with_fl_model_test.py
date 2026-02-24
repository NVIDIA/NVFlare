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

from types import SimpleNamespace

import numpy as np

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.client.flare_agent_with_fl_model import FlareAgentWithFLModel, _ConverterContext


class _AddOneConverter(ParamsConverter):
    def convert(self, params, fl_ctx):
        fl_ctx.set_prop("from_called", True)
        return {k: v + 1 for k, v in params.items()}


class _AddTwoConverter(ParamsConverter):
    def convert(self, params, fl_ctx):
        assert fl_ctx.get_prop("from_called", False)
        return {k: v + 2 for k, v in params.items()}


def _make_model(value: int) -> FLModel:
    return FLModel(params={"w": np.asarray([value], dtype=np.float32)}, params_type=ParamsType.FULL)


def _make_agent(from_converter=None, to_converter=None):
    agent = FlareAgentWithFLModel.__new__(FlareAgentWithFLModel)
    agent.from_nvflare_converter = from_converter
    agent.to_nvflare_converter = to_converter
    agent._converter_ctx = _ConverterContext()
    agent.current_task = None
    return agent


def test_shareable_to_task_data_applies_from_converter():
    shareable = FLModelUtils.to_shareable(_make_model(1))
    shareable.set_header(FLContextKey.TASK_NAME, "train")
    agent = _make_agent(from_converter=_AddOneConverter(["train"]))

    model = agent.shareable_to_task_data(shareable)
    np.testing.assert_array_equal(model.params["w"], np.asarray([2], dtype=np.float32))


def test_task_result_to_shareable_applies_to_converter():
    agent = _make_agent(
        from_converter=_AddOneConverter(["train"]),
        to_converter=_AddTwoConverter(["train"]),
    )
    agent.current_task = SimpleNamespace(task_name="train")

    # First call primes converter context state used by to-converter assertion.
    input_shareable = FLModelUtils.to_shareable(_make_model(1))
    input_shareable.set_header(FLContextKey.TASK_NAME, "train")
    _ = agent.shareable_to_task_data(input_shareable)

    output_shareable = agent.task_result_to_shareable(_make_model(10), ReturnCode.OK)
    output_model = FLModelUtils.from_shareable(output_shareable)
    np.testing.assert_array_equal(output_model.params["w"], np.asarray([12], dtype=np.float32))
