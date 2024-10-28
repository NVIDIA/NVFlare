# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType


class TestFLModel:
    model = FLModel(
        params_type=ParamsType.FULL,
        params={"a": 100, "b": 200, "c": {"c1": 100}},
        optimizer_params={},
        metrics={"loss": 100, "accuracy": 0.9},
        start_round=1,
        current_round=100,
        total_rounds=12000,
    )
    summary = model.summary()
    assert summary["params"] == 3
    assert summary["metrics"] == 2
    assert summary["params_type"] == ParamsType.FULL
    assert summary["start_round"] == 1
    assert summary["current_round"] == 100
    assert summary["total_rounds"] == 12000
