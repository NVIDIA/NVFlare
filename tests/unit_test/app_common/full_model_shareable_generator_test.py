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

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator


def test_weight_diff_conversion_does_not_partially_mutate_model_on_failure():
    generator = FullModelShareableGenerator()
    model = make_model_learnable(weights={"weight": 1}, meta_props={})
    fl_ctx = FLContext()
    fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, model, private=True, sticky=False)
    shareable = DXO(DataKind.WEIGHT_DIFF, {"weight": 2, "missing": 1}).to_shareable()

    with pytest.raises(KeyError):
        generator.shareable_to_learnable(shareable, fl_ctx)

    assert model[ModelLearnableKey.WEIGHTS] == {"weight": 1}
