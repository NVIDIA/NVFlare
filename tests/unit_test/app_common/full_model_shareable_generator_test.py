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

import weakref

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


def test_weight_diff_conversion_validates_values_before_mutating_model():
    generator = FullModelShareableGenerator()
    model = make_model_learnable(weights={"first": 1, "second": 2}, meta_props={})
    fl_ctx = FLContext()
    fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, model, private=True, sticky=False)
    shareable = DXO(DataKind.WEIGHT_DIFF, {"first": 3, "second": "invalid"}).to_shareable()

    with pytest.raises(TypeError):
        generator.shareable_to_learnable(shareable, fl_ctx)

    assert model[ModelLearnableKey.WEIGHTS] == {"first": 1, "second": 2}


def test_weight_diff_conversion_retains_at_most_one_temporary_parameter():
    class TrackedValue:
        instances = weakref.WeakSet()
        peak_instances = 0

        def __init__(self, value):
            self.value = value
            self.instances.add(self)
            type(self).peak_instances = max(type(self).peak_instances, len(self.instances))

        def __add__(self, other):
            return TrackedValue(self.value + other.value)

    generator = FullModelShareableGenerator()
    parameter_count = 4
    model = make_model_learnable(
        weights={f"weight-{i}": TrackedValue(i) for i in range(parameter_count)}, meta_props={}
    )
    model_diff = {f"weight-{i}": TrackedValue(1) for i in range(parameter_count)}
    fl_ctx = FLContext()
    fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, model, private=True, sticky=False)

    generator.shareable_to_learnable(DXO(DataKind.WEIGHT_DIFF, model_diff).to_shareable(), fl_ctx)

    # The model and diff account for two full parameter sets. Validation or
    # application may allocate one additional parameter, but never a third model.
    assert TrackedValue.peak_instances <= 2 * parameter_count + 1
