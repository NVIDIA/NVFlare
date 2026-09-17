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

import inspect

import pytest

from nvflare.app_common.abstract.model_learner import _MODEL_LEARNER_DEPRECATION_MSG, ModelLearner
from nvflare.fuel.utils.deprecated import _WARNED_DEPRECATION_MESSAGES


class TestModelLearner:
    def setup_method(self):
        _WARNED_DEPRECATION_MESSAGES.discard(_MODEL_LEARNER_DEPRECATION_MSG)

    def test_init_warns_deprecated(self):
        with pytest.warns(DeprecationWarning, match="ModelLearner is deprecated"):
            learner = ModelLearner()

        assert isinstance(learner, ModelLearner)
        assert inspect.isclass(ModelLearner)

    def test_subclass_init_warns_deprecated(self):
        class TestLearner(ModelLearner):
            pass

        with pytest.warns(DeprecationWarning, match="ModelLearner is deprecated"):
            learner = TestLearner()

        assert isinstance(learner, TestLearner)
        assert isinstance(learner, ModelLearner)
        assert inspect.isclass(TestLearner)
