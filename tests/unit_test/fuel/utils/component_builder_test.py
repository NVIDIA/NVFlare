# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import re

import pytest

from nvflare.app_common.np.np_model_locator import NPModelLocator
from tests.unit_test.fuel.utils.mock_component_builder import MockComponentBuilder


class MyComponent:
    def __init__(self, model):
        self.mode = model


class MyComponentFailure:
    def __init__(self):
        raise RuntimeError("initialization failed")


class MyComponentWithDictArgs:
    def __init__(self, model: dict):
        self.mode = model


class TestComponentBuilder:
    def test_empty_dict(self):
        builder = MockComponentBuilder()
        b = builder.build_component({})
        assert b is None

    def test_component(self):
        config = {"id": "id", "path": "nvflare.app_common.np.np_model_locator.NPModelLocator", "args": {}}
        builder = MockComponentBuilder()

        assert isinstance(config, dict)
        b = builder.build_component(config)
        assert isinstance(b, NPModelLocator)

    def test_component_failure(self):
        config = {"id": "id", "path": "nvflare.app_common.np.np_model_locator.NPModelLocator", "args": {"xyz": 1}}
        builder = MockComponentBuilder()

        assert isinstance(config, dict)
        b = None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Class nvflare.app_common.np.np_model_locator.NPModelLocator has parameters error: __init__() got an unexpected keyword argument 'xyz'."
            ),
        ):
            b = builder.build_component(config)

    def test_component_init_failure(self):
        config = {
            "id": "id",
            "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponentFailure",
            "args": {},
        }
        builder = MockComponentBuilder()
        assert isinstance(config, dict)
        with pytest.raises(RuntimeError, match="initialization failed"):
            builder.build_component(config)

    def test_embedded_component(self):
        config = {
            "id": "id",
            "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponent",
            "args": {"model": {"path": "nvflare.app_common.np.np_model_locator.NPModelLocator", "args": {}}},
        }
        builder = MockComponentBuilder()
        assert isinstance(config, dict)
        b = builder.build_component(config)
        assert isinstance(b, MyComponent)

    def test_embedded_component_with_dict_args(self):
        config = {
            "id": "id",
            "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponent",
            "args": {
                "model": {
                    "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponentWithDictArgs",
                    "args": {"model": {"a": "b"}},
                }
            },
        }
        builder = MockComponentBuilder()
        assert isinstance(config, dict)
        b = builder.build_component(config)
        assert isinstance(b, MyComponent)
        assert isinstance(b.mode, MyComponentWithDictArgs)
        assert b.mode.mode == {"a": "b"}

    def test_embedded_component_failure(self):
        config = {
            "id": "id",
            "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponent",
            "args": {"model": {"path": "nvflare.app_common.np.np_model_locator.NPModelLocator", "args": {"abc": 1}}},
        }
        builder = MockComponentBuilder()
        assert isinstance(config, dict)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "failed to instantiate class: Class nvflare.app_common.np.np_model_locator.NPModelLocator has parameters error: __init__() got an unexpected keyword argument 'abc'."
            ),
        ):
            b = builder.build_component(config)
