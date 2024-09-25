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

import re
from platform import python_version

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
    def __init__(self, model: dict = None):
        self.mode = model


class MyComponentWithPathArgs:
    def __init__(self, path: str = None):
        self.path = path


def is_python_greater_than_309():
    version = python_version()
    version_value = 0
    if version.startswith("3.7."):
        version_value = 307
    elif version.startswith("3.8."):
        version_value = 308
    elif version.startswith("3.9."):
        version_value = 309
    elif version.startswith("3.10."):
        version_value = 310
    elif version.startswith("3.11."):
        version_value = 311
    elif version.startswith("3.12."):
        version_value = 312
    else:
        raise ValueError("unknown version")
    return version_value > 309


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

        # the failure message changes since 3.10
        if is_python_greater_than_309():
            msg = "Class nvflare.app_common.np.np_model_locator.NPModelLocator has parameters error: TypeError: NPModelLocator.__init__() got an unexpected keyword argument 'xyz'."
        else:
            msg = "Class nvflare.app_common.np.np_model_locator.NPModelLocator has parameters error: TypeError: __init__() got an unexpected keyword argument 'xyz'."

        assert isinstance(config, dict)
        b = None
        with pytest.raises(ValueError, match=re.escape(msg)):
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

        # the failure message changes since 3.10
        if is_python_greater_than_309():
            msg = "failed to instantiate class: ValueError: Class nvflare.app_common.np.np_model_locator.NPModelLocator has parameters error: TypeError: NPModelLocator.__init__() got an unexpected keyword argument 'abc'."
        else:
            msg = "failed to instantiate class: ValueError: Class nvflare.app_common.np.np_model_locator.NPModelLocator has parameters error: TypeError: __init__() got an unexpected keyword argument 'abc'."

        builder = MockComponentBuilder()
        assert isinstance(config, dict)

        with pytest.raises(
            ValueError,
            match=re.escape(msg),
        ):
            b = builder.build_component(config)

    def test_component_wo_args(self):
        config = {"id": "id", "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponentWithDictArgs"}
        builder = MockComponentBuilder()
        assert isinstance(config, dict)
        b = builder.build_component(config)
        assert isinstance(b, MyComponentWithDictArgs)

    def test_embedded_component_wo_args(self):
        config = {
            "id": "id",
            "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponent",
            "args": {
                "model": {
                    "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponentWithDictArgs",
                }
            },
        }
        builder = MockComponentBuilder()
        assert isinstance(config, dict)
        b = builder.build_component(config)
        assert isinstance(b, MyComponent)
        assert isinstance(b.mode, MyComponentWithDictArgs)

    def test_embedded_component_with_path_args(self):
        config = {
            "id": "id",
            "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponent",
            "args": {
                "model": {
                    "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponentWithPathArgs",
                    "args": {"path": "/tmp/nvflare"},
                }
            },
        }
        builder = MockComponentBuilder()
        assert isinstance(config, dict)
        b = builder.build_component(config)
        assert isinstance(b, MyComponent)
        assert isinstance(b.mode, MyComponentWithPathArgs)

    def test_nested_component_component_type(self):
        config = {
            "id": "id",
            "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponent",
            "args": {
                "model": {
                    "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponentWithDictArgs",
                    "config_type": "component",
                }
            },
        }
        builder = MockComponentBuilder()
        assert isinstance(config, dict)
        b = builder.build_component(config)
        assert isinstance(b, MyComponent)
        assert isinstance(b.mode, MyComponentWithDictArgs)

    def test_nested_dict_component_type(self):
        config = {
            "id": "id",
            "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponent",
            "args": {
                "model": {
                    "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponentWithDictArgs",
                    "config_type": "dict",
                }
            },
        }
        builder = MockComponentBuilder()
        assert isinstance(config, dict)
        b = builder.build_component(config)
        assert isinstance(b, MyComponent)
        assert b.mode == {
            "path": "tests.unit_test.fuel.utils.component_builder_test.MyComponentWithDictArgs",
            "config_type": "dict",
        }
