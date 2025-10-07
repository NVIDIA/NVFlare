# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.fuel.utils.class_utils import get_component_init_parameters, resolve_component_attribute_key
from nvflare.private.fed.server.client_manager import ClientManager


class A:
    def __init__(self, name: str = None):
        self.name = name


class B(A):
    def __init__(self, type: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = type


class TestClassUtils:
    def test_get_component_init_parameters(self):
        manager = ClientManager(project_name="Sample project", min_num_clients=1)
        constructor = ClientManager.__init__
        expected_parameters = inspect.signature(constructor).parameters
        parameters = get_component_init_parameters(manager)

        assert parameters == expected_parameters

    def test_nested_init_parameters(self):
        expected_parameters = {}
        constructor = B.__init__
        b = B(name="name", type="type")
        expected_parameters.update(inspect.signature(constructor).parameters)
        constructor = A.__init__
        expected_parameters.update(inspect.signature(constructor).parameters)
        parameters = get_component_init_parameters(b)

        assert parameters == expected_parameters

    def test_resolve_component_attribute_key_simple_attribute(self):
        """Test resolve_component_attribute_key with simple attribute"""

        class SimpleComponent:
            def __init__(self, value: int = 42):
                self.value = value

        component = SimpleComponent(100)
        attr_key = resolve_component_attribute_key(component, "value")

        assert attr_key == "value"

    def test_resolve_component_attribute_key_underscore_attribute(self):
        """Test resolve_component_attribute_key with underscore attribute"""

        class ComponentWithUnderscore:
            def __init__(self, threshold: float = 0.5):
                self._threshold = threshold

        component = ComponentWithUnderscore(0.8)
        attr_key = resolve_component_attribute_key(component, "threshold")

        assert attr_key == "_threshold"

    def test_resolve_component_attribute_key_ambiguous_naming(self):
        """Test resolve_component_attribute_key with ambiguous attribute naming"""

        class ComponentWithBoth:
            def __init__(self, model_name: str = "default"):
                self.model_name = "public_value"
                self._model_name = "private_value"

        component = ComponentWithBoth()

        with pytest.raises(ValueError, match="Ambiguous attribute naming.*both 'model_name' and '_model_name' exist"):
            resolve_component_attribute_key(component, "model_name")

    def test_resolve_component_attribute_key_property_vs_instance_conflict(self):
        """Test resolve_component_attribute_key with property vs instance variable conflict"""

        class ComponentWithPropertyConflict:
            def __init__(self, threshold: float = 0.5):
                self._threshold = threshold

            @property
            def threshold(self) -> float:
                return self._threshold * 2

        component = ComponentWithPropertyConflict(0.8)

        with pytest.raises(ValueError, match="Ambiguous attribute naming.*both 'threshold' and '_threshold' exist"):
            resolve_component_attribute_key(component, "threshold")

    def test_resolve_component_attribute_key_property_only(self):
        """Test resolve_component_attribute_key with property only (no conflict)"""

        class ComponentWithPropertyOnly:
            def __init__(self, value: int = 42):
                self._internal_value = value

            @property
            def computed_value(self) -> int:
                return self._internal_value * 2

        component = ComponentWithPropertyOnly(10)
        attr_key = resolve_component_attribute_key(component, "computed_value")

        assert attr_key == "computed_value"
        assert component.computed_value == 20

    def test_resolve_component_attribute_key_nonexistent_attribute(self):
        """Test resolve_component_attribute_key with non-existent attribute"""

        class ComponentWithMissingAttrs:
            def __init__(self, existing_param: str = "default"):
                self.existing_param = existing_param

        component = ComponentWithMissingAttrs()
        attr_key = resolve_component_attribute_key(component, "nonexistent_param")

        assert attr_key is None

    def test_resolve_component_attribute_key_slots_component(self):
        """Test resolve_component_attribute_key with __slots__ component (no __dict__)"""

        class ComponentWithSlots:
            __slots__ = ["value", "name"]

            def __init__(self, value: int = 42, name: str = "test"):
                self.value = value
                self.name = name

        component = ComponentWithSlots(100, "example")

        value_key = resolve_component_attribute_key(component, "value")
        name_key = resolve_component_attribute_key(component, "name")

        assert value_key == "value"
        assert name_key == "name"

    def test_resolve_component_attribute_key_inheritance_defaults(self):
        """Test resolve_component_attribute_key with inheritance (child class takes precedence)"""

        class Parent:
            def __init__(self, value: int = 10):
                self.value = value

        class Child(Parent):
            def __init__(self, value: int = 20, *args, **kwargs):  # Different default
                super().__init__(*args, **kwargs)
                self.value = value

        child = Child()
        parameters = get_component_init_parameters(child)

        # Child's default should take precedence
        assert parameters["value"].default == 20

        attr_key = resolve_component_attribute_key(child, "value")
        assert attr_key == "value"

    def test_resolve_component_attribute_key_complex_inheritance(self):
        """Test resolve_component_attribute_key with complex inheritance hierarchy"""

        class GrandParent:
            def __init__(self, gp_param: str = "gp_default"):
                self.gp_param = gp_param

        class Parent(GrandParent):
            def __init__(self, p_param: str = "p_default", *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.p_param = p_param

        class Child(Parent):
            def __init__(self, c_param: str = "c_default", *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.c_param = c_param

        child = Child(c_param="child_value", p_param="parent_value", gp_param="grandparent_value")

        # All parameters should be resolvable
        c_key = resolve_component_attribute_key(child, "c_param")
        p_key = resolve_component_attribute_key(child, "p_param")
        gp_key = resolve_component_attribute_key(child, "gp_param")

        assert c_key == "c_param"
        assert p_key == "p_param"
        assert gp_key == "gp_param"

        # Parameter collection should include all levels
        parameters = get_component_init_parameters(child)
        expected_params = {"self", "c_param", "p_param", "gp_param", "args", "kwargs"}
        assert set(parameters.keys()) == expected_params
