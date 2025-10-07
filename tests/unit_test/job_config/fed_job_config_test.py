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

import os
import tempfile

import pytest

from nvflare.job_config.fed_job_config import FedJobConfig


class TestFedJobConfig:
    def test_locate_imports(self):
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        cwd = os.path.dirname(__file__)
        source_file = os.path.join(cwd, "../data/job_config/sample_code.data")
        expected = [
            "from typing import Any, Dict, List",
            "from nvflare.fuel.f3.drivers.base_driver import BaseDriver",
            "from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo ",
            "from nvflare.fuel.f3.drivers.driver_params import DriverCap",
        ]
        with open(source_file, "r") as sf:
            with tempfile.NamedTemporaryFile(dir=cwd, suffix=".py") as dest_file:
                imports = list(job_config.locate_imports(sf, dest_file=dest_file.name))
        assert imports == expected

    def test_trim_whitespace(self):
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        expected = "site-0,site-1"
        assert expected == job_config._trim_whitespace("site-0,site-1")
        assert expected == job_config._trim_whitespace("site-0, site-1")
        assert expected == job_config._trim_whitespace(" site-0,site-1 ")
        assert expected == job_config._trim_whitespace(" site-0, site-1 ")

    def test_get_args_instance_attributes(self):
        """Test _get_args with components that have instance attributes in __dict__"""

        class ComponentWithInstanceAttrs:
            def __init__(self, param1: str = "default1", param2: int = 42):
                self.param1 = param1
                self.param2 = param2

        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        component = ComponentWithInstanceAttrs("custom_value", 100)

        args = job_config._get_args(component, custom_dir=".")

        # Should extract both parameters since they differ from defaults
        assert args["param1"] == "custom_value"
        assert args["param2"] == 100

    def test_get_args_instance_attributes_use_defaults(self):
        """Test _get_args with components that have instance attributes in __dict__"""

        class ComponentWithInstanceAttrs:
            def __init__(self, param1: str = "default1", param2: int = 42):
                self.param1 = param1
                self.param2 = param2

        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        component = ComponentWithInstanceAttrs()

        args = job_config._get_args(component, custom_dir=".")

        # Should extract both parameters since they differ from defaults
        assert args == {}

    def test_get_args_instance_attributes_underscore(self):
        """Test _get_args with components that have properties (not in __dict__)"""

        class ComponentWithProperties:
            def __init__(self, value: str = "default"):
                self._value = value

        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        component = ComponentWithProperties("test_value")

        args = job_config._get_args(component, custom_dir=".")

        assert args["value"] == "test_value"

    def test_get_args_property_ambiguous(self):
        """Test _get_args with property attribute that is also in __dict__"""

        class ComponentWithProperty:
            def __init__(self, threshold: float = 0.5):
                self._threshold = threshold

            @property
            def threshold(self) -> float:
                return self._threshold * 2

        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        component = ComponentWithProperty(0.8)

        with pytest.raises(ValueError, match="Ambiguous attribute naming.*both 'threshold' and '_threshold' exist"):
            job_config._get_args(component, custom_dir=".")

    def test_get_args_class_attribute_fallback(self):
        """Test _get_args with class-level attributes when instance attributes don't exist"""

        class ComponentWithClassDefaults:
            # Class-level defaults that won't be in instance __dict__
            config_path: str = "/default/path"
            batch_size: int = 32

            def __init__(self, config_path: str = "/default/path", batch_size: int = 32):
                # Only set attributes if they differ from class defaults
                if config_path != "/default/path":
                    self.config_path = config_path
                if batch_size != 32:
                    self.batch_size = batch_size

        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        component = ComponentWithClassDefaults("/custom/path", 32)  # batch_size uses class default

        args = job_config._get_args(component, custom_dir=".")

        assert args["config_path"] == "/custom/path"

    def test_get_args_attribute_priority_param_vs_underscore(self):
        """Test _get_args priority when both param and _param exist"""

        class ComponentWithBothAttributes:
            def __init__(self, model_name: str = "default_model"):
                # Intentionally set both param and _param to test priority
                self.model_name = "public_value"  # Direct parameter name
                self._model_name = "private_value"  # Underscore version

        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        component = ComponentWithBothAttributes()

        with pytest.raises(ValueError, match="Ambiguous attribute naming.*both 'model_name' and '_model_name' exist"):
            job_config._get_args(component, custom_dir=".")

    def test_get_args_edge_case_missing_attributes_graceful_handling(self):
        """Test _get_args graceful handling when expected attributes are completely missing"""

        class ComponentMissingExpectedAttrs:
            def __init__(self, required_param: str = "default", optional_param: int = 100):
                # Simulate a component that doesn't store its constructor parameters
                # This could happen with custom __init__ logic or inheritance issues
                self.some_other_attr = "exists"
                # Notice: required_param and optional_param are NOT stored

        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        component = ComponentMissingExpectedAttrs("test", 200)

        args = job_config._get_args(component, custom_dir=".")

        # Should return empty dict since no expected parameters were found
        assert args == {}
        # Verify the component exists and our test setup is correct
        assert hasattr(component, "some_other_attr")

    def test_get_args_inherit_from_base_class(self):
        """Test _get_args with components that inherit from base class"""

        class BaseComponent:
            def __init__(self, base_param: str = "default"):
                self.base_param = base_param

        class DerivedComponent(BaseComponent):
            def __init__(self, derived_param: str = "derived", *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.derived_param = derived_param

        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        component = DerivedComponent("derived_value", base_param="base_value")

        args = job_config._get_args(component, custom_dir=".")

        assert args["base_param"] == "base_value"
        assert args["derived_param"] == "derived_value"

    def test_get_args_inherit_from_base_class_with_new_default_in_derived(self):
        """Test _get_args with components that inherit from base class"""

        class BaseComponent:
            def __init__(self, base_param: str = "default"):
                self.base_param = base_param

        class DerivedComponent(BaseComponent):
            def __init__(self, derived_param: str = "derived", base_param: str = "new_default", *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.derived_param = derived_param
                self.base_param = base_param

        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        component = DerivedComponent("derived_value", base_param="base_value")

        args = job_config._get_args(component, custom_dir=".")

        assert args["base_param"] == "base_value"
        assert args["derived_param"] == "derived_value"

    def test_get_args_inherit_from_base_class_with_new_default(self):
        """Test _get_args with components that inherit from base class"""

        class BaseComponent:
            def __init__(self, base_param: str = "default"):
                self.base_param = base_param

        class DerivedComponent(BaseComponent):
            def __init__(self, derived_param: str = "derived", base_param: str = "new_default", *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.derived_param = derived_param
                self.base_param = base_param

        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        component = DerivedComponent("derived_value", base_param="new_default")

        args = job_config._get_args(component, custom_dir=".")

        assert "base_param" not in args
        assert args["derived_param"] == "derived_value"
