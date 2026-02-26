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
# WITHOUT WARRANTIES OR CONDITIONS OF KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for wfconf Configurator get_class_path and build_component (path/class_path behavior)."""

import os
import tempfile

import pytest

from nvflare.app_common.np.np_model_locator import NPModelLocator
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.wfconf import Configurator


@pytest.fixture
def wfconf_configurator():
    """Create a Configurator instance with a minimal temp config file (required by __init__)."""
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("{}")
        configurator = Configurator(
            app_root="/tmp",
            cmd_vars=None,
            env_config=None,
            wf_config_file_name=path,
            base_pkgs=["nvflare"],
            module_names=["api", "app_commons", "app_opt", "fuel", "private", "utils"],
        )
        yield configurator
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class TestWfconfGetClassPathAndBuildComponent:
    """Test path/class_path behavior in wfconf.Configurator (consistent with component_builder)."""

    def test_get_class_path_with_path(self, wfconf_configurator):
        """get_class_path returns the value when 'path' is specified."""
        config = {"path": "nvflare.app_common.np.np_model_locator.NPModelLocator"}
        assert wfconf_configurator.get_class_path(config) == config["path"]

    def test_get_class_path_with_class_path(self, wfconf_configurator):
        """get_class_path returns the value when 'class_path' is specified (no path)."""
        config = {"class_path": "nvflare.app_common.np.np_model_locator.NPModelLocator"}
        assert wfconf_configurator.get_class_path(config) == config["class_path"]

    def test_get_class_path_path_takes_precedence_over_class_path(self, wfconf_configurator):
        """When both 'path' and 'class_path' are present, get_class_path uses 'path'."""
        config = {
            "path": "nvflare.app_common.np.np_model_locator.NPModelLocator",
            "class_path": "some.other.Class",
        }
        assert wfconf_configurator.get_class_path(config) == config["path"]

    def test_build_component_with_class_path(self, wfconf_configurator):
        """build_component works when config uses 'class_path' instead of 'path'."""
        config = {
            "class_path": "nvflare.app_common.np.np_model_locator.NPModelLocator",
            "args": {},
        }
        b = wfconf_configurator.build_component(config)
        assert isinstance(b, NPModelLocator)

    def test_build_component_path_takes_precedence_over_class_path(self, wfconf_configurator):
        """When both 'path' and 'class_path' are present, build_component uses 'path'."""
        config = {
            "path": "nvflare.app_common.np.np_model_locator.NPModelLocator",
            "class_path": "some.other.Class",
            "args": {},
        }
        b = wfconf_configurator.build_component(config)
        assert isinstance(b, NPModelLocator)

    def test_empty_path_raises_even_when_class_path_present(self, wfconf_configurator):
        """Empty 'path' is validated and raises ConfigError; we do not silently use class_path."""
        config = {
            "path": "",
            "class_path": "nvflare.app_common.np.np_model_locator.NPModelLocator",
            "args": {},
        }
        with pytest.raises(ConfigError, match="path spec must not be empty"):
            wfconf_configurator.build_component(config)
