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

import configparser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
CONFIG = configparser.ConfigParser()
CONFIG.read(ROOT / "setup.cfg")
WEB_DEPENDENCIES = {"flask", "flask-jwt-extended", "flask-sqlalchemy", "gunicorn"}


def _dependency_names(section, option):
    return {
        canonicalize_name(Requirement(line.strip()).name)
        for line in CONFIG.get(section, option).splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def test_base_dependencies_do_not_include_optional_web_stack():
    base = _dependency_names("options", "install_requires")
    assert base.isdisjoint(WEB_DEPENDENCIES)
    assert "pyjwt" in base  # Direct dependency of confidential-computing authorizers.


def test_dashboard_dependencies_are_available_as_an_extra():
    assert _dependency_names("options.extras_require", "dashboard") == WEB_DEPENDENCIES


def test_edge_http_does_not_require_dashboard_extensions():
    assert _dependency_names("options.extras_require", "edge-http") == {"flask"}


@pytest.mark.parametrize(
    "extra", ("all", "all_mac", "all_cpu", "test", "test_mac", "test_cpu", "dev", "dev_mac", "dev_cpu")
)
def test_full_development_profiles_keep_web_tests_available(extra):
    assert WEB_DEPENDENCIES <= _dependency_names("options.extras_require", extra)
