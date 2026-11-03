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

import datetime
import importlib.util
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import setuptools

ROOT_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup_metadata(monkeypatch):
    """Exercise setup's metadata decisions without launching a build or Git process."""
    versioneer = SimpleNamespace(
        get_versions=lambda: {"version": "0+unknown", "error": "unable to compute version"},
        get_cmdclass=lambda: {},
    )
    monkeypatch.setattr(importlib.util, "spec_from_file_location", Mock())
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda _: versioneer)
    setup = Mock()
    monkeypatch.setattr(setuptools, "setup", setup)
    monkeypatch.setattr(setuptools, "find_packages", lambda **kwargs: ["nvflare"])
    monkeypatch.delenv("NVFL_BASE_VERSION", raising=False)
    return versioneer, setup


def test_versioneer_failure_without_base_version_fails_clearly(setup_metadata):
    with pytest.raises(RuntimeError, match="Versioneer could not determine the NVFlare package version") as error:
        runpy.run_path(str(ROOT_DIR / "setup.py"))

    assert "set NVFL_BASE_VERSION explicitly" in str(error.value)


def test_versioneer_failure_uses_configured_base_version_override(setup_metadata, monkeypatch):
    monkeypatch.setenv("NVFL_BASE_VERSION", "3.0.0")
    before = datetime.date.today()
    runpy.run_path(str(ROOT_DIR / "setup.py"))
    after = datetime.date.today()

    expected_versions = {f"3.0.0.dev{date:%y%m%d}" for date in (before, after)}
    metadata = setup_metadata[1].call_args.kwargs
    assert metadata["version"] in expected_versions
    assert metadata["name"] == "nvflare"


@pytest.mark.parametrize("release", (None, "0", "1"))
@pytest.mark.parametrize("version", ("2.10.0", "2.10.0rc1", "2.10.0.dev0+125.gd05110ab4"))
def test_release_and_development_builds_share_package_identity(setup_metadata, monkeypatch, release, version):
    versioneer, setup = setup_metadata
    versioneer.get_versions = lambda: {"version": version, "error": None}
    if release is None:
        monkeypatch.delenv("NVFL_RELEASE", raising=False)
    else:
        monkeypatch.setenv("NVFL_RELEASE", release)

    runpy.run_path(str(ROOT_DIR / "setup.py"))

    metadata = setup.call_args.kwargs
    assert metadata["name"] == "nvflare"
    assert metadata["version"] == version
