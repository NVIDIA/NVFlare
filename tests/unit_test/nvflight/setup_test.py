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

import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
import setuptools

ROOT_DIR = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("release", "version"),
    (
        (None, "2.10.0"),
        (None, "2.10.0rc1"),
        (None, "2.10.0.dev0+125.gd05110ab4"),
        ("0", "2.10.0.dev0"),
        ("1", "2.10.0"),
    ),
)
def test_builds_share_package_identity_and_preserve_version(monkeypatch, release, version):
    """Capture build metadata without invoking packaging commands or subprocesses."""
    setup = Mock()
    monkeypatch.setattr(setuptools, "setup", setup)
    monkeypatch.setattr(setuptools, "find_packages", lambda **kwargs: ["nvflare"])
    monkeypatch.setenv("NVFL_VERSION", version)
    if release is None:
        monkeypatch.delenv("NVFL_RELEASE", raising=False)
    else:
        monkeypatch.setenv("NVFL_RELEASE", release)

    runpy.run_path(str(ROOT_DIR / "nvflight" / "setup.py"))

    setup.assert_called_once()
    metadata = setup.call_args.kwargs
    assert metadata["name"] == "nvflare-light"
    assert metadata["version"] == version
    assert metadata["package_dir"] == {"nvflare": "nvflare"}
