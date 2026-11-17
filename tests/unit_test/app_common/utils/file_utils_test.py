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

import os

import pytest

from nvflare.app_common.utils.file_utils import resolve_path_under_root


def test_resolve_path_under_root_accepts_nested_relative_path(tmp_path):
    assert resolve_path_under_root(str(tmp_path), "models/server.npy") == os.path.realpath(
        tmp_path / "models" / "server.npy"
    )


def test_resolve_path_under_root_rejects_absolute_path(tmp_path):
    with pytest.raises(ValueError, match="must be relative"):
        resolve_path_under_root(str(tmp_path), str(tmp_path / "outside.npy"), "model path")


def test_resolve_path_under_root_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="must stay inside"):
        resolve_path_under_root(str(tmp_path), "../outside.npy", "model path")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
def test_resolve_path_under_root_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must stay inside"):
        resolve_path_under_root(str(tmp_path), "link/outside.npy", "model path")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
def test_resolve_path_under_root_can_preserve_an_in_root_leaf_symlink(tmp_path):
    target = tmp_path / "target.npy"
    target.touch()
    alias = tmp_path / "current.npy"
    alias.symlink_to(target)

    assert resolve_path_under_root(str(tmp_path), alias.name, preserve_leaf=True) == str(alias)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
def test_resolve_path_under_root_does_not_preserve_an_escaping_leaf_symlink(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside.npy"
    outside.touch()
    alias = tmp_path / "current.npy"
    alias.symlink_to(outside)

    with pytest.raises(ValueError, match="must stay inside"):
        resolve_path_under_root(str(tmp_path), alias.name, preserve_leaf=True)
