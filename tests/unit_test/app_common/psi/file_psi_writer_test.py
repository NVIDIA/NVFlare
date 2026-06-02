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

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.psi.file_psi_writer import FilePSIWriter


def _make_fl_ctx(tmp_path):
    app_root = tmp_path / "app_site-1"
    app_root.mkdir()
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.APP_ROOT, str(app_root), private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.IDENTITY_NAME, value="site-1", private=False, sticky=True)
    return fl_ctx


def test_file_psi_writer_accepts_relative_output_path(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    writer = FilePSIWriter(output_path="psi/intersection.txt")

    writer.save(["a", "b"], overwrite_existing=True, fl_ctx=fl_ctx)

    expected_path = tmp_path / "site-1" / "psi" / "intersection.txt"
    assert expected_path.read_text() == "a\nb"


@pytest.mark.parametrize("output_path", ["/tmp/intersection.txt", "../intersection.txt"])
def test_file_psi_writer_rejects_escaping_output_path(tmp_path, output_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    writer = FilePSIWriter(output_path=output_path)

    with pytest.raises(ValueError, match="must (be relative|stay inside)"):
        writer.save(["a"], overwrite_existing=True, fl_ctx=fl_ctx)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
def test_file_psi_writer_rejects_symlink_escape(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    site_dir = tmp_path / "site-1"
    site_dir.mkdir()
    (site_dir / "link").symlink_to(outside, target_is_directory=True)
    writer = FilePSIWriter(output_path="link/intersection.txt")

    with pytest.raises(ValueError, match="must stay inside"):
        writer.save(["a"], overwrite_existing=True, fl_ctx=fl_ctx)
