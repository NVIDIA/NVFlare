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
import os
import tempfile

import pytest

from nvflare.apis.workspace import Workspace


class TestWorkspace:
    @pytest.mark.parametrize(
        "root_dir, expected",
        [
            ("a", ("a", None, None, None)),
            ("a:", ("a", None, None, None)),
            ("a::", ("a", None, None, None)),
            ("a:::", ("a", None, None, None)),
            ("a:b", ("a", "b", "b", "b")),
            ("a:b:c", ("a", "b", "c", "c")),
            ("a:b:c:", ("a", "b", "c", "c")),
            ("a:b:c:d", ("a", "b", "c", "d")),
            ("a:b::d", ("a", "b", "b", "d")),
            ("a::c:d", ("a", None, "c", "d")),
            ("a::c:", ("a", None, "c", "c")),
            ("a:::d", ("a", None, None, "d")),
        ],
    )
    def test_init(self, root_dir, expected):
        # we have to create real dirs since Workspace checks existence of the specified roots
        with tempfile.TemporaryDirectory() as tmp_dir:
            parts = root_dir.split(":")
            dirs = []
            for p in parts:
                if p:
                    d = os.path.join(tmp_dir, p)
                    os.makedirs(d, exist_ok=True)
                    dirs.append(d)
                else:
                    dirs.append("")

            r = dirs[0]
            os.makedirs(os.path.join(r, "startup"), exist_ok=True)
            os.makedirs(os.path.join(r, "local"), exist_ok=True)

            root_dir = ":".join(dirs)
            ws = Workspace(root_dir)
            result = (
                os.path.relpath(ws.root_dir, tmp_dir),
                os.path.relpath(ws.data_root, tmp_dir) if ws.data_root else None,
                os.path.relpath(ws.log_root, tmp_dir) if ws.log_root else None,
                os.path.relpath(ws.audit_root, tmp_dir) if ws.audit_root else None,
            )
            assert result == expected
