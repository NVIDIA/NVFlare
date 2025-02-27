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

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.apis.workspace import Workspace


class TestWorkspace:
    @pytest.mark.parametrize(
        "root_vars, expected",
        [
            (("r", "l", "a"), ("r", "l", "a")),
            (("r", "l", None), ("r", "l", "l")),
            (("r", None, None), ("r", "r", "r")),
            (("r", None, "a"), ("r", "a", "a")),
            ((None, "l", "a"), (None, "l", "a")),
            ((None, "l", None), (None, "l", "l")),
            ((None, None, None), (None, None, None)),
        ],
    )
    def test_init(self, root_vars, expected):
        # we have to create real dirs since Workspace checks existence of the specified roots
        r, l, a = root_vars
        var_dict = {
            WorkspaceConstants.ENV_VAR_RESULT_ROOT: r,
            WorkspaceConstants.ENV_VAR_LOG_ROOT: l,
            WorkspaceConstants.ENV_VAR_AUDIT_ROOT: a,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            for n, v in var_dict.items():
                if v:
                    os.environ[n] = os.path.join(tmp_dir, v)
                else:
                    os.environ.pop(n, None)

            root_dir = os.path.join(tmp_dir, "config")
            os.makedirs(root_dir, exist_ok=True)
            os.makedirs(os.path.join(root_dir, "startup"), exist_ok=True)
            os.makedirs(os.path.join(root_dir, "local"), exist_ok=True)

            ws = Workspace(root_dir)
            result = (
                os.path.relpath(ws.result_root, tmp_dir) if ws.result_root else None,
                os.path.relpath(ws.log_root, tmp_dir) if ws.log_root else None,
                os.path.relpath(ws.audit_root, tmp_dir) if ws.audit_root else None,
            )
            assert result == expected
