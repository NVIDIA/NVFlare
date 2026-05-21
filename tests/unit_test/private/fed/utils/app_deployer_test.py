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

from nvflare.apis.workspace import Workspace
from nvflare.private.fed.utils.app_deployer import AppDeployer


def _make_workspace(root_dir: str) -> Workspace:
    os.makedirs(os.path.join(root_dir, "startup"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "local"), exist_ok=True)
    return Workspace(root_dir, site_name="site-1")


def test_deploy_rejects_traversing_job_id_before_touching_outside_path(tmp_path):
    workspace = _make_workspace(str(tmp_path / "workspace"))
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker.txt"
    marker.write_text("keep")

    with pytest.raises(Exception, match="invalid job_id"):
        AppDeployer().deploy(
            workspace=workspace,
            job_id="good/../../outside",
            job_meta={},
            app_name="app",
            app_data=b"not-needed",
            fl_ctx=None,
        )

    assert marker.read_text() == "keep"
