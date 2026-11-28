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
from pathlib import Path

import pytest

from nvflare.lighter.constants import CtxKey
from nvflare.lighter.impl.aws import AWSBuilder
from nvflare.lighter.impl.azure import AzureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from tests.process_test_utils import run_owned_process


@pytest.mark.parametrize("cloud", ["aws", "azure"])
@pytest.mark.parametrize("participant", ["server", "site-1"])
def test_cloud_start_sources_config_with_spaces(tmp_path, cloud, participant):
    project = prepare_project(
        {
            "api_version": 3,
            "name": "cloud-test",
            "participants": [
                {"type": "server", "name": "server", "org": "org"},
                {"type": "client", "name": "site-1", "org": "org"},
            ],
        }
    )
    builders = [WorkspaceBuilder(), StaticFileBuilder(), AWSBuilder(), AzureBuilder()]
    ctx = Provisioner(str(tmp_path / "workspace"), builders).provision(project)
    assert ctx[CtxKey.PROVISION_SUCCESS]
    start = Path(ctx[CtxKey.CURRENT_PROD_DIR]) / participant / "startup/start.sh"

    # The AWS helper queries region and identity before sourcing the config.
    tools = tmp_path / "bin"
    tools.mkdir()
    for command in ("aws", "az", "sshpass", "dig", "jq"):
        stub = tools / command
        stub.write_text('#!/bin/sh\n[ "$*" = "configure get region" ] || [ "$*" = "sts get-caller-identity" ]\n')
        stub.chmod(0o755)

    config = tmp_path / "FLARE #1" / "cloud config.sh"
    config.parent.mkdir()
    config.write_text('echo "Loaded the selected config"\nexit 0\n')
    result = run_owned_process(
        [str(start), "--cloud", cloud, "--config", str(config)],
        cwd=tmp_path,
        env={"PATH": f"{tools}{os.pathsep}{os.defpath}", "TMPDIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Loaded the selected config" in result.stdout
