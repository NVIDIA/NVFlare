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
from unittest.mock import MagicMock, patch

import pytest

from nvflare.collab.runtime.worker.launcher import SubprocessLauncher


def _make_launcher(listener_url="tcp://localhost:1234", **kwargs):
    parent_cell = MagicMock()
    parent_cell.get_internal_listener_url.return_value = listener_url
    return SubprocessLauncher(
        site_name="site-1",
        training_module="training.module",
        parent_cell=parent_cell,
        **kwargs,
    )


@pytest.mark.parametrize("listener_url", [None, ""])
def test_launcher_requires_parent_internal_listener(listener_url):
    with pytest.raises(RuntimeError, match="parent cell does not have an internal listener URL"):
        _make_launcher(listener_url=listener_url)


def test_launcher_reports_listener_lookup_failure():
    parent_cell = MagicMock()
    parent_cell.get_internal_listener_url.side_effect = ValueError("bad listener")

    with pytest.raises(RuntimeError, match="failed to get the parent cell's internal listener URL") as exc_info:
        SubprocessLauncher(
            site_name="site-1",
            training_module="training.module",
            parent_cell=parent_cell,
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_subprocess_env_does_not_set_master_port():
    launcher = _make_launcher(site_index=3)

    with patch.dict(os.environ, {}, clear=True):
        env = launcher._build_subprocess_env()

    assert "MASTER_PORT" not in env


def test_torchrun_command_uses_site_specific_master_port():
    launcher = _make_launcher(run_cmd="/usr/bin/torchrun --nproc_per_node=2", site_index=3)

    command = launcher._build_subprocess_cmd()

    assert "--master-port=29503" in command
