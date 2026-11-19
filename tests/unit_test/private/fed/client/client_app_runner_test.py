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

"""Select task accounting at the existing validated configuration boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.private.fed.client import client_app_runner


@pytest.mark.parametrize("lifetime", ["job", "task"])
def test_resource_policy_is_selected_after_admission_before_component_construction(tmp_path, monkeypatch, lifetime):
    events = []
    config = MagicMock(config_data={"execution_lifetime": lifetime})
    config.configure.side_effect = RuntimeError("stop before constructing components")
    monkeypatch.setattr(client_app_runner, "Workspace", MagicMock())
    monkeypatch.setattr(client_app_runner, "ClientJsonConfigurator", lambda **_kwargs: config)
    monkeypatch.setattr(client_app_runner, "get_job_meta_from_workspace", lambda *_args: {})
    monkeypatch.setattr(client_app_runner, "validate_cj_execution_lifetime", lambda *_args: events.append("admission"))
    monkeypatch.setattr(client_app_runner, "configure_task_collection", lambda: events.append("task statistics"))
    args = SimpleNamespace(
        workspace=str(tmp_path), client_name="site", job_id="job", client_config="config.json", set=[]
    )
    with pytest.raises(RuntimeError, match="stop before constructing"):
        client_app_runner.ClientAppRunner().create_client_runner(str(tmp_path), args, "config", MagicMock(), False)
    assert events == (["admission", "task statistics"] if lifetime == "task" else ["admission"])
    config.configure.assert_called_once_with()
