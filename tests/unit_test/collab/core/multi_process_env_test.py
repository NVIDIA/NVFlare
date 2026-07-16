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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.collab.core.multi_process_env import MultiProcessEnv


def test_rejects_missing_client_configuration():
    with pytest.raises(ValueError, match="num_clients must be greater than 0"):
        MultiProcessEnv(num_clients=None)


@pytest.mark.parametrize(
    "clean_up,poc_running,workspace_removed,raises",
    [
        (False, False, False, False),
        (True, True, False, True),
        (True, False, True, False),
    ],
)
@patch("nvflare.collab.core.multi_process_env._clean_poc")
@patch("nvflare.collab.core.multi_process_env._stop_poc")
@patch("nvflare.collab.core.multi_process_env.is_poc_running")
@patch("nvflare.collab.core.multi_process_env.setup_service_config")
def test_stop_only_removes_workspace_when_requested_and_stopped(
    mock_setup,
    mock_is_running,
    mock_stop_poc,
    mock_clean_poc,
    tmp_path,
    clean_up,
    poc_running,
    workspace_removed,
    raises,
):
    mock_setup.return_value = ({"name": "test"}, {"server": "server"})
    mock_is_running.return_value = poc_running

    workspace = tmp_path / "poc"
    workspace.mkdir()
    env = MultiProcessEnv()
    env.poc_workspace = str(workspace)

    with (
        patch("nvflare.collab.core.multi_process_env.STOP_POC_TIMEOUT", 1),
        patch("nvflare.collab.core.multi_process_env.time.sleep"),
    ):
        if raises:
            with pytest.raises(RuntimeError, match="POC is still running"):
                env.stop(clean_up=clean_up)
        else:
            env.stop(clean_up=clean_up)

    assert workspace.exists() is not workspace_removed
    mock_stop_poc.assert_called_once()
    if workspace_removed:
        mock_clean_poc.assert_called_once_with(str(workspace))
    else:
        mock_clean_poc.assert_not_called()


def test_stop_propagates_shutdown_failure_and_clears_session_manager(tmp_path):
    workspace = tmp_path / "poc"
    workspace.mkdir()
    env = MultiProcessEnv()
    env.poc_workspace = str(workspace)
    env._session_manager = MagicMock()

    with (
        patch("nvflare.collab.core.multi_process_env.setup_service_config", return_value=({}, {})),
        patch("nvflare.collab.core.multi_process_env._stop_poc", side_effect=RuntimeError("shutdown failed")),
        pytest.raises(RuntimeError, match="shutdown failed"),
    ):
        env.stop(clean_up=True)

    assert env._session_manager is None


def test_deploy_refuses_to_provision_while_existing_poc_is_running():
    env = MultiProcessEnv()
    job = MagicMock(name="job")

    with (
        patch.object(env, "_check_poc_running", side_effect=[True, True]),
        patch.object(env, "stop") as mock_stop,
        patch("nvflare.collab.core.multi_process_env.prepare_poc_provision") as mock_prepare,
        pytest.raises(RuntimeError, match="existing POC services are still running"),
    ):
        env.deploy(job)

    mock_stop.assert_called_once_with(clean_up=True)
    mock_prepare.assert_not_called()
