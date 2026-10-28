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


"""Isolated worker startup shared by unit checks and process integration tests."""

import logging
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nvflare.private.fed.app.client import worker_process
from nvflare.private.fed.client.client_app_runner import ClientAppRunner


@contextmanager
def _worker_runtime(workspace_root, runner, mock_upload=True):
    """Use the real worker/app runner with isolated startup and transport."""
    args = SimpleNamespace(
        set=[],
        workspace=str(workspace_root),
        client_name="site-1",
        job_id="job-1",
        token="test",
        token_signature="test",
        ssid="test",
        sp_target="localhost:8002",
        sp_scheme="tcp",
    )
    client = MagicMock()
    config = MagicMock()
    config.base_deployer.create_fed_client.return_value = client
    workspace = MagicMock()
    workspace.get_app_dir.return_value = str(workspace_root / "job-1" / "app_site-1")
    workspace.get_file_path_in_root.side_effect = lambda name: str(workspace_root / name)

    with ExitStack() as stack:
        for name in (
            "download_workspace",
            "refresh_custom_dir_import_path",
            "set_stats_pool_config_for_job",
            "fobs_initialize",
            "security_init_for_job",
            "register_ext_decomposers",
            "configure_logging",
        ):
            stack.enter_context(patch.object(worker_process, name))
        if mock_upload:
            stack.enter_context(patch.object(worker_process, "upload_results_on_shutdown"))
        stack.enter_context(patch.object(worker_process, "Workspace", return_value=workspace))
        stack.enter_context(patch.object(worker_process, "FLClientStarterConfiger", return_value=config))
        stack.enter_context(patch.object(worker_process, "create_stats_pool_files_for_job", return_value=None))
        stack.enter_context(patch.object(worker_process, "get_script_logger", return_value=logging.getLogger("worker")))
        stack.enter_context(
            patch.object(worker_process, "monitor_parent_process", lambda _runner, _pid, event: event.wait())
        )
        stack.enter_context(patch.object(ClientAppRunner, "create_client_runner", return_value=runner))
        stack.enter_context(patch.object(ClientAppRunner, "start_command_agent"))
        stack.enter_context(patch.object(ClientAppRunner, "sync_up_parents_process"))
        stack.enter_context(patch.object(ClientAppRunner, "notify_job_status"))
        yield args, client
