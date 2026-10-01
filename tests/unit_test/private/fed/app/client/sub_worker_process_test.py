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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.common.multi_process_executor_constants import CommunicationMetaData
from nvflare.fuel.f3.cellnet.core_cell import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.private.fed.app.client import sub_worker_process as module


def test_event_reply_is_sent_after_rank_handlers_complete():
    worker = module.SubWorkerExecutor.__new__(module.SubWorkerExecutor)
    worker.run_manager = Mock()
    relayer = worker.run_manager.get_component.return_value
    completed = []
    relayer.relay_event.side_effect = lambda *_args: completed.append("handled")
    data = {"event": "end_run"}

    reply = worker._handle_event(data)

    worker.run_manager.get_component.assert_called_once_with(CommunicationMetaData.RELAYER)
    relayer.relay_event.assert_called_once_with(worker.run_manager, data)
    assert completed == ["handled"]
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK


def test_failed_event_handler_does_not_acknowledge_cleanup():
    worker = module.SubWorkerExecutor.__new__(module.SubWorkerExecutor)
    worker.run_manager = Mock()
    worker.run_manager.get_component.return_value.relay_event.side_effect = RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        worker._handle_event({})


@pytest.mark.parametrize("run_fails", [False, True])
def test_rank_main_stops_and_joins_parent_monitor_even_when_run_fails(monkeypatch, run_fails):
    args = SimpleNamespace(
        workspace="workspace",
        client_name="site-1",
        job_id="job-1",
        decomposer_module="",
        num_processes="2",
        parent_pid=1,
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    workspace = Mock()
    monkeypatch.setattr(module, "Workspace", Mock(return_value=workspace))
    for name in (
        "configure_logging",
        "fobs_initialize",
        "register_ext_decomposers",
        "create_privacy_manager",
        "set_stats_pool_config_for_job",
    ):
        monkeypatch.setattr(module, name, Mock())
    monkeypatch.setattr(module.SecurityContentService, "initialize", Mock())
    monkeypatch.setattr(module.AuditService, "initialize", Mock())
    monkeypatch.setattr(module.AuditService, "close", Mock())
    monkeypatch.setattr(module.PrivacyService, "initialize", Mock())
    monkeypatch.setattr(module, "create_stats_pool_files_for_job", Mock(return_value=None))
    worker = Mock()
    monkeypatch.setattr(module, "SubWorkerExecutor", Mock(return_value=worker))
    if run_fails:
        worker.run.side_effect = RuntimeError("rank failed")

    monitor = Mock()
    thread_factory = Mock(return_value=monitor)
    monkeypatch.setattr(module.threading, "Thread", thread_factory)

    def check_stop_before_join(timeout):
        stop_event = thread_factory.call_args.kwargs["args"][2]
        assert stop_event.is_set()
        assert timeout == 2.0

    monitor.join.side_effect = check_stop_before_join

    if run_fails:
        with pytest.raises(RuntimeError, match="rank failed"):
            module.main(args)
    else:
        module.main(args)

    monitor.start.assert_called_once()
    monitor.join.assert_called_once_with(timeout=2.0)
