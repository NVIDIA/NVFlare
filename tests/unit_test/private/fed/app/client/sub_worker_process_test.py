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

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ComponentNotAuthorized
from nvflare.fuel.common.multi_process_executor_constants import CommunicationMetaData as Meta
from nvflare.fuel.f3.cellnet.core_cell import make_reply
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellRC
from nvflare.private.fed.app.client.sub_worker_process import SubWorkerExecutor
from tests.multiprocess_test_utils import RankExecutor


@pytest.fixture
def worker(tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "meta.json").write_text("{}")
    (tmp_path / "local" / "resources.json").write_text('{"class_allow_list": ["tests.", "nvflare."]}')
    sub = SubWorkerExecutor.__new__(SubWorkerExecutor)
    sub.workspace = Workspace(str(tmp_path), "site-1")
    sub.args = SimpleNamespace(client_name="site-1", job_id="job", simulator_engine="false")
    sub.components = {}
    sub.handlers = []
    sub.local_rank = 0
    sub.num_of_processes = 2
    sub.cell = Mock()
    sub.cell.send_request.return_value = make_reply(CellRC.OK)
    sub.logger = Mock()
    return sub


def initialize(worker, root):
    return worker._initialize(
        {
            Meta.FL_CTX: FLContext(),
            Meta.LOCAL_EXECUTOR: "trainer",
            Meta.COMPONENTS: [
                {"id": "trainer", "path": "tests.multiprocess_test_utils.RankExecutor", "args": {"root": str(root)}}
            ],
        }
    )


@pytest.mark.parametrize("simulator", ["true", "false"])
def test_rank_initializes_with_cell_and_authorized_dynamic_builder(worker, tmp_path, simulator):
    worker.args.simulator_engine = simulator
    assert initialize(worker, tmp_path).get_header(MessageHeaderKey.RETURN_CODE) == CellRC.OK
    assert (tmp_path / "started-0").exists()
    with worker.run_manager.new_context() as ctx:
        assert ctx.get_engine() is worker.run_manager
        assert worker.run_manager.get_cell() is worker.cell
    with pytest.raises(ComponentNotAuthorized):
        worker.run_manager.build_component({"path": "socket.socket"})


@pytest.mark.parametrize("rank", [0, 1])
def test_each_rank_reports_application_exception(worker, tmp_path, rank):
    initialize(worker, tmp_path)
    worker.local_rank = rank
    worker.executor = Mock(spec=RankExecutor)
    worker.executor.execute.side_effect = RuntimeError("training failed")
    worker._execute_task(
        {Meta.TASK_NAME: "train", Meta.SHAREABLE: Shareable(), Meta.FL_CTX: FLContext(), Meta.EXECUTION_ID: "attempt"}
    )
    report = worker.cell.send_request.call_args.kwargs["request"].payload
    assert report[Meta.RANK_NUMBER] == rank
    assert report[Meta.EXECUTION_ID] == "attempt"
    assert report[Meta.SHAREABLE].get_return_code() == ReturnCode.EXECUTION_EXCEPTION
