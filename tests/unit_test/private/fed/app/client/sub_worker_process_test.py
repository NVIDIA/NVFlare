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

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.decomposers.flare_decomposers import ContextDecomposer
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ComponentNotAuthorized
from nvflare.fuel.common.multi_process_executor_constants import CommunicationMetaData as Meta
from nvflare.fuel.f3.cellnet.core_cell import make_reply
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import fobs as fobs_registry
from nvflare.fuel.utils.tree_utils import Forest, Node
from nvflare.private.fed.app.client.sub_worker_process import SubWorkerExecutor


class StartupExecutor(Executor):
    started = False

    def __init__(self, fail_startup=False):
        super().__init__()
        self.fail_startup = fail_startup

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.START_RUN:
            if self.fail_startup:
                raise RuntimeError("startup handler failed")
            engine = fl_ctx.get_engine()
            assert engine.get_cell() is not None
            engine.build_component(
                {"path": "nvflare.app_common.widgets.component_path_authorizer.ComponentPathAuthorizer"}
            )
            self.started = True

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        return Shareable()


@pytest.mark.parametrize("simulator", ["true", "false"])
@pytest.mark.parametrize("fail_startup", [False, True])
def test_sub_worker_initialization_preserves_runtime_services_and_authorization(tmp_path, simulator, fail_startup):
    for folder in ["startup", "local", "job"]:
        (tmp_path / folder).mkdir()
    (tmp_path / "job" / "meta.json").write_text("{}")
    (tmp_path / "local" / "resources.json").write_text(
        '{"class_allow_list": ["' + __name__ + '.", "nvflare.app_common.widgets."]}'
    )
    worker = SubWorkerExecutor.__new__(SubWorkerExecutor)
    worker.workspace = Workspace(str(tmp_path), "site-1")
    worker.args = SimpleNamespace(client_name="site-1", job_id="job", simulator_engine=simulator)
    worker.components = {}
    worker.handlers = []
    worker.local_rank = 0
    worker.num_of_processes = 2
    worker.cell = Mock()
    worker.cell.send_request.return_value = make_reply(ReturnCode.OK)
    parent_context = FLContext()
    parent_context.set_prop(
        FLContextKey.EXCEPTIONS, {"earlier_parent_event": RuntimeError("earlier error")}, sticky=False
    )
    data = {
        Meta.FL_CTX: parent_context,
        Meta.LOCAL_EXECUTOR: "trainer",
        Meta.COMPONENTS: [
            {"id": "trainer", "path": __name__ + ".StartupExecutor", "args": {"fail_startup": fail_startup}}
        ],
    }
    if fail_startup:
        with pytest.raises(RuntimeError, match="rank startup handlers failed"):
            worker._initialize(data)
        assert not worker.executor.started
        return
    reply = worker._initialize(data)
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert worker.executor.started  # Event dispatch catches handler exceptions.
    assert worker.run_manager.get_cell() is worker.cell
    worker.cell.send_request.assert_called_once()
    assert parent_context.get_prop(FLContextKey.EXCEPTIONS)
    with pytest.raises(ComponentNotAuthorized):
        worker.run_manager.build_component({"path": "socket.socket"})


def test_rank_context_decodes_without_parent_hierarchy_decomposers(monkeypatch):
    # Isolate sender registrations, then decode as a fresh rank would.
    monkeypatch.setattr(fobs_registry, "_decomposers", {})
    monkeypatch.setattr(fobs_registry, "_dot_handlers", {})
    fobs.register(ContextDecomposer)
    hierarchy = Forest()
    hierarchy.roots = ["site-1"]
    hierarchy.nodes["site-1"] = Node("site-1", None)
    parent_context = FLContext()
    parent_context.set_prop(FLContextKey.CLIENT_HIERARCHY, hierarchy, private=True, sticky=True)
    parent_context.set_prop(FLContextKey.CURRENT_JOB_ID, "job", private=False, sticky=True)
    encoded = fobs.dumps(get_serializable_data(parent_context))

    monkeypatch.setattr(fobs_registry, "_decomposers", {})
    fobs.register(ContextDecomposer)
    rank_context = fobs.loads(encoded)

    assert rank_context.get_prop(FLContextKey.CLIENT_HIERARCHY) is None
    assert rank_context.get_prop(FLContextKey.CURRENT_JOB_ID) == "job"
    assert parent_context.get_prop(FLContextKey.CLIENT_HIERARCHY) is hierarchy
