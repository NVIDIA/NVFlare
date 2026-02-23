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

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.workflows.base_model_controller import BaseModelController


class _MockEngine:
    def __init__(self):
        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name="mock_engine",
            job_id="mock_job",
            public_stickers={},
            private_stickers={},
        )
        self.events = []

    def fire_event(self, event_type, fl_ctx):
        self.events.append((event_type, fl_ctx.get_prop(AppConstants.CURRENT_ROUND, None)))


class _TestController(BaseModelController):
    def run(self):
        pass


def _make_result_shareable(current_round=None):
    model = FLModel(params={"w": 1.0}, params_type=ParamsType.FULL, current_round=current_round)
    return FLModelUtils.to_shareable(model)


def test_process_result_sets_current_round_from_task_before_accept_event():
    engine = _MockEngine()
    fl_ctx = engine.fl_ctx_mgr.new_context()
    controller = _TestController()

    task_data = Shareable()
    task_data.set_header(AppConstants.CURRENT_ROUND, 7)
    task = Task(name=AppConstants.TASK_TRAIN, data=task_data)
    client_task = ClientTask(client=Client("site-1", "token"), task=task)
    client_task.result = _make_result_shareable(current_round=None)

    controller._process_result(client_task, fl_ctx)

    assert engine.events[0] == (AppEventType.BEFORE_CONTRIBUTION_ACCEPT, 7)
    assert fl_ctx.get_prop(AppConstants.CURRENT_ROUND) == 7


def test_process_result_falls_back_to_result_round_before_accept_event():
    engine = _MockEngine()
    fl_ctx = engine.fl_ctx_mgr.new_context()
    controller = _TestController()

    task = Task(name=AppConstants.TASK_TRAIN, data=Shareable())
    client_task = ClientTask(client=Client("site-1", "token"), task=task)
    client_task.result = _make_result_shareable(current_round=3)

    controller._process_result(client_task, fl_ctx)

    assert engine.events[0] == (AppEventType.BEFORE_CONTRIBUTION_ACCEPT, 3)
    assert fl_ctx.get_prop(AppConstants.CURRENT_ROUND) == 3
