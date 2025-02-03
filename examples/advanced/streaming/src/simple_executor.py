# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class SimpleExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.aborted = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABORT_TASK:
            self.log_info(fl_ctx, "Trainer is aborted")
            self.aborted = True

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # This is a dummy executor which does nothing
        self.log_info(fl_ctx, f"Executor is called with task {task_name}")
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={})
        return dxo.to_shareable()
