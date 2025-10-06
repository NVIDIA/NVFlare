# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class ExampleExecutor(Executor):
    def __init__(self):
        super().__init__()

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        if task_name == "train":
            return self.train(shareable, fl_ctx, abort_signal)
        else:
            raise ValueError(f"Unknown task name: {task_name}")

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        fl_model = FLModelUtils.from_shareable(shareable)

        params = fl_model.params
        if "state_dict" in params:
            params = params["state_dict"]

        print(f"Total tensors: {len(params)}")

        print("Building model...")
        output_model = FLModel(params={k: v.cpu() for k, v in params.items()})

        return FLModelUtils.to_shareable(output_model)
