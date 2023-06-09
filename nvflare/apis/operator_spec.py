# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ControllerSpec
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class OperatorSpec(ABC):
    @abstractmethod
    def operate(
        self,
        op_description: dict,
        controller: ControllerSpec,
        task_name: str,
        task_data: Shareable,
        abort_signal: Signal,
        fl_ctx: FLContext,
    ) -> Union[Shareable, None]:
        pass

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass
