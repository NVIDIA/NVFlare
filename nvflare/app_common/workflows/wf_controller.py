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

from abc import ABC, abstractmethod
from typing import Callable, List, Union

from nvflare.apis.wf_controller_spec import WFControllerSpec
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.model_controller import ModelController


class WFController(ModelController, WFControllerSpec, ABC):
    """Workflow Controller for FLModel based ModelController."""

    @abstractmethod
    def run(self):
        """Main `run` routine for the controller workflow."""
        raise NotImplementedError

    def send_model(
        self,
        task_name: str = "train",
        data: FLModel = None,
        targets: Union[List[str], None] = None,
        timeout: int = 0,
        wait_time_after_min_received: int = 10,
        blocking: bool = True,
        callback: Callable[[FLModel], None] = None,
    ) -> Union[List[FLModel], None]:
        """Send a task with data to targets.

        Args:
            task_name (str, optional): name of the task. Defaults to "train".
            data (FLModel, optional): FLModel to be sent to clients. Defaults to None.
            targets (List[str], optional): the list of target client names or None (all clients). Defaults to None.
            timeout (int, optional): time to wait for clients to perform task. Defaults to 0 (never time out).
            wait_time_after_min_received (int, optional): time to wait after minimum number of client responses have been received. Defaults to 10.
            blocking (bool, optional): whether to block to wait for task result. Defaults to True.
            callback (Callable[[FLModel], None], optional): callback when a result is received. Only called when blocking=False. Defaults to None.

        Returns:
            List[FLModel] if blocking = True else None
        """
        return super().send_model(
            task_name=task_name,
            data=data,
            targets=targets,
            timeout=timeout,
            wait_time_after_min_received=wait_time_after_min_received,
            blocking=blocking,
            callback=callback,
        )
