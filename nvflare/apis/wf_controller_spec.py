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


class WFControllerSpec(ABC):
    @abstractmethod
    def run(self):
        """Main `run` routine for the controller workflow."""
        raise NotImplementedError

    def send_model(
        self,
        task_name: str,
        data: any,
        targets: Union[List[str], None],
        timeout: int,
        wait_time_after_min_received: int,
        blocking: bool,
        callback: Callable,
    ) -> List:
        """Send a task with data to a list of targets.

        Args:
            task_name (str): name of the task.
            data (any): data to be sent to clients.
            targets (List[str]): the list of target client names.
            timeout (int): time to wait for clients to perform task.
            wait_time_after_min_received (int): time to wait after
                minimum number of clients responses has been received.
            blocking (bool): whether to block to wait for task result.
            callback (Callable[any]): callback when a result is received, only called when blocking=False.

        Returns:
            List[any] if blocking=True else None
        """
        raise NotImplementedError
