# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.signal import Signal

from .fl_component import FLComponent
from .fl_context import FLContext
from .shareable import Shareable


class Executor(FLComponent, ABC):
    """Executors run on federated client side.

    Each job can contain multiple applications or apps folder.
    Each site (server or client) will have 1 app deployed for that job.
    The server side app contains a Controller that will schedule `Task`.
    The client side app contains an Executor that will execute corresponding logic based on `Task`'s name.

    """

    def __init__(self):
        FLComponent.__init__(self)
        self.unsafe = False

    @abstractmethod
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Executes a task.

        Args:
            task_name (str): task name.
            shareable (Shareable): input shareable.
            fl_ctx (FLContext): fl context.
            abort_signal (Signal): signal to check during execution to determine whether this task is aborted.

        Returns:
            An output shareable.
        """
        pass
