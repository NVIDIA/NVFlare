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

import os
from abc import ABC, abstractmethod
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace


class Launcher(ABC):
    def __init__(self):
        self.fl_ctx = None

    def initialize(self, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx

    def finalize(self, fl_ctx: FLContext) -> None:
        pass

    def get_app_dir(self) -> Optional[str]:
        fl_ctx = self.fl_ctx
        if fl_ctx is not None:
            workspace: Workspace = fl_ctx.get_engine().get_workspace()
            app_dir = workspace.get_app_dir(fl_ctx.get_job_id())
            if app_dir is not None:
                return os.path.abspath(app_dir)
        return None

    @abstractmethod
    def launch_task(self, task_name: str, shareable: Shareable, abort_signal: Signal) -> bool:
        """Launches external system to handle a task.

        Returns:
            Whether launch success or not.
        """
        pass

    @abstractmethod
    def stop_task(self, task_name: str) -> None:
        """Stops external system and free up resources."""
        pass
