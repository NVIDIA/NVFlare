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

from nvflare.apis.fl_context import FLContext


class AppRunner(ABC):

    """An AppRunner implements App (server or client) processing logic."""

    def initialize(self, fl_ctx: FLContext):
        """Called by Controller/Executor to initialize the runner.
        This happens when the job is about to start.

        Args:
            fl_ctx: FL context

        Returns: None

        """
        pass

    @abstractmethod
    def run(self, ctx: dict):
        """Called to start the execution of app processing logic.

        Args:
            ctx: the contextual info to help the runner execution

        Returns: None

        """
        pass

    @abstractmethod
    def stop(self):
        """Called to stop the runner.

        Returns:

        """
        pass

    @abstractmethod
    def is_stopped(self) -> (bool, int):
        """Called to check whether the runner is already stopped.

        Returns: whether the runner is stopped. If stopped, the exit code.

        """
        pass
