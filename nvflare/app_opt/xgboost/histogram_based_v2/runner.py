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
from typing import Tuple

from nvflare.apis.fl_context import FLContext


class XGBRunner(ABC):

    """An XGBRunner implements XGB (server or client) processing logic."""

    def initialize(self, fl_ctx: FLContext):
        """Initializes the runner.
        This happens when the job is about to start.

        Args:
            fl_ctx: FL context

        Returns:
            None
        """
        pass

    @abstractmethod
    def run(self, ctx: dict):
        """Runs XGB processing logic.

        Args:
            ctx: the contextual info to help the runner execution

        Returns:
            None
        """
        pass

    @abstractmethod
    def stop(self):
        """Stops the runner.

        Returns:
            None
        """
        pass

    @abstractmethod
    def is_stopped(self) -> Tuple[bool, int]:
        """Checks whether the runner is already stopped.

        Returns:
            A tuple of (whether the runner is stopped, exit code)

        """
        pass
