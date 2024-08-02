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

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class Applet(ABC, FLComponent):

    """An Applet implements App (server or client) processing logic."""

    def __init__(self):
        FLComponent.__init__(self)

    def initialize(self, fl_ctx: FLContext):
        """Called by Controller/Executor to initialize the applet.
        This happens when the job is about to start.

        Args:
            fl_ctx: FL context

        Returns: None

        """
        pass

    @abstractmethod
    def start(self, app_ctx: dict):
        """Called to start the execution of the applet.

        Args:
            app_ctx: the contextual info to help the applet execution

        Returns: None

        """
        pass

    @abstractmethod
    def stop(self, timeout=0.0) -> int:
        """Called to stop the applet.

        Args:
            timeout: the max amount of time (seconds) to stop the applet

        Returns: the exit code after stopped

        """
        pass

    @abstractmethod
    def is_stopped(self) -> (bool, int):
        """Called to check whether the applet is already stopped.

        Returns: whether the applet is stopped, and the exit code if stopped.

        """
        pass
