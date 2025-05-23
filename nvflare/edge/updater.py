# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class Updater(FLComponent, ABC):

    def __init__(self):
        FLComponent.__init__(self)
        self.current_state = None

    def start_task(self, task_data: Shareable, fl_ctx: FLContext) -> Any:
        """This is called by HUG at the start of a task.

        Args:
            task_data: the task received from parent
            fl_ctx: FLContext object

        Returns: initialized task data

        """
        self.current_state = task_data
        return task_data

    def get_current_state(self, fl_ctx: FLContext) -> Any:
        """Get the current state of the updater, which will be used as the value of the "task" in TaskInfo
        for edge devices.

        Args:
            fl_ctx: FLContext object

        Returns: the current state of the updater

        """
        return self.current_state

    @abstractmethod
    def prepare_update_for_parent(self, fl_ctx: FLContext) -> Optional[Shareable]:
        """This is called by HUG to prepare update report to be sent to the parent.


        Args:
            fl_ctx: FLContext object

        Returns: a Shareable object for the update report; or None if no update

        """
        pass

    @abstractmethod
    def process_parent_update_reply(self, reply: Shareable, fl_ctx: FLContext):
        """This is called by HUG to process the reply received from the parent for the update report sent to it.
        The reply typically contains information for how to update the current state of this Updater.

        Args:
            reply: the reply from the parent
            fl_ctx: FLContext object

        Returns: None

        """
        pass

    @abstractmethod
    def process_child_update(self, update: Shareable, fl_ctx: FLContext) -> (bool, Optional[Shareable]):
        """This is called by HUG to process an update report from a child.

        Args:
            update: the update report from the child
            fl_ctx: FLContext object

        Returns: a tuple of (whether the update is accepted, reply to the child)

        """
        pass

    def end_task(self, fl_ctx: FLContext):
        """This is called by HUG at the end of the task.

        Args:
            fl_ctx: FLContext object

        Returns: None

        """
        self.current_state = None
