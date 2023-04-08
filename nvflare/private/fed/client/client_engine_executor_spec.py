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

import time
from abc import ABC, abstractmethod
from typing import List, Union

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.apis.engine_spec import EngineSpec
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.workspace import Workspace
from nvflare.widgets.widget import Widget


class TaskAssignment(object):
    def __init__(self, name: str, task_id: str, data: Shareable):
        """Init TaskAssignment.

        Keeps track of information about the assignment of a task, including the time
        that it was created after being fetched by the Client Run Manager.

        Args:
            name: task name
            task_id: task id
            data: the Shareable data for the task assignment
        """
        self.name = name
        self.task_id = task_id
        self.data = data
        self.receive_time = time.time()


class ClientEngineExecutorSpec(ClientEngineSpec, EngineSpec, ABC):
    """The ClientEngineExecutorSpec defines the ClientEngine APIs running in the child process."""

    @abstractmethod
    def get_task_assignment(self, fl_ctx: FLContext) -> TaskAssignment:
        pass

    @abstractmethod
    def send_task_result(self, result: Shareable, fl_ctx: FLContext) -> bool:
        pass

    @abstractmethod
    def get_workspace(self) -> Workspace:
        pass

    @abstractmethod
    def get_widget(self, widget_id: str) -> Widget:
        pass

    @abstractmethod
    def get_all_components(self) -> dict:
        pass

    @abstractmethod
    def register_aux_message_handler(self, topic: str, message_handle_func):
        """Register aux message handling function with specified topics.

        Exception is raised when:
            a handler is already registered for the topic;
            bad topic - must be a non-empty string
            bad message_handle_func - must be callable

        Implementation Note:
        This method should simply call the ClientAuxRunner's register_aux_message_handler method.

        Args:
            topic: the topic to be handled by the func
            message_handle_func: the func to handle the message. Must follow aux_message_handle_func_signature.

        """
        pass

    @abstractmethod
    def send_aux_request(
        self,
        targets: Union[None, str, List[str]],
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        optional=False,
    ) -> dict:
        """Send a request to Server via the aux channel.

        Implementation: simply calls the ClientAuxRunner's send_aux_request method.

        Args:
            targets: aux messages targets. None or empty list means the server.
            topic: topic of the request
            request: request to be sent
            timeout: number of secs to wait for replies. 0 means fire-and-forget.
            fl_ctx: FL context
            optional: whether the request is optional

        Returns:
            a dict of reply Shareable in the format of:
                { site_name: reply_shareable }

        """
        pass

    @abstractmethod
    def fire_and_forget_aux_request(
        self, topic: str, request: Shareable, fl_ctx: FLContext, optional=False
    ) -> Shareable:
        """Send an async request to Server via the aux channel.

        Args:
            topic: topic of the request
            request: request to be sent
            fl_ctx: FL context
            optional: whether the request is optional

        Returns:

        """
        pass

    @abstractmethod
    def build_component(self, config_dict):
        """Build a component from the config_dict.

        Args:
            config_dict: config dict

        """
        pass

    @abstractmethod
    def abort_app(self, job_id: str, fl_ctx: FLContext):
        """Abort the running FL App on the client.

        Args:
            job_id: current_job_id
            fl_ctx: FLContext

        """
        pass
