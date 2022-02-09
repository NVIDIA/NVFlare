# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Tuple

from nvflare.apis.shareable import Shareable
from nvflare.widgets.widget import Widget

from .client import Client
from .fl_context import FLContext
from .workspace import Workspace


class ServerEngineSpec(ABC):
    def fire_event(self, event_type: str, fl_ctx: FLContext):
        pass

    @abstractmethod
    def get_clients(self) -> List[Client]:
        pass

    @abstractmethod
    def validate_clients(self, client_names: List[str]) -> Tuple[List[Client], List[str]]:
        """Validate specified client names.

        Args:
            client_names: list of names to be validated

        Returns: a list of validate clients  and a list of invalid client names

        """
        pass

    @abstractmethod
    def new_context(self) -> FLContext:
        # the engine must use FLContextManager to create a new context!
        pass

    @abstractmethod
    def get_workspace(self) -> Workspace:
        pass

    @abstractmethod
    def get_component(self, component_id: str) -> object:
        pass

    @abstractmethod
    def register_aux_message_handler(self, topic: str, message_handle_func):
        """Register aux message handling function with specified topics.

        Exception is raised when:
            a handler is already registered for the topic;
            bad topic - must be a non-empty string
            bad message_handle_func - must be callable

        Implementation Note:
            This method should simply call the ServerAuxRunner's register_aux_message_handler method.

        Args:
            topic: the topic to be handled by the func
            message_handle_func: the func to handle the message. Must follow aux_message_handle_func_signature.

        """
        pass

    @abstractmethod
    def send_aux_request(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> dict:
        """Send a request to specified clients via the aux channel.

        Implementation: simply calls the ServerAuxRunner's send_aux_request method.

        Args:
            targets: target clients. None or empty list means all clients
            topic: topic of the request
            request: request to be sent
            timeout: number of secs to wait for replies. 0 means fire-and-forget.
            fl_ctx: FL context

        Returns: a dict of replies (client name => reply Shareable)

        """
        pass

    def fire_and_forget_aux_request(self, targets: [], topic: str, request: Shareable, fl_ctx: FLContext) -> dict:
        return self.send_aux_request(targets, topic, request, 0.0, fl_ctx)

    @abstractmethod
    def get_widget(self, widget_id: str) -> Widget:
        """Get the widget with the specified ID.

        Args:
            widget_id: ID of the widget

        Returns: the widget or None if not found

        """
        pass
