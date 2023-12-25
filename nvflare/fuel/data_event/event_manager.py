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

from typing import Any, Optional

from nvflare.fuel.data_event.data_bus import DataBus


class EventManager:
    """
    Class for managing events by interacting with a DataBus.

    Args:
        data_bus (DataBus): An instance of the DataBus class used for event communication.
    """

    def __init__(self, data_bus: "DataBus"):
        """
        Initialize the EventManager with a DataBus instance.

        Args:
            data_bus (DataBus): An instance of the DataBus class used for event communication.
        """
        self.data_bus = data_bus

    def fire_event(self, event_name: str, event_data: Optional[Any] = None) -> None:
        """
        Fire an event by publishing it to the DataBus.

        Args:
            event_name (str): The name of the event to be fired.
            event_data (Any, optional): Additional data associated with the event (default is None).
        """
        self.data_bus.publish([event_name], event_data)
