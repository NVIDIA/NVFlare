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
from typing import Any, Callable, List


class EventPubSub:
    def subscribe(self, topics: List[str], callback: Callable[[str, Any, "DataBus"], None]) -> None:
        """
        Subscribe a callback function to one or more topics.

        Args:
            topics (List[str]): A list of topics to subscribe to.
            callback (Callable): The callback function to be called when messages are published to the subscribed topics.
        """

    def publish(self, topics: List[str], datum: Any) -> None:
        """
        Publish a message to one or more topics, notifying all subscribed callbacks.

        Args:
            topics (List[str]): A list of topics to publish the message to.
            datum (Any): The message to be published to the specified topics.
        """
