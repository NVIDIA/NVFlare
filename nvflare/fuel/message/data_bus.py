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
import threading
from typing import Any, Callable, List


class DataBus:
    """
    Singleton class for a simple data bus implementation.

    This class allows components to subscribe to topics, publish messages to topics,
    and store/retrieve messages associated with specific keys and topics.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "DataBus":
        """
        Create a new instance of the DataBus class.

        This method ensures that only one instance of the class is created (singleton pattern).
        """
        with cls._lock:
            if not cls._instance:
                cls._instance = super(DataBus, cls).__new__(cls)
                cls._instance.subscribers = {}
                cls._instance.message_store = {}
        return cls._instance

    def subscribe(self, topics: List[str], callback: Callable) -> None:
        """
        Subscribe a callback function to one or more topics.

        Args:
            topics (List[str]): A list of topics to subscribe to.
            callback (Callable): The callback function to be called when messages are published to the subscribed topics.
        """
        if topics:
            for topic in topics:
                if topic not in self.subscribers:
                    self.subscribers[topic] = []
                self.subscribers[topic].append(callback)

    def publish(self, topics: List[str], message: Any) -> None:
        """
        Publish a message to one or more topics, notifying all subscribed callbacks.

        Args:
            topics (List[str]): A list of topics to publish the message to.
            message (Any): The message to be published to the specified topics.
        """
        if topics:
            for topic in topics:
                if topic in self.subscribers:
                    for callback in self.subscribers[topic]:
                        callback(message, topic)

    def send_message(self, key: Any, message: Any, topic: str = "default") -> None:
        """
        Store a message associated with a key and topic.

        Args:
            key (Any): The key to associate with the stored message.
            message (Any): The message to be stored.
            topic (str): The topic under which the message is stored (default is "default").
        """
        if topic not in self.message_store:
            self.message_store[topic] = {}

        self.message_store[topic][key] = message

    def receive_message(self, key: Any, topic: str = "default") -> Any:
        """
        Retrieve a stored message associated with a key and topic.

        Args:
            key (Any): The key associated with the stored message.
            topic (str): The topic under which the message is stored (default is "default").

        Returns:
            Any: The stored message if found, or None if not found.
        """
        return self.message_store.get(topic, {}).get(key)
