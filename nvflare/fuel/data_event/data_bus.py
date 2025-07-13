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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Union

from nvflare.fuel.data_event.pub_sub import EventPubSub


class DataBus(EventPubSub):
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
        The databus


        """
        with cls._lock:
            if not cls._instance:
                cls._instance = super(DataBus, cls).__new__(cls)
                cls._instance.subscribers = {}
                cls._instance.data_store = {}
        return cls._instance

    def subscribe(
        self,
        topics: List[str],
        callback: Callable[[str, Any, "DataBus"], None],
        one_shot=False,
        **cb_kwargs,
    ) -> None:
        """
        Subscribe a callback function to one or more topics.

        Args:
            topics (List[str]): A list of topics to subscribe to.
            callback (Callable): The callback function to be called when messages are published to the subscribed topics.
            one_shot: whether the callback is used once.
        """

        if not topics:
            raise ValueError("topics must non-empty")

        for topic in topics:
            if topic.isspace():
                raise ValueError(f"topics {topics}contains white space topic")

            with self._lock:
                if topic not in self.subscribers:
                    self.subscribers[topic] = []
                self.subscribers[topic].append((callback, one_shot, cb_kwargs))

        print(f"total subscribers after subscribe: {len(self.subscribers)}")

    def publish(self, topics: List[str], datum: Any) -> None:
        """
        Publish a data to one or more topics, notifying all subscribed callbacks.

        Args:
            topics (List[str]): A list of topics to publish the data to.
            datum (Any): The data to be published to the specified topics.
        """
        if not topics:
            return

        # minimize the time of lock - only manage the subscribers data structure within the lock
        # do not run the CBs within the lock
        with self._lock:
            subs_to_execute = []
            for topic in topics:
                subs_to_delete = []
                subscribers = self.subscribers.get(topic)
                if subscribers:
                    for sub in subscribers:
                        callback, one_shot, kwargs = sub
                        subs_to_execute.append((topic, callback, kwargs))
                        if one_shot:
                            subs_to_delete.append(sub)

                for sub in subs_to_delete:
                    subscribers.remove(sub)

                if not subscribers:
                    self.subscribers.pop(topic, None)

        if not subs_to_execute:
            return

        executor = ThreadPoolExecutor(max_workers=len(subs_to_execute))
        for sub in subs_to_execute:
            topic, callback, kwargs = sub
            executor.submit(callback, topic, datum, self, **kwargs)
        executor.shutdown()

    def put_data(self, key: Any, datum: Any) -> None:
        """
        Store a data associated with a key and topic.

        Args:
            key (Any): The key to associate with the stored message.
            datum (Any): The message to be stored.
        """
        with self._lock:
            self.data_store[key] = datum

    def get_data(self, key: Any) -> Any:
        """
        Retrieve a stored data associated with a key and topic.

        Args:
            key (Any): The key associated with the stored message.

        Returns:
            Any: The stored datum if found, or None if not found.
        """
        return self.data_store.get(key)


def dynamic_topic(base_topic: str, values: Union[str, List[str]]) -> str:
    parts = [base_topic]
    if isinstance(values, str):
        parts.append(values)
    else:
        parts.extend(values)
    return "_".join(parts)
