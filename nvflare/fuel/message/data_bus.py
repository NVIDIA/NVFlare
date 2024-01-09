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
from typing import Callable, List


class DataBus:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(DataBus, cls).__new__(cls)
                cls._instance.subscribers = {}
                cls._instance.message_store = {}
        return cls._instance

    def subscribe(self, topics: List[str], callback: Callable):
        if topics:
            for topic in topics:
                if topic not in self.subscribers:
                    self.subscribers[topic] = []
                self.subscribers[topic].append(callback)

    def publish(self, topics: List[str], message: any):
        if topics:
            for topic in topics:
                if topic in self.subscribers:
                    for callback in self.subscribers[topic]:
                        callback(message, topic)

    def send_message(self, key, message, topic: str = "default"):
        if topic not in self.message_store:
            self.message_store[topic] = {}

        self.message_store[topic][key] = message

    def receive_messages(self, key, topic: str = "default"):
        return self.message_store.get(topic, {}).get(key)
