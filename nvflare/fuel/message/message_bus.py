# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


class MessageBus:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(MessageBus, cls).__new__(cls)
                # Initialize the message bus here
        return cls._instance

    def __init__(self):
        self.subscribers = {}
        self.message_store = {}

    def subscribe(self, topic, callback):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def publish(self, topic, message):
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(message)

    def send_message(self, key, message, topic: str = "default"):
        if topic not in self.message_store:
            self.message_store[topic] = {}

        self.message_store[topic][key] = message

        self.publish(key, message)  # Notify subscribers about the new message

    def receive_messages(self, key, topic: str = "default"):
        return self.message_store.get(topic, {}).get(key)
