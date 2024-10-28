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
import unittest

from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.data_event.event_manager import EventManager


class TestMessageBus(unittest.TestCase):
    def setUp(self):
        self.data_bus = DataBus()
        self.event_manager = EventManager(self.data_bus)

    def test_subscribe_and_publish(self):
        result = {"count": 0}

        def callback_function(topic, datum, data_bus):
            result["count"] += 1

        self.data_bus.subscribe(["test_topic"], callback_function)
        self.data_bus.publish(["test_topic"], "Test Message 1")
        self.data_bus.publish(["test_topic"], "Test Message 2")

        self.assertEqual(result["count"], 2)

    def test_singleton_message_bus(self):
        data_bus1 = DataBus()
        data_bus1.put_data("user_1", "Hello from User 1!")
        user_1_message = data_bus1.get_data("user_1")
        self.assertEqual(user_1_message, "Hello from User 1!")

        message_bus2 = DataBus()
        user_1_message = message_bus2.get_data("user_1")
        self.assertEqual(user_1_message, "Hello from User 1!")

    def test_send_message_and_receive_messages(self):
        self.data_bus.put_data("user_1", "Hello from User 1!")
        self.data_bus.put_data("user_2", "Greetings from User 2!")

        user_1_message = self.data_bus.get_data("user_1")
        user_2_message = self.data_bus.get_data("user_2")

        self.assertEqual(user_1_message, "Hello from User 1!")
        self.assertEqual(user_2_message, "Greetings from User 2!")

        self.data_bus.put_data("user_1", "2nd greetings from User 1!")
        user_1_message = self.data_bus.get_data("user_1")
        self.assertEqual(user_1_message, "2nd greetings from User 1!")

    def test_send_message_and_receive_messages_abnormal(self):
        user_3_message = self.data_bus.get_data("user_3")
        self.assertEqual(user_3_message, None)

    def test_fire_event(self):

        result = {
            "test_event": {"event_received": False},
            "dev_event": {"event_received": False},
            "prod_event": {"event_received": False},
        }

        def event_handler(topic, data, data_bus):
            result[topic]["event_received"] = True
            if data_bus.get_data("hi") == "hello":
                self.data_bus.put_data("hi", "hello-world")

        self.data_bus.put_data("hi", "hello")

        self.data_bus.subscribe(["test_event", "dev_event", "prod_event"], event_handler)
        self.event_manager.fire_event("test_event", {"key": "value"})
        self.event_manager.fire_event("dev_event", {"key": "value"})

        self.assertTrue(result["test_event"]["event_received"])
        self.assertTrue(result["dev_event"]["event_received"])
        self.assertFalse(result["prod_event"]["event_received"])
