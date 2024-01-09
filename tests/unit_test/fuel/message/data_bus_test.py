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

from nvflare.fuel.message.data_bus import DataBus
from nvflare.fuel.message.event_manager import EventManager


class TestMessageBus(unittest.TestCase):
    def setUp(self):
        self.message_bus = DataBus()
        self.event_manager = EventManager(self.message_bus)

    def test_subscribe_and_publish(self):
        result = {"count": 0}

        def callback_function(message, topic="default"):
            result["count"] += 1

        self.message_bus.subscribe(["test_topic"], callback_function)
        self.message_bus.publish(["test_topic"], "Test Message 1")
        self.message_bus.publish(["test_topic"], "Test Message 2")

        self.assertEqual(result["count"], 2)

    def test_singleton_message_bus(self):
        message_bus1 = DataBus()
        message_bus1.send_message("user_1", "Hello from User 1!")
        user_1_message = message_bus1.receive_messages("user_1")
        self.assertEqual(user_1_message, "Hello from User 1!")

        message_bus2 = DataBus()
        user_1_message = message_bus2.receive_messages("user_1")
        self.assertEqual(user_1_message, "Hello from User 1!")

    def test_send_message_and_receive_messages(self):
        self.message_bus.send_message("user_1", "Hello from User 1!")
        self.message_bus.send_message("user_2", "Greetings from User 2!")

        user_1_message = self.message_bus.receive_messages("user_1")
        user_2_message = self.message_bus.receive_messages("user_2")

        self.assertEqual(user_1_message, "Hello from User 1!")
        self.assertEqual(user_2_message, "Greetings from User 2!")

        self.message_bus.send_message("user_1", "2nd greetings from User 1!")
        user_1_message = self.message_bus.receive_messages("user_1")
        self.assertEqual(user_1_message, "2nd greetings from User 1!")

        self.message_bus.send_message("user_1", "3rd greetings from User 1!", topic="channel-3")
        user_1_message = self.message_bus.receive_messages("user_1")
        self.assertEqual(user_1_message, "2nd greetings from User 1!")

        user_1_message = self.message_bus.receive_messages("user_1", topic="channel-3")
        self.assertEqual(user_1_message, "3rd greetings from User 1!")

    def test_send_message_and_receive_messages_abnormal(self):
        user_3_message = self.message_bus.receive_messages("user_3")
        self.assertEqual(user_3_message, None)

        user_3_message = self.message_bus.receive_messages("user_3", topic="channel")
        self.assertEqual(user_3_message, None)

    def test_fire_event(self):

        result = {
            "test_event": {"event_received": False},
            "dev_event": {"event_received": False},
            "prod_event": {"event_received": False},
        }

        def event_handler(data, topic):
            result[topic]["event_received"] = True

        self.message_bus.subscribe(["test_event", "dev_event", "prod_event"], event_handler)
        self.event_manager.fire_event("test_event", {"key": "value"})
        self.event_manager.fire_event("dev_event", {"key": "value"})

        self.assertTrue(result["test_event"]["event_received"])
        self.assertTrue(result["dev_event"]["event_received"])
        self.assertFalse(result["prod_event"]["event_received"])
