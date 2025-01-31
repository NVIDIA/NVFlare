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

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.client.config import ConfigKey
from nvflare.client.in_process.api import (
    TOPIC_ABORT,
    TOPIC_GLOBAL_RESULT,
    TOPIC_LOCAL_RESULT,
    TOPIC_LOG_DATA,
    TOPIC_STOP,
    InProcessClientAPI,
)
from nvflare.fuel.data_event.data_bus import DataBus


class TestInProcessClientAPI(unittest.TestCase):
    def setUp(self):
        # Create a mock task_metadata for testing
        self.task_metadata = {
            FLMetaKey.JOB_ID: "123",
            FLMetaKey.SITE_NAME: "site-1",
            "TASK_NAME": "train",
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.TRAIN_WITH_EVAL: "train_with_eval",
                ConfigKey.EXCHANGE_FORMAT: "pytorch",
                ConfigKey.TRANSFER_TYPE: "DIFF",
                ConfigKey.TRAIN_TASK_NAME: "train",
                ConfigKey.EVAL_TASK_NAME: "evaluate",
                ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
            },
        }

    def test_init(self):
        # Test the initialization of InProcessClientAPI
        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()
        assert client_api.get_site_name() == "site-1"
        assert client_api.get_task_name() == "train"
        assert client_api.get_job_id() == "123"
        assert client_api.is_train() is True
        assert client_api.is_evaluate() is False
        assert client_api.is_submit_model() is False

        assert client_api.sys_info == {FLMetaKey.JOB_ID: "123", FLMetaKey.SITE_NAME: "site-1"}

    def test_init_with_custom_interval(self):
        # Test initialization with a custom result_check_interval
        client_api = InProcessClientAPI(self.task_metadata, result_check_interval=5.0)
        self.assertEqual(client_api.result_check_interval, 5.0)

    def test_init_subscriptions(self):
        client_api = InProcessClientAPI(self.task_metadata)
        xs = list(client_api.data_bus.subscribers.keys())

        # Depending on the timing of this test, the data bus may have other subscribed topics
        # since the data bus is a singleton!
        assert set(xs).issuperset([TOPIC_ABORT, TOPIC_GLOBAL_RESULT, TOPIC_STOP])

    def local_result_callback(self, data, topic):
        pass

    def log_result_callback(self, data, topic):
        pass

    def test_init_subscriptions2(self):
        data_bus = DataBus()
        data_bus.subscribers.clear()

        data_bus.subscribe([TOPIC_LOCAL_RESULT], self.local_result_callback)
        data_bus.subscribe([TOPIC_LOG_DATA], self.log_result_callback)
        assert list(data_bus.subscribers.keys()) == [TOPIC_LOCAL_RESULT, TOPIC_LOG_DATA]
        client_api = InProcessClientAPI(self.task_metadata)
        assert list(client_api.data_bus.subscribers.keys()) == [
            TOPIC_LOCAL_RESULT,
            TOPIC_LOG_DATA,
            TOPIC_GLOBAL_RESULT,
            TOPIC_ABORT,
            TOPIC_STOP,
        ]

    # Add more test methods for other functionalities in the class
