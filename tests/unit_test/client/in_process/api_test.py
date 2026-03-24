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
from nvflare.app_common.abstract.fl_model import FLModel
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

    def test_memory_management_defaults(self):
        """Test that memory management is disabled by default."""
        client_api = InProcessClientAPI(self.task_metadata)
        assert client_api._memory_gc_rounds == 0
        assert client_api._cuda_empty_cache is False
        assert client_api._round_count == 0

    def test_configure_memory_management(self):
        """Test configure_memory_management method."""
        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()

        client_api.configure_memory_management(gc_rounds=5, cuda_empty_cache=True)

        assert client_api._memory_gc_rounds == 5
        assert client_api._cuda_empty_cache is True

    def test_maybe_cleanup_memory_disabled(self):
        """Test that _maybe_cleanup_memory does nothing when disabled."""
        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()

        # With gc_rounds=0 (disabled), should not increment round count
        client_api._maybe_cleanup_memory()
        assert client_api._round_count == 0

    def test_maybe_cleanup_memory_enabled(self):
        """Test that _maybe_cleanup_memory increments round count when enabled."""
        from unittest.mock import patch

        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()
        client_api.configure_memory_management(gc_rounds=2, cuda_empty_cache=False)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            # First round - should not trigger cleanup
            client_api._maybe_cleanup_memory()
            assert client_api._round_count == 1
            mock_cleanup.assert_not_called()

            # Second round - should trigger cleanup (every 2 rounds)
            client_api._maybe_cleanup_memory()
            assert client_api._round_count == 2
            mock_cleanup.assert_called_once_with(cuda_empty_cache=False)

            # Third round - should not trigger cleanup
            mock_cleanup.reset_mock()
            client_api._maybe_cleanup_memory()
            assert client_api._round_count == 3
            mock_cleanup.assert_not_called()

            # Fourth round - should trigger cleanup
            client_api._maybe_cleanup_memory()
            assert client_api._round_count == 4
            mock_cleanup.assert_called_once()

    def test_maybe_cleanup_memory_every_round(self):
        """Test cleanup every round (gc_rounds=1)."""
        from unittest.mock import patch

        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()
        client_api.configure_memory_management(gc_rounds=1, cuda_empty_cache=True)

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            client_api._maybe_cleanup_memory()
            mock_cleanup.assert_called_with(cuda_empty_cache=True)

            mock_cleanup.reset_mock()
            client_api._maybe_cleanup_memory()
            mock_cleanup.assert_called_with(cuda_empty_cache=True)

    # ------------------------------------------------------------------ #
    #  release_params behaviour inside send()                             #
    # ------------------------------------------------------------------ #

    def _fire_global_model(self, client_api, params):
        """Simulate the executor publishing a global model to the data bus."""
        from nvflare.apis.dxo import DXO, DataKind

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=params)
        shareable = dxo.to_shareable()
        client_api.data_bus.publish([TOPIC_GLOBAL_RESULT], shareable)

    def test_send_nulls_sent_model_params_when_clear_cache(self):
        """model.params is None after send() with clear_cache=True (default)."""
        import numpy as np

        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()

        params = {"w": np.ones((50, 50))}
        self._fire_global_model(client_api, params)
        input_model = client_api.receive()
        self.assertIsNotNone(input_model.params)

        output_model = FLModel(params={"w": np.zeros((50, 50))})
        client_api.send(output_model, clear_cache=True)

        self.assertIsNone(output_model.params)
        self.assertIsNone(output_model.optimizer_params)

    def test_send_nulls_received_model_params_when_clear_cache(self):
        """input_model.params is None after send() with clear_cache=True."""
        import numpy as np

        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()

        params = {"w": np.ones((50, 50))}
        self._fire_global_model(client_api, params)
        input_model = client_api.receive()
        self.assertIsNotNone(input_model.params)

        output_model = FLModel(params={"w": np.zeros((50, 50))})
        client_api.send(output_model, clear_cache=True)

        # fl_model is set to None by clear_cache, but the object's params
        # must have been nulled before the reference was dropped
        self.assertIsNone(input_model.params)

    def test_send_preserves_params_when_clear_cache_false(self):
        """model.params is NOT None after send() with clear_cache=False."""
        import numpy as np

        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()

        params = {"w": np.ones((50, 50))}
        self._fire_global_model(client_api, params)
        client_api.receive()

        output_model = FLModel(params={"w": np.zeros((50, 50))})
        client_api.send(output_model, clear_cache=False)

        self.assertIsNotNone(output_model.params)

    def test_send_preserves_received_params_when_clear_cache_false(self):
        """input_model.params is NOT None after send() with clear_cache=False."""
        import numpy as np

        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()

        params = {"w": np.ones((50, 50))}
        self._fire_global_model(client_api, params)
        input_model = client_api.receive()

        output_model = FLModel(params={"w": np.zeros((50, 50))})
        client_api.send(output_model, clear_cache=False)

        self.assertIsNotNone(input_model.params)

    def test_send_metrics_preserved_after_release(self):
        """metrics are not affected by release_params logic."""
        import numpy as np

        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()

        self._fire_global_model(client_api, {"w": np.ones((10,))})
        client_api.receive()

        output_model = FLModel(params={"w": np.zeros((10,))}, metrics={"loss": 0.42})
        client_api.send(output_model, clear_cache=True)

        self.assertIsNone(output_model.params)
        self.assertEqual(output_model.metrics, {"loss": 0.42})

    # Add more test methods for other functionalities in the class
