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
from copy import deepcopy

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.params_converter import ParamsConverter
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


class _AddOneConverter(ParamsConverter):
    def convert(self, params, fl_ctx):
        fl_ctx.set_prop("from_converter_called", True)
        return {k: v + 1 for k, v in params.items()}


class _AddTwoConverter(ParamsConverter):
    def convert(self, params, fl_ctx):
        assert fl_ctx.get_prop("from_converter_called")
        return {k: v + 2 for k, v in params.items()}


class _FailingConverter(ParamsConverter):
    def convert(self, params, fl_ctx):
        raise ValueError("bad conversion")


class TestInProcessClientAPI(unittest.TestCase):
    def setUp(self):
        # Create a mock task_metadata for testing
        self.task_metadata = {
            FLMetaKey.JOB_ID: "123",
            FLMetaKey.SITE_NAME: "site-1",
            "TASK_NAME": "train",
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.TRAIN_WITH_EVAL: "train_with_eval",
                ConfigKey.EXCHANGE_FORMAT: "numpy",
                ConfigKey.SERVER_EXPECTED_FORMAT: "numpy",
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

    def test_close_unsubscribes_api_callbacks(self):
        data_bus = DataBus()
        data_bus.subscribers.clear()
        client_api = InProcessClientAPI(self.task_metadata)

        client_api.close()
        client_api.close()  # idempotent

        assert TOPIC_GLOBAL_RESULT not in data_bus.subscribers
        assert TOPIC_ABORT not in data_bus.subscribers
        assert TOPIC_STOP not in data_bus.subscribers

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

    def test_diff_config_allows_metrics_only_result(self):
        """DIFF applies only to parameter results; validation metrics remain metrics."""
        import numpy as np

        from nvflare.apis.dxo import DataKind, from_shareable

        client_api = InProcessClientAPI(self.task_metadata)
        client_api.init()
        sent = []

        def capture(_topic, data, _databus):
            sent.append(data)

        client_api.data_bus.subscribe([TOPIC_LOCAL_RESULT], capture)
        try:
            self._fire_global_model(client_api, {"w": np.ones((10,))})
            self.assertIsNotNone(client_api.receive())

            client_api.send(FLModel(metrics={"accuracy": 0.75}), clear_cache=False)

            self.assertEqual(len(sent), 1)
            result = from_shareable(sent[0])
            self.assertEqual(result.data_kind, DataKind.METRICS)
            self.assertEqual(result.data, {"accuracy": 0.75})
        finally:
            client_api.data_bus.unsubscribe(TOPIC_LOCAL_RESULT, capture)

    def test_receive_timeout_does_not_arm_send_under_congestion(self):
        """A missing next-round model should time out cleanly and not permit send()."""
        client_api = InProcessClientAPI(self.task_metadata, result_check_interval=0.001)
        client_api.init()

        self.assertIsNone(client_api.receive(timeout=0.01))
        self.assertFalse(client_api.receive_called)
        with self.assertRaisesRegex(RuntimeError, '"receive" needs to be called'):
            client_api.send(FLModel(params={"w": 1}))

    def test_receive_timeout_then_later_model_allows_send(self):
        """A timeout must not poison the next successful receive/send cycle."""
        import numpy as np

        client_api = InProcessClientAPI(self.task_metadata, result_check_interval=0.001)
        client_api.init()

        self.assertIsNone(client_api.receive(timeout=0.01))
        self.assertFalse(client_api.receive_called)

        self._fire_global_model(client_api, {"w": np.ones((10,))})
        input_model = client_api.receive(timeout=0.01)
        self.assertIsNotNone(input_model)
        self.assertIsNotNone(input_model.params)
        self.assertTrue(client_api.receive_called)

        client_api.send(FLModel(params={"w": np.zeros((10,))}), clear_cache=False)
        self.assertTrue(client_api.receive_called)

    def test_declared_pytorch_conversion_runs_at_receive_send_boundary(self):
        import numpy as np

        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        meta = deepcopy(self.task_metadata)
        exchange = meta[ConfigKey.TASK_EXCHANGE]
        exchange[ConfigKey.EXCHANGE_FORMAT] = "pytorch"
        exchange[ConfigKey.SERVER_EXPECTED_FORMAT] = "numpy"
        exchange[ConfigKey.TRANSFER_TYPE] = "FULL"
        client_api = InProcessClientAPI(meta)
        client_api.init()
        sent = []

        def capture(_topic, data, _databus):
            sent.append(data)

        client_api.data_bus.subscribe([TOPIC_LOCAL_RESULT], capture)
        try:
            self._fire_global_model(client_api, {"w": np.asarray([1.0, 2.0])})
            received = client_api.receive()
            self.assertIsInstance(received.params["w"], torch.Tensor)

            client_api.send(FLModel(params={"w": received.params["w"] + 1}), clear_cache=False)

            from nvflare.apis.dxo import from_shareable

            wire_params = from_shareable(sent[-1]).data
            self.assertIsInstance(wire_params["w"], np.ndarray)
            np.testing.assert_array_equal(wire_params["w"], np.asarray([2.0, 3.0]))
        finally:
            client_api.data_bus.unsubscribe(TOPIC_LOCAL_RESULT, capture)

    def test_custom_converters_run_at_receive_send_boundary(self):
        import numpy as np

        meta = deepcopy(self.task_metadata)
        meta[ConfigKey.TASK_EXCHANGE][ConfigKey.TRANSFER_TYPE] = "FULL"
        fl_ctx = FLContext()
        client_api = InProcessClientAPI(
            meta,
            from_nvflare_converter=_AddOneConverter(["train"]),
            to_nvflare_converter=_AddTwoConverter(["train"]),
        )
        client_api.init()
        client_api.set_meta(meta, fl_ctx)
        sent = []

        def capture(_topic, data, _databus):
            sent.append(data)

        client_api.data_bus.subscribe([TOPIC_LOCAL_RESULT], capture)
        try:
            self._fire_global_model(client_api, {"w": np.asarray([1.0])})
            received = client_api.receive()
            np.testing.assert_array_equal(received.params["w"], np.asarray([2.0]))

            client_api.send(FLModel(params={"w": np.asarray([10.0])}), clear_cache=False)

            from nvflare.apis.dxo import from_shareable

            wire_params = from_shareable(sent[-1]).data
            np.testing.assert_array_equal(wire_params["w"], np.asarray([12.0]))
        finally:
            client_api.data_bus.unsubscribe(TOPIC_LOCAL_RESULT, capture)
            client_api.close()

    def test_receive_surfaces_converter_failure(self):
        client_api = InProcessClientAPI(
            self.task_metadata,
            result_check_interval=0.001,
            from_nvflare_converter=_FailingConverter(),
        )
        client_api.init()
        client_api.set_meta(self.task_metadata, FLContext())
        try:
            self._fire_global_model(client_api, {"w": 1.0})
            with self.assertRaisesRegex(RuntimeError, "failed to receive task: bad conversion"):
                client_api.receive(timeout=0.01)
        finally:
            client_api.close()

    def test_clear_resets_receive_guard_between_rounds(self):
        """A prior successful round must not arm send() after a later receive timeout."""
        import numpy as np

        client_api = InProcessClientAPI(self.task_metadata, result_check_interval=0.001)
        client_api.init()

        self._fire_global_model(client_api, {"w": np.ones((10,))})
        self.assertIsNotNone(client_api.receive(timeout=0.01))
        self.assertTrue(client_api.receive_called)

        client_api.send(FLModel(params={"w": np.zeros((10,))}), clear_cache=True)
        self.assertFalse(client_api.receive_called)

        self.assertIsNone(client_api.receive(timeout=0.01))
        self.assertFalse(client_api.receive_called)
        with self.assertRaisesRegex(RuntimeError, '"receive" needs to be called'):
            client_api.send(FLModel(params={"w": np.zeros((10,))}))

    # Add more test methods for other functionalities in the class
