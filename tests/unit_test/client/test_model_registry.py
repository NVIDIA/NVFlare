# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for ModelRegistry.release_params()."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.config import ClientConfig, ConfigKey, ExchangeFormat, TransferType
from nvflare.client.flare_agent import Task
from nvflare.client.model_registry import ModelRegistry


def _make_registry(transfer_type: str = TransferType.FULL) -> ModelRegistry:
    config = ClientConfig(
        config={
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.NUMPY,
                ConfigKey.TRANSFER_TYPE: transfer_type,
                ConfigKey.TRAIN_TASK_NAME: "train",
                ConfigKey.EVAL_TASK_NAME: "evaluate",
                ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
            }
        }
    )
    mock_agent = MagicMock()
    return ModelRegistry(config, rank="0", flare_agent=mock_agent)


def _make_model(shape=(100, 100), with_optimizer=False, with_metrics=False) -> FLModel:
    params = {"layer": np.ones(shape)}
    opt = {"lr": np.array([0.01])} if with_optimizer else None
    metrics = {"loss": 0.5} if with_metrics else None
    return FLModel(params=params, optimizer_params=opt, metrics=metrics)


def _inject_received_task(registry: ModelRegistry, shape=(100, 100)) -> FLModel:
    received_model = _make_model(shape=shape)
    task = Task(task_name="train", task_id="test-task-1", data=received_model)
    registry._set_task(task)
    return received_model


class TestReleaseParamsSentModel(unittest.TestCase):
    """release_params() nulls params on the sent model."""

    def test_params_nulled(self):
        registry = _make_registry()
        model = _make_model()
        registry.release_params(model)
        self.assertIsNone(model.params)

    def test_optimizer_params_nulled(self):
        registry = _make_registry()
        model = _make_model(with_optimizer=True)
        registry.release_params(model)
        self.assertIsNone(model.optimizer_params)

    def test_metrics_preserved(self):
        """metrics are not touched — only params and optimizer_params."""
        registry = _make_registry()
        model = _make_model(with_metrics=True)
        registry.release_params(model)
        self.assertIsNone(model.params)
        self.assertEqual(model.metrics, {"loss": 0.5})

    def test_already_none_params_is_noop(self):
        registry = _make_registry()
        model = FLModel(metrics={"acc": 0.9})
        registry.release_params(model)  # should not raise
        self.assertIsNone(model.params)
        self.assertIsNone(model.optimizer_params)


class TestReleaseParamsReceivedModel(unittest.TestCase):
    """release_params() also nulls the received (input) model's params."""

    def test_received_params_nulled(self):
        registry = _make_registry()
        received = _inject_received_task(registry)
        sent = _make_model()

        registry.release_params(sent)

        self.assertIsNone(sent.params)
        self.assertIsNone(received.params)

    def test_received_optimizer_params_nulled(self):
        registry = _make_registry()
        received_model = _make_model(with_optimizer=True)
        task = Task(task_name="train", task_id="test-task-1", data=received_model)
        registry._set_task(task)

        registry.release_params(_make_model())

        self.assertIsNone(received_model.optimizer_params)

    def test_no_received_task_does_not_raise(self):
        """Safe to call when no model has been received yet."""
        registry = _make_registry()
        self.assertIsNone(registry.received_task)
        model = _make_model()
        registry.release_params(model)  # must not raise
        self.assertIsNone(model.params)

    def test_received_task_data_none_does_not_raise(self):
        """Safe when received_task exists but data is None."""
        registry = _make_registry()
        task = Task(task_name="train", task_id="test-task-1", data=None)
        registry._set_task(task)
        model = _make_model()
        registry.release_params(model)  # must not raise
        self.assertIsNone(model.params)

    def test_same_object_passthrough(self):
        """User passes the received model unchanged to send — both refs become None."""
        registry = _make_registry()
        received = _inject_received_task(registry)
        # Same object used as sent model (passthrough evaluation pattern)
        registry.release_params(received)
        self.assertIsNone(received.params)


class TestReleaseParamsMemoryRelease(unittest.TestCase):
    """Verify that numpy arrays are actually dereferenced after release_params."""

    def test_large_arrays_dereferenced(self):
        import gc
        import sys

        registry = _make_registry()
        received = _inject_received_task(registry, shape=(1000, 1000))
        sent = _make_model(shape=(1000, 1000))

        # Hold weak-like check via sys.getrefcount
        # refcount before: FLModel.params + local var 'arr' + getrefcount arg = 3
        arr_ref = sent.params
        initial_refcount = sys.getrefcount(arr_ref)

        registry.release_params(sent)
        gc.collect()

        # After release: sent.params is None, one reference dropped
        self.assertIsNone(sent.params)
        self.assertIsNone(received.params)
        # refcount of the dict should have decreased by 1 (model no longer holds it)
        self.assertEqual(sys.getrefcount(arr_ref), initial_refcount - 1)


class TestReleaseParamsDiffMode(unittest.TestCase):
    """In DIFF mode, both the diff (sent) and original (received) params are released."""

    def test_diff_mode_releases_both(self):
        registry = _make_registry(transfer_type=TransferType.DIFF)
        received = _inject_received_task(registry)

        # After _prepare_param_diff, model.params holds the diff dict
        diff_params = {"layer": np.zeros((100, 100))}
        sent = FLModel(params=diff_params)

        registry.release_params(sent)

        self.assertIsNone(sent.params)
        self.assertIsNone(received.params)


if __name__ == "__main__":
    unittest.main()
