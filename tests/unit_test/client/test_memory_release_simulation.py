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

"""
Memory release simulation tests — hello-world style.

Simulates multiple FL rounds (receive → train → send) using InProcessClientAPI
with large numpy arrays and verifies:
  1. model.params and input_model.params are None after flare.send()
  2. Python's reference count on the large arrays drops to the expected level
  3. (Optional) RSS does not grow across rounds when psutil is available

Run directly for a human-readable memory report:
    python -m pytest tests/unit_test/client/test_memory_release_simulation.py -v -s
"""

import gc
import sys
import unittest

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType
from nvflare.client.in_process.api import TOPIC_GLOBAL_RESULT, InProcessClientAPI
from nvflare.fuel.data_event.data_bus import DataBus

# ── helpers ────────────────────────────────────────────────────────────────


def _task_metadata():
    from nvflare.apis.fl_constant import FLMetaKey

    return {
        FLMetaKey.JOB_ID: "sim-job-1",
        FLMetaKey.SITE_NAME: "site-1",
        "TASK_NAME": "train",
        ConfigKey.TASK_EXCHANGE: {
            ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.NUMPY,
            ConfigKey.TRANSFER_TYPE: TransferType.FULL,
            ConfigKey.TRAIN_TASK_NAME: "train",
            ConfigKey.EVAL_TASK_NAME: "evaluate",
            ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
        },
    }


def _publish_global_model(data_bus: DataBus, params: dict):
    """Simulate executor publishing a global model to the client."""
    dxo = DXO(data_kind=DataKind.WEIGHTS, data=params)
    shareable = dxo.to_shareable()
    data_bus.publish([TOPIC_GLOBAL_RESULT], shareable)


def _rss_mb() -> float:
    """Return current process RSS in MB, or -1 if psutil not available."""
    try:
        import os

        import psutil  # noqa: E401

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


# ── tests ──────────────────────────────────────────────────────────────────


class TestInProcessMemoryRelease(unittest.TestCase):
    """Verify params are released on every round for InProcessClientAPI."""

    ARRAY_SHAPE = (512, 512)  # ~2 MB per array
    NUM_ROUNDS = 5

    def setUp(self):
        self.api = InProcessClientAPI(_task_metadata())
        self.api.init()

    def _one_round(self, round_num: int):
        """Simulate one FL round: receive → train → send."""
        global_params = {"weights": np.full(self.ARRAY_SHAPE, float(round_num))}
        _publish_global_model(self.api.data_bus, global_params)

        input_model = self.api.receive()
        self.assertIsNotNone(input_model, f"round {round_num}: receive returned None")
        self.assertIsNotNone(input_model.params, f"round {round_num}: input_model.params is already None")

        # Simulate training: add 1 to each weight
        trained_params = {k: v + 1.0 for k, v in input_model.params.items()}
        output_model = FLModel(params=trained_params)

        self.api.send(output_model, clear_cache=True)

        return input_model, output_model

    def test_params_nulled_after_each_send(self):
        """After send(), both input_model.params and output_model.params must be None."""
        for r in range(self.NUM_ROUNDS):
            input_model, output_model = self._one_round(r)
            self.assertIsNone(
                output_model.params,
                f"round {r}: output_model.params not released",
            )
            self.assertIsNone(
                input_model.params,
                f"round {r}: input_model.params not released",
            )

    def test_array_refcount_drops_after_send(self):
        """After send(), output_model.params and input_model.params are None;
        the output array's refcount drops (it has no external holders)."""
        global_params = {"weights": np.ones(self.ARRAY_SHAPE)}
        _publish_global_model(self.api.data_bus, global_params)

        input_model = self.api.receive()
        output_model = FLModel(params={"weights": input_model.params["weights"] + 1.0})

        # Keep a local ref to output array to measure its refcount independently.
        # The output array is brand-new (result of +1.0); only output_model.params
        # and this local variable hold it — no external caches.
        output_arr = output_model.params["weights"]
        output_refcount_before = sys.getrefcount(output_arr)

        self.api.send(output_model, clear_cache=True)
        gc.collect()

        # params refs cleared by send()
        self.assertIsNone(input_model.params, "input_model.params should be None after send")
        self.assertIsNone(output_model.params, "output_model.params should be None after send")

        # output array: FLModel.params ref dropped → refcount decreased by at least 1
        self.assertLess(
            sys.getrefcount(output_arr),
            output_refcount_before,
            "output array refcount should decrease after send",
        )

    def test_params_preserved_when_clear_cache_false(self):
        """With clear_cache=False nothing is nulled."""
        global_params = {"weights": np.ones(self.ARRAY_SHAPE)}
        _publish_global_model(self.api.data_bus, global_params)

        input_model = self.api.receive()
        output_model = FLModel(params={"weights": np.zeros(self.ARRAY_SHAPE)})

        self.api.send(output_model, clear_cache=False)

        self.assertIsNotNone(output_model.params)
        self.assertIsNotNone(input_model.params)

    def test_rss_does_not_grow_across_rounds(self):
        """
        RSS should not grow monotonically across rounds.

        Skipped silently when psutil is not installed.
        With params released on each send(), the GC can reclaim large arrays
        between rounds, so peak RSS stays roughly constant.
        """
        if _rss_mb() < 0:
            self.skipTest("psutil not available — skipping RSS check")

        rss_readings = []
        for r in range(self.NUM_ROUNDS):
            _publish_global_model(
                self.api.data_bus,
                {"weights": np.ones(self.ARRAY_SHAPE) * r},
            )
            input_model = self.api.receive()
            output_model = FLModel(params={"weights": input_model.params["weights"] + 1})
            self.api.send(output_model, clear_cache=True)
            gc.collect()
            rss_readings.append(_rss_mb())
            print(f"  Round {r}: RSS = {rss_readings[-1]:.1f} MB")

        # RSS at the last round must not exceed RSS at round 1 by more than 50 MB
        # (a generous threshold to account for Python runtime overhead)
        rss_growth = rss_readings[-1] - rss_readings[1]
        self.assertLess(
            rss_growth,
            50.0,
            f"RSS grew by {rss_growth:.1f} MB over {self.NUM_ROUNDS} rounds — params not being freed",
        )


class TestExProcessMemoryRelease(unittest.TestCase):
    """Verify release_params() integration for ExProcessClientAPI via ModelRegistry."""

    def test_release_params_end_to_end(self):
        """ModelRegistry.release_params() correctly nulls both models."""
        from unittest.mock import MagicMock

        from nvflare.client.config import ClientConfig
        from nvflare.client.flare_agent import Task
        from nvflare.client.model_registry import ModelRegistry

        config = ClientConfig(
            config={
                ConfigKey.TASK_EXCHANGE: {
                    ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.NUMPY,
                    ConfigKey.TRANSFER_TYPE: TransferType.FULL,
                    ConfigKey.TRAIN_TASK_NAME: "train",
                    ConfigKey.EVAL_TASK_NAME: "evaluate",
                    ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
                }
            }
        )
        registry = ModelRegistry(config, rank="0", flare_agent=MagicMock())

        # Simulate receive
        received_params = {"w": np.ones((256, 256))}
        received_model = FLModel(params=received_params)
        registry._set_task(Task(task_name="train", task_id="test-task-1", data=received_model))

        # Simulate send
        sent_params = {"w": np.zeros((256, 256))}
        sent_model = FLModel(params=sent_params)
        registry.release_params(sent_model)

        self.assertIsNone(sent_model.params, "sent model params must be None")
        self.assertIsNone(received_model.params, "received model params must be None")

    def test_multiple_rounds_no_accumulation(self):
        """After each round, the old arrays must be dereferenced."""
        import gc
        from unittest.mock import MagicMock

        from nvflare.client.config import ClientConfig
        from nvflare.client.flare_agent import Task
        from nvflare.client.model_registry import ModelRegistry

        config = ClientConfig(
            config={
                ConfigKey.TASK_EXCHANGE: {
                    ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.NUMPY,
                    ConfigKey.TRANSFER_TYPE: TransferType.FULL,
                    ConfigKey.TRAIN_TASK_NAME: "train",
                    ConfigKey.EVAL_TASK_NAME: "evaluate",
                    ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
                }
            }
        )
        registry = ModelRegistry(config, rank="0", flare_agent=MagicMock())

        prev_received = None
        prev_sent = None

        for r in range(5):
            received_model = FLModel(params={"w": np.ones((256, 256)) * r})
            registry._set_task(Task(task_name="train", task_id="test-task-1", data=received_model))
            sent_model = FLModel(params={"w": np.zeros((256, 256)) + r})

            registry.release_params(sent_model)
            gc.collect()

            if prev_received is not None:
                self.assertIsNone(prev_received.params, f"round {r}: stale received params still alive")
            if prev_sent is not None:
                self.assertIsNone(prev_sent.params, f"round {r}: stale sent params still alive")

            prev_received = received_model
            prev_sent = sent_model


if __name__ == "__main__":
    # Run with -s to see RSS output
    unittest.main(verbosity=2)
