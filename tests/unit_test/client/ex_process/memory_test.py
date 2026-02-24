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

"""Tests for ExProcessClientAPI memory management (send/release_params behaviour)."""

import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.api_spec import APISpec
from nvflare.client.ex_process.api import ExProcessClientAPI


def _make_api_with_mock_registry() -> tuple:
    """Create ExProcessClientAPI with a fully mocked ModelRegistry, bypassing init()."""
    api = ExProcessClientAPI.__new__(ExProcessClientAPI)
    APISpec.__init__(api)
    api.config_file = "dummy.json"
    api.flare_agent = None
    api.receive_called = True  # pretend receive() was already called

    mock_registry = MagicMock()
    api.model_registry = mock_registry
    return api, mock_registry


def _make_model() -> FLModel:
    return FLModel(params={"w": np.ones((50, 50))}, optimizer_params={"lr": np.array([0.01])})


class TestExProcessSendReleasesParams(unittest.TestCase):
    """ExProcessClientAPI.send() calls release_params when clear_cache=True."""

    def test_release_params_called_with_clear_cache_true(self):
        api, mock_registry = _make_api_with_mock_registry()
        model = _make_model()

        api.send(model, clear_cache=True)

        mock_registry.submit_model.assert_called_once_with(model=model)
        mock_registry.release_params.assert_called_once_with(model)

    def test_release_params_before_clear(self):
        """release_params must be called before clear() so received_task is still available."""
        api, mock_registry = _make_api_with_mock_registry()
        model = _make_model()
        call_order = []

        mock_registry.release_params.side_effect = lambda m: call_order.append("release")
        mock_registry.clear.side_effect = lambda: call_order.append("clear")

        api.send(model, clear_cache=True)

        self.assertEqual(call_order, ["release", "clear"])

    def test_release_params_not_called_when_clear_cache_false(self):
        api, mock_registry = _make_api_with_mock_registry()
        model = _make_model()

        api.send(model, clear_cache=False)

        mock_registry.submit_model.assert_called_once_with(model=model)
        mock_registry.release_params.assert_not_called()
        mock_registry.clear.assert_not_called()

    def test_receive_called_flag_reset_after_send(self):
        api, mock_registry = _make_api_with_mock_registry()
        api.send(_make_model(), clear_cache=True)
        self.assertFalse(api.receive_called)

    def test_receive_called_flag_unchanged_when_no_clear(self):
        api, mock_registry = _make_api_with_mock_registry()
        api.receive_called = True
        api.send(_make_model(), clear_cache=False)
        self.assertTrue(api.receive_called)

    def test_send_without_receive_raises(self):
        api, _ = _make_api_with_mock_registry()
        api.receive_called = False
        with self.assertRaises(RuntimeError):
            api.send(_make_model())


class TestExProcessMemoryCleanupIntegration(unittest.TestCase):
    """gc cleanup setting in ExProcessClientAPI (via api_spec._maybe_cleanup_memory)."""

    def test_memory_cleanup_called_every_n_rounds(self):
        api, mock_registry = _make_api_with_mock_registry()
        api._memory_gc_rounds = 2

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            # Round 1: no cleanup
            api.receive_called = True
            api.send(_make_model())
            mock_cleanup.assert_not_called()

            # Round 2: cleanup fires
            api.receive_called = True
            api.send(_make_model())
            mock_cleanup.assert_called_once()

    def test_memory_cleanup_not_called_when_disabled(self):
        api, mock_registry = _make_api_with_mock_registry()
        api._memory_gc_rounds = 0

        with patch("nvflare.fuel.utils.memory_utils.cleanup_memory") as mock_cleanup:
            for _ in range(5):
                api.receive_called = True
                api.send(_make_model())
            mock_cleanup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
