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

"""Unit tests for hello-numpy-cross-val client: train/evaluate helpers and fail-fast behavior.

The client must:
- Raise RuntimeError when a train task has no params (no NUMPY_KEY), so the server
  does not receive an empty response and break aggregation.
- Raise RuntimeError when submit_model is requested but last_params is missing,
  so CSE does not silently submit wrong weights.

Note: Other tests in this folder (fedavg_recipe_test, fedopt_recipe_test, eval_recipe_test)
test recipe classes (job construction, config). This file tests an *example client script*
that uses the public nvflare.client API (imported as ``flare``). To run the client's main()
without a real FL runtime we replace ``flare`` with a mock object (mock_flare) that
provides receive(), is_train(), is_evaluate(), is_submit_model(), send(), etc., so we can
drive one loop iteration and assert fail-fast behavior. The client/in_process/api_test.py
tests the real InProcessClientAPI implementation; here we mock the API to test the
example's response to bad inputs.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nvflare.app_common.abstract.fl_model import FLModel


def _client_dir():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "examples",
        "hello-world",
        "hello-numpy-cross-val",
    )


def _import_client():
    client_dir = os.path.abspath(_client_dir())
    if client_dir not in sys.path:
        sys.path.insert(0, client_dir)
    import client as client_mod  # noqa: E402

    return client_mod


class TestHelloNumpyCrossValClientHelpers:
    """Test pure helper functions used by the client."""

    def test_train_adds_one(self):
        client_mod = _import_client()
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        out = client_mod.train(x)
        np.testing.assert_array_almost_equal(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_evaluate_returns_weight_mean(self):
        client_mod = _import_client()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        metrics = client_mod.evaluate(x)
        assert "weight_mean" in metrics
        assert metrics["weight_mean"] == 2.0


def _make_mock_flare(*, receive_return, is_train, is_evaluate, is_submit_model):
    """Build a mock for nvflare.client (flare) so the example client's main() can run one loop.

    Used only in TestHelloNumpyCrossValClientFailFast. Other recipe tests in this folder
    test recipe classes and use patch("os.path.*"); they do not run or mock client scripts.

    is_running.side_effect = [True, False] so the while loop runs at most one iteration then
    exits; avoids hanging if client logic stops raising.
    """
    mock = MagicMock()
    mock.init.return_value = None
    mock.system_info.return_value = {"site_name": "site-1"}
    mock.is_running.side_effect = [True, False]
    mock.receive.return_value = receive_return
    mock.is_train.return_value = is_train
    mock.is_evaluate.return_value = is_evaluate
    mock.is_submit_model.return_value = is_submit_model
    mock.send.return_value = None
    mock.ParamsType = MagicMock()
    mock.ParamsType.FULL = "FULL"
    mock.ParamsType.DIFF = "DIFF"
    return mock


class TestHelloNumpyCrossValClientFailFast:
    """Test that the client raises RuntimeError instead of sending invalid responses."""

    def test_train_task_with_no_params_raises(self):
        """Train task with params None or missing NUMPY_KEY must raise; empty response breaks aggregation."""
        client_mod = _import_client()
        mock_flare = _make_mock_flare(
            receive_return=FLModel(params=None, current_round=0),
            is_train=True,
            is_evaluate=False,
            is_submit_model=False,
        )
        with patch("sys.argv", ["client.py"]), patch.object(client_mod, "flare", mock_flare):
            with pytest.raises(RuntimeError, match="Train task received no model params"):
                client_mod.main()

    def test_train_task_with_params_missing_numpy_key_raises(self):
        """Train task with params dict but no NUMPY_KEY must raise."""
        client_mod = _import_client()
        mock_flare = _make_mock_flare(
            receive_return=FLModel(params={}, current_round=0),
            is_train=True,
            is_evaluate=False,
            is_submit_model=False,
        )
        with patch("sys.argv", ["client.py"]), patch.object(client_mod, "flare", mock_flare):
            with pytest.raises(RuntimeError, match="Train task received no model params"):
                client_mod.main()

    def test_submit_model_with_no_last_params_raises(self):
        """submit_model when last_params was never set must raise; otherwise wrong weights are submitted."""
        client_mod = _import_client()
        mock_flare = _make_mock_flare(
            receive_return=FLModel(params=None, current_round=None),
            is_train=False,
            is_evaluate=False,
            is_submit_model=True,
        )
        with patch("sys.argv", ["client.py"]), patch.object(client_mod, "flare", mock_flare):
            with pytest.raises(RuntimeError, match="submit_model called but no local model"):
                client_mod.main()
