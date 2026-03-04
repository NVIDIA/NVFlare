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

"""Unit tests for Bug 2 fix: CSE _prepare_local_model() persistor fallback.

Bug 2 — CSE + external process submit_model incompatibility:
    Root Cause: _prepare_local_model() calls submit_model_executor.execute()
    which launches a fresh subprocess with no trained model state.  With
    external process executors (ClientAPILauncherExecutor), the subprocess
    cannot find or return the trained model.

    Fix: Add a persistor-based retrieval path BEFORE the executor path.
    The best model is already saved to disk during training via
    persistor.save().  This is consistent with _prepare_global_model().

CONTRACT:
- When persistor is available and returns a model → use it, return OK
- When persistor.get(model_name) returns None → try inventory keys
- When persistor raises an exception → fall back to submit_model_executor
- When persistor is None → fall back to submit_model_executor
- Persistor path logs info on success, warning on exception
"""

from unittest.mock import MagicMock, Mock, patch

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.ccwf.cse_client_ctl import CrossSiteEvalClientController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_controller(persistor=None, submit_model_executor=None):
    """Create a minimal CrossSiteEvalClientController without calling __init__."""
    ctrl = CrossSiteEvalClientController.__new__(CrossSiteEvalClientController)
    ctrl.persistor = persistor
    ctrl.submit_model_executor = submit_model_executor
    ctrl.submit_model_task_name = "submit_model"
    ctrl.log_info = Mock()
    ctrl.log_warning = Mock()
    ctrl.log_error = Mock()
    ctrl.log_exception = Mock()
    ctrl._set_prepared_model = Mock()
    return ctrl


def _make_persistor(model_learnable=None, inventory=None, get_side_effect=None):
    """Create a mock ModelPersistor."""
    persistor = MagicMock(spec=ModelPersistor)

    if get_side_effect:
        persistor.get.side_effect = get_side_effect
    else:
        persistor.get.return_value = model_learnable

    persistor.get_model_inventory.return_value = inventory
    return persistor


def _make_fl_ctx():
    return MagicMock(spec=FLContext)


def _make_abort_signal():
    return MagicMock(spec=Signal)


def _make_model_learnable():
    """Return a mock model learnable that converts to a DXO and Shareable."""
    learnable = MagicMock()
    return learnable


# ---------------------------------------------------------------------------
# Bug 2: _prepare_local_model persistor fallback
# ---------------------------------------------------------------------------


class TestPrepareLocalModelPersistorFallback:
    """_prepare_local_model() must try persistor before submit_model_executor."""

    @patch("nvflare.app_common.ccwf.cse_client_ctl.model_learnable_to_dxo")
    def test_persistor_returns_model_directly(self, mock_to_dxo):
        """When persistor.get(model_name) returns a model, use it and return OK."""
        model_learnable = _make_model_learnable()
        mock_dxo = MagicMock()
        mock_dxo.to_shareable.return_value = Shareable()
        mock_to_dxo.return_value = mock_dxo

        persistor = _make_persistor(model_learnable=model_learnable)
        executor = MagicMock()
        ctrl = _make_controller(persistor=persistor, submit_model_executor=executor)

        reply = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        assert reply.get_return_code() == ReturnCode.OK
        ctrl._set_prepared_model.assert_called_once()
        executor.execute.assert_not_called()
        ctrl.log_info.assert_called()

    @patch("nvflare.app_common.ccwf.cse_client_ctl.model_learnable_to_dxo")
    def test_persistor_falls_back_to_inventory(self, mock_to_dxo):
        """When persistor.get(model_name) returns None, iterate inventory keys."""
        model_learnable = _make_model_learnable()
        mock_dxo = MagicMock()
        mock_dxo.to_shareable.return_value = Shareable()
        mock_to_dxo.return_value = mock_dxo

        # get("best_model") returns None, get("FL_global_model.pt") returns model
        def get_side_effect(name, fl_ctx):
            if name == "FL_global_model.pt":
                return model_learnable
            return None

        persistor = _make_persistor(
            get_side_effect=get_side_effect,
            inventory={"FL_global_model.pt": MagicMock()},
        )
        ctrl = _make_controller(persistor=persistor)

        reply = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        assert reply.get_return_code() == ReturnCode.OK
        ctrl._set_prepared_model.assert_called_once()

    def test_persistor_exception_falls_back_to_executor(self):
        """When persistor raises an exception, fall back to submit_model_executor."""
        persistor = _make_persistor(get_side_effect=RuntimeError("disk error"))

        executor = MagicMock()
        executor_result = Shareable()
        executor_result.set_return_code(ReturnCode.OK)
        executor.execute.return_value = executor_result

        ctrl = _make_controller(persistor=persistor, submit_model_executor=executor)

        reply = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        ctrl.log_warning.assert_called()
        executor.execute.assert_called_once()

    def test_no_persistor_uses_executor(self):
        """When persistor is None, fall back directly to submit_model_executor."""
        executor = MagicMock()
        executor_result = Shareable()
        executor_result.set_return_code(ReturnCode.OK)
        executor.execute.return_value = executor_result

        ctrl = _make_controller(persistor=None, submit_model_executor=executor)

        reply = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        executor.execute.assert_called_once()

    def test_no_persistor_no_executor_returns_bad_request(self):
        """When both persistor and submit_model_executor are absent, return BAD_REQUEST_DATA."""
        ctrl = _make_controller(persistor=None, submit_model_executor=None)

        reply = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        assert reply.get_return_code() == ReturnCode.BAD_REQUEST_DATA
        ctrl.log_error.assert_called()

    @patch("nvflare.app_common.ccwf.cse_client_ctl.model_learnable_to_dxo")
    def test_persistor_empty_inventory_no_model(self, mock_to_dxo):
        """When persistor.get returns None and inventory is empty, fall back to executor."""
        persistor = _make_persistor(model_learnable=None, inventory={})

        executor = MagicMock()
        executor_result = Shareable()
        executor_result.set_return_code(ReturnCode.OK)
        executor.execute.return_value = executor_result

        ctrl = _make_controller(persistor=persistor, submit_model_executor=executor)

        reply = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        executor.execute.assert_called_once()
