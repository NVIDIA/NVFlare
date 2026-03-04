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

"""Unit tests for Bug 2 fix: _prepare_local_model() persistor-first path.

Root cause: _prepare_local_model() called submit_model_executor.execute() which
for ClientAPILauncherExecutor launches a fresh subprocess with no trained model
state.  The best model is already on disk (saved by PTFileModelPersistor during
_process_final_result()).

Fix: Try persistor first.  Fall back to executor only if persistor is absent,
not a ModelPersistor, or raises an exception.

Additional correctness fixes over PR #4263:
- isinstance guard (not assert) — safe under python -O
- Prefer "best" inventory key when model_name contains "best" (avoids returning
  last-round model instead of best model — insertion order: FL_global_model first)
- Logs which inventory key was selected for auditability

Tests verify:
  1. Persistor returns model directly → OK, executor NOT called.
  2. persistor.get(name) returns None, inventory has "best_" key → selects "best_" key.
  3. persistor.get(name) returns None, no "best_" key → selects first loadable key.
  4. Persistor raises exception → warning logged, falls back to executor.
  5. No persistor → executor used directly.
  6. Persistor is not ModelPersistor → warning logged, falls back to executor.
  7. Empty inventory → falls back to executor.
  8. Executor fallback succeeds → OK.
  9. No persistor + no executor → BAD_REQUEST_DATA.
"""

from unittest.mock import MagicMock, patch

from nvflare.apis.fl_constant import ReturnCode
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.ccwf.cse_client_ctl import CrossSiteEvalClientController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_controller(persistor=None, submit_model_executor=None):
    """Build CrossSiteEvalClientController with mocked components, bypassing __init__."""
    ctrl = CrossSiteEvalClientController.__new__(CrossSiteEvalClientController)
    ctrl.persistor = persistor
    ctrl.submit_model_executor = submit_model_executor
    ctrl.submit_model_task_name = "submit_model"
    ctrl.logger = MagicMock()
    ctrl.log_info = MagicMock()
    ctrl.log_warning = MagicMock()
    ctrl.log_error = MagicMock()
    ctrl.log_exception = MagicMock()
    ctrl._set_prepared_model = MagicMock()
    return ctrl


def _make_fl_ctx():
    return MagicMock()


def _make_abort_signal():
    return MagicMock()


def _make_model_persistor(inventory=None, get_result=None):
    """Return a MagicMock spec'd as ModelPersistor with isinstance() support.

    Setting __class__ = ModelPersistor makes isinstance(mock, ModelPersistor)
    return True, which is required for the isinstance guard in _prepare_local_model()
    to pass and reach the persistor code path.
    """
    persistor = MagicMock(spec=ModelPersistor)
    persistor.__class__ = ModelPersistor  # isinstance(persistor, ModelPersistor) → True
    persistor.get.return_value = get_result  # None by default
    persistor.get_model_inventory.return_value = inventory if inventory is not None else {}
    return persistor


def _make_model_learnable():
    from nvflare.app_common.abstract.model import ModelLearnable

    ml = MagicMock(spec=ModelLearnable)
    # ModelLearnable is dict-like; MagicMock(spec=...) auto-mocks __len__ → 0, making
    # bool(ml) == False. The production code uses "if not model_learnable:" to skip
    # None returns, so the mock must be truthy to simulate a loaded model.
    ml.__len__.return_value = 1
    return ml


def _mock_dxo_patch():
    """Context manager: patches model_learnable_to_dxo to return a mock shareable."""
    from nvflare.apis.shareable import Shareable

    mock_dxo = MagicMock()
    mock_dxo.to_shareable.return_value = MagicMock(spec=Shareable)
    return patch("nvflare.app_common.ccwf.cse_client_ctl.model_learnable_to_dxo", return_value=mock_dxo)


def _ok_executor_result():
    from nvflare.apis.shareable import Shareable

    result = Shareable()
    result.set_return_code(ReturnCode.OK)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrepareLocalModelPersistorFirst:

    def test_persistor_returns_model_directly_no_executor_call(self):
        """persistor.get(model_name) succeeds → OK returned, executor NOT called.

        Before the fix, _prepare_local_model() always invoked execute() on a fresh
        subprocess, which had no model state.  Now we try persistor first.
        """
        ml = _make_model_learnable()
        persistor = _make_model_persistor(get_result=ml)
        executor = MagicMock()
        ctrl = _make_controller(persistor=persistor, submit_model_executor=executor)

        with _mock_dxo_patch():
            result = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        assert result.get_return_code() == ReturnCode.OK
        executor.execute.assert_not_called()

    def test_best_key_preferred_when_model_name_contains_best(self):
        """Inventory scan: "best_" key preferred when model_name contains "best".

        PTFileModelPersistor insertion order is FL_global_model → best_FL_global_model.
        Without the preference logic, first-match returns last-round model, not best.
        """
        ml = _make_model_learnable()

        def _get(key, fl_ctx):
            # Simulate PTFileModelPersistor: semantic names like "best_model" return None;
            # only filesystem-derived inventory keys ("best_FL_global_model") return the model.
            if key == "best_FL_global_model":
                return ml
            return None

        persistor = _make_model_persistor(
            inventory={"FL_global_model": MagicMock(), "best_FL_global_model": MagicMock()}
        )
        persistor.get.side_effect = _get
        ctrl = _make_controller(persistor=persistor)

        with _mock_dxo_patch():
            result = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        assert result.get_return_code() == ReturnCode.OK
        # Log must mention the selected "best_" key
        log_msgs = " ".join(str(c) for c in ctrl.log_info.call_args_list)
        assert "best_FL_global_model" in log_msgs, (
            "Expected 'best_FL_global_model' in log (preferred over 'FL_global_model' "
            "because model_name contains 'best')"
        )

    def test_first_key_used_when_model_name_has_no_best(self):
        """Inventory scan: first key used when model_name does not contain "best"."""
        ml = _make_model_learnable()

        def _get(key, fl_ctx):
            if key == "FL_global_model":
                return ml
            return None

        persistor = _make_model_persistor(
            inventory={"FL_global_model": MagicMock(), "best_FL_global_model": MagicMock()}
        )
        persistor.get.side_effect = _get
        ctrl = _make_controller(persistor=persistor)

        with _mock_dxo_patch():
            result = ctrl._prepare_local_model("local_model", _make_fl_ctx(), _make_abort_signal())

        assert result.get_return_code() == ReturnCode.OK

    def test_persistor_exception_warns_and_falls_back_to_executor(self):
        """Persistor.get() raises exception → warning logged, executor tried.

        The executor fallback ensures backward-compatibility for setups where
        the persistor is misconfigured but the executor works.
        """
        persistor = _make_model_persistor()
        persistor.get.side_effect = RuntimeError("disk read error")

        executor = MagicMock()
        executor.execute.return_value = _ok_executor_result()

        ctrl = _make_controller(persistor=persistor, submit_model_executor=executor)
        ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        ctrl.log_warning.assert_called()
        warning_text = " ".join(str(c) for c in ctrl.log_warning.call_args_list)
        assert "persistor" in warning_text.lower(), "Warning must mention persistor failure"
        executor.execute.assert_called_once()

    def test_no_persistor_uses_executor_directly(self):
        """No persistor configured → executor used directly without warning."""
        executor = MagicMock()
        executor.execute.return_value = _ok_executor_result()

        ctrl = _make_controller(persistor=None, submit_model_executor=executor)
        result = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        assert result.get_return_code() == ReturnCode.OK
        executor.execute.assert_called_once()
        ctrl.log_warning.assert_not_called()

    def test_persistor_not_model_persistor_warns_and_falls_back(self):
        """Persistor is not ModelPersistor → isinstance guard logs warning, falls back.

        Uses if-not-isinstance (not assert) so it is safe under python -O.
        """
        bad_persistor = MagicMock()  # not spec'd as ModelPersistor — isinstance returns False
        executor = MagicMock()
        executor.execute.return_value = _ok_executor_result()

        ctrl = _make_controller(persistor=bad_persistor, submit_model_executor=executor)
        ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        ctrl.log_warning.assert_called()
        executor.execute.assert_called_once()

    def test_inventory_key_returns_none_warns_and_falls_back(self):
        """Inventory key selected but persistor.get(chosen_key) returns None → warning logged, executor used.

        Covers the edge case where the checkpoint file is deleted between the
        inventory query and the actual read (e.g. workspace cleanup race).
        """
        # get() returns None for all keys — simulates deleted checkpoint
        persistor = _make_model_persistor(
            inventory={"best_FL_global_model": MagicMock()},
            get_result=None,
        )
        executor = MagicMock()
        executor.execute.return_value = _ok_executor_result()

        ctrl = _make_controller(persistor=persistor, submit_model_executor=executor)
        ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        # Must warn that the key was found but returned None
        ctrl.log_warning.assert_called()
        warning_text = " ".join(str(c) for c in ctrl.log_warning.call_args_list)
        assert "best_FL_global_model" in warning_text, "Warning must name the inventory key that returned None"
        executor.execute.assert_called_once()

    def test_empty_inventory_falls_back_to_executor(self):
        """persistor.get() returns None and inventory is empty → executor used."""
        persistor = _make_model_persistor(inventory={}, get_result=None)
        executor = MagicMock()
        executor.execute.return_value = _ok_executor_result()

        ctrl = _make_controller(persistor=persistor, submit_model_executor=executor)
        ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        executor.execute.assert_called_once()

    def test_executor_bad_return_code_gives_execution_exception(self):
        """Executor returns non-OK return code → EXECUTION_EXCEPTION."""
        from nvflare.apis.shareable import Shareable

        bad = Shareable()
        bad.set_return_code(ReturnCode.EXECUTION_EXCEPTION)

        executor = MagicMock()
        executor.execute.return_value = bad

        ctrl = _make_controller(persistor=None, submit_model_executor=executor)
        result = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())

        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_no_persistor_no_executor_returns_bad_request(self):
        """No persistor AND no executor → BAD_REQUEST_DATA (nothing to load from)."""
        ctrl = _make_controller(persistor=None, submit_model_executor=None)
        result = ctrl._prepare_local_model("best_model", _make_fl_ctx(), _make_abort_signal())
        assert result.get_return_code() == ReturnCode.BAD_REQUEST_DATA
