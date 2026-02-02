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

from nvflare.app_common.utils.error_handling_utils import get_error_handling_message, should_ignore_result_error


class TestShouldIgnoreResultError:
    """Test should_ignore_result_error utility function."""

    def test_true_mode_always_ignores(self):
        """Test ignore_result_error=True always returns True (ignore error)."""
        failed_clients = set()
        result = should_ignore_result_error(
            ignore_result_error=True,
            client_name="site-1",
            failed_clients=failed_clients,
            num_targets=5,
            min_responses=3,
        )
        assert result is True
        # Client should not be added to failed_clients in True mode
        assert "site-1" not in failed_clients

    def test_false_mode_always_panics(self):
        """Test ignore_result_error=False always returns False (panic)."""
        failed_clients = set()
        result = should_ignore_result_error(
            ignore_result_error=False,
            client_name="site-1",
            failed_clients=failed_clients,
            num_targets=5,
            min_responses=3,
        )
        assert result is False
        # Client should not be added to failed_clients in False mode
        assert "site-1" not in failed_clients

    def test_dynamic_mode_ignores_when_min_responses_reachable(self):
        """Test ignore_result_error=None ignores error when min_responses still reachable."""
        failed_clients = set()
        # 5 targets, 3 min_responses, 1 failure -> 4 remaining >= 3 -> ignore
        result = should_ignore_result_error(
            ignore_result_error=None,
            client_name="site-1",
            failed_clients=failed_clients,
            num_targets=5,
            min_responses=3,
        )
        assert result is True
        assert "site-1" in failed_clients

    def test_dynamic_mode_panics_when_min_responses_not_reachable(self):
        """Test ignore_result_error=None panics when min_responses not reachable."""
        failed_clients = {"site-1", "site-2"}  # 2 already failed
        # 5 targets, 3 min_responses, 3 failures (including new one) -> 2 remaining < 3 -> panic
        result = should_ignore_result_error(
            ignore_result_error=None,
            client_name="site-3",
            failed_clients=failed_clients,
            num_targets=5,
            min_responses=3,
        )
        assert result is False
        assert "site-3" in failed_clients

    def test_dynamic_mode_exact_threshold(self):
        """Test ignore_result_error=None at exact threshold boundary."""
        failed_clients = {"site-1"}  # 1 already failed
        # 5 targets, 3 min_responses, 2 failures -> 3 remaining == 3 -> ignore (just enough)
        result = should_ignore_result_error(
            ignore_result_error=None,
            client_name="site-2",
            failed_clients=failed_clients,
            num_targets=5,
            min_responses=3,
        )
        assert result is True
        assert "site-2" in failed_clients

    def test_dynamic_mode_all_must_succeed(self):
        """Test dynamic mode when min_responses equals num_targets (all must succeed)."""
        failed_clients = set()
        # 3 targets, 3 min_responses -> any failure means panic
        result = should_ignore_result_error(
            ignore_result_error=None,
            client_name="site-1",
            failed_clients=failed_clients,
            num_targets=3,
            min_responses=3,
        )
        assert result is False
        assert "site-1" in failed_clients

    def test_dynamic_mode_one_must_succeed(self):
        """Test dynamic mode when min_responses is 1."""
        failed_clients = {"site-1", "site-2"}  # 2 already failed
        # 3 targets, 1 min_responses, 3 failures -> 0 remaining < 1 -> panic
        result = should_ignore_result_error(
            ignore_result_error=None,
            client_name="site-3",
            failed_clients=failed_clients,
            num_targets=3,
            min_responses=1,
        )
        assert result is False
        assert "site-3" in failed_clients

    def test_dynamic_mode_can_lose_all_but_one(self):
        """Test dynamic mode allows losing all but min_responses clients."""
        failed_clients = {"site-1"}  # 1 already failed
        # 3 targets, 1 min_responses, 2 failures -> 1 remaining >= 1 -> ignore
        result = should_ignore_result_error(
            ignore_result_error=None,
            client_name="site-2",
            failed_clients=failed_clients,
            num_targets=3,
            min_responses=1,
        )
        assert result is True
        assert "site-2" in failed_clients


class TestGetErrorHandlingMessage:
    """Test get_error_handling_message utility function."""

    def test_true_mode_message(self):
        """Test message for ignore_result_error=True."""
        failed_clients = set()
        msg = get_error_handling_message(
            ignore_result_error=True,
            client_name="site-1",
            error_code="EXECUTION_EXCEPTION",
            current_round=5,
            controller_name="FedAvg",
            failed_clients=failed_clients,
            num_targets=3,
            min_responses=2,
        )
        assert "Ignore the result from site-1" in msg
        assert "round 5" in msg
        assert "EXECUTION_EXCEPTION" in msg

    def test_false_mode_message(self):
        """Test message for ignore_result_error=False."""
        failed_clients = set()
        msg = get_error_handling_message(
            ignore_result_error=False,
            client_name="site-1",
            error_code="TASK_ABORTED",
            current_round=3,
            controller_name="ScatterAndGather",
            failed_clients=failed_clients,
            num_targets=3,
            min_responses=2,
        )
        assert "Result from site-1 is bad" in msg
        assert "TASK_ABORTED" in msg
        assert "ScatterAndGather exiting" in msg
        assert "round 3" in msg

    def test_dynamic_mode_ignore_message(self):
        """Test message for ignore_result_error=None when ignoring."""
        failed_clients = {"site-1"}  # 1 already failed, will add site-2
        failed_clients.add("site-2")  # Simulate what should_ignore_result_error does
        msg = get_error_handling_message(
            ignore_result_error=None,
            client_name="site-2",
            error_code="EXECUTION_EXCEPTION",
            current_round=2,
            controller_name="FedAvg",
            failed_clients=failed_clients,
            num_targets=5,
            min_responses=3,
        )
        assert "Ignore the result from site-2" in msg
        assert "Remaining good clients (3) >= min_responses (3)" in msg

    def test_dynamic_mode_panic_message(self):
        """Test message for ignore_result_error=None when panicking."""
        failed_clients = {"site-1", "site-2", "site-3"}  # 3 already failed
        msg = get_error_handling_message(
            ignore_result_error=None,
            client_name="site-3",
            error_code="EXECUTION_EXCEPTION",
            current_round=1,
            controller_name="FedAvg",
            failed_clients=failed_clients,
            num_targets=5,
            min_responses=3,
        )
        assert "Result from site-3 is bad" in msg
        assert "Cannot reach min_responses" in msg
        assert "remaining good clients (2) < min_responses (3)" in msg


class TestIgnoreResultErrorIntegration:
    """Integration tests for ignore_result_error behavior in controllers."""

    def test_base_model_controller_default_is_none(self):
        """Test BaseModelController defaults to ignore_result_error=None."""
        from nvflare.app_common.workflows.fedavg import FedAvg

        controller = FedAvg()
        assert controller._ignore_result_error is None

    def test_base_model_controller_accepts_true(self):
        """Test BaseModelController accepts ignore_result_error=True."""
        from nvflare.app_common.workflows.fedavg import FedAvg

        controller = FedAvg(ignore_result_error=True)
        assert controller._ignore_result_error is True

    def test_base_model_controller_accepts_false(self):
        """Test BaseModelController accepts ignore_result_error=False."""
        from nvflare.app_common.workflows.fedavg import FedAvg

        controller = FedAvg(ignore_result_error=False)
        assert controller._ignore_result_error is False

    def test_scatter_and_gather_default_is_none(self):
        """Test ScatterAndGather defaults to ignore_result_error=None."""
        from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather

        controller = ScatterAndGather()
        assert controller.ignore_result_error is None

    def test_scatter_and_gather_accepts_all_modes(self):
        """Test ScatterAndGather accepts all three modes."""
        from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather

        controller_none = ScatterAndGather(ignore_result_error=None)
        assert controller_none.ignore_result_error is None

        controller_true = ScatterAndGather(ignore_result_error=True)
        assert controller_true.ignore_result_error is True

        controller_false = ScatterAndGather(ignore_result_error=False)
        assert controller_false.ignore_result_error is False


class TestAcceptTrainResultErrorHandling:
    """Test _accept_train_result error handling behavior in BaseModelController."""

    def _create_mock_result(self, return_code):
        """Create a mock Shareable result with given return code."""
        from nvflare.apis.shareable import Shareable

        result = Shareable()
        result.set_return_code(return_code)
        return result

    def _create_mock_fl_ctx(self):
        """Create a mock FLContext."""
        from unittest.mock import MagicMock

        fl_ctx = MagicMock()
        fl_ctx.set_prop = MagicMock()
        return fl_ctx

    def test_normal_task_with_ok_result_sets_context(self):
        """Test that normal task with OK result sets the result in context and returns True."""
        from nvflare.apis.fl_constant import ReturnCode
        from nvflare.app_common.app_constant import AppConstants
        from nvflare.app_common.workflows.fedavg import FedAvg

        controller = FedAvg()
        controller._current_failed_clients = set()
        controller._current_num_targets = 5
        controller._current_min_responses = 3

        result = self._create_mock_result(ReturnCode.OK)
        fl_ctx = self._create_mock_fl_ctx()

        accepted = controller._accept_train_result(client_name="site-1", result=result, fl_ctx=fl_ctx)

        assert accepted is True
        fl_ctx.set_prop.assert_called_once_with(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)

    def test_normal_task_with_error_and_ignore_does_not_set_context(self):
        """Test that normal task with error when should_ignore=True returns False without setting context."""
        from unittest.mock import MagicMock

        from nvflare.apis.fl_constant import ReturnCode
        from nvflare.app_common.workflows.fedavg import FedAvg

        controller = FedAvg(ignore_result_error=True)  # Always ignore
        controller._current_failed_clients = set()
        controller._current_num_targets = 5
        controller._current_min_responses = 3
        controller.warning = MagicMock()

        result = self._create_mock_result(ReturnCode.EXECUTION_EXCEPTION)
        fl_ctx = self._create_mock_fl_ctx()

        accepted = controller._accept_train_result(client_name="site-1", result=result, fl_ctx=fl_ctx)

        assert accepted is False
        # Should NOT set the errored result in context
        fl_ctx.set_prop.assert_not_called()
        # Should log a warning
        controller.warning.assert_called_once()

    def test_normal_task_with_error_and_panic_does_not_set_context(self):
        """Test that normal task with error when should_ignore=False returns False and panics."""
        from unittest.mock import MagicMock

        from nvflare.apis.fl_constant import ReturnCode
        from nvflare.app_common.workflows.fedavg import FedAvg

        controller = FedAvg(ignore_result_error=False)  # Always panic
        controller._current_failed_clients = set()
        controller._current_num_targets = 5
        controller._current_min_responses = 3
        controller.panic = MagicMock()

        result = self._create_mock_result(ReturnCode.EXECUTION_EXCEPTION)
        fl_ctx = self._create_mock_fl_ctx()

        accepted = controller._accept_train_result(client_name="site-1", result=result, fl_ctx=fl_ctx)

        assert accepted is False
        # Should NOT set the errored result in context
        fl_ctx.set_prop.assert_not_called()
        # Should panic
        controller.panic.assert_called_once()

    def test_unknown_task_always_ignores_errors(self):
        """Test that unknown/late tasks always ignore errors and return False regardless of controller setting."""
        from unittest.mock import MagicMock

        from nvflare.apis.fl_constant import ReturnCode
        from nvflare.app_common.workflows.fedavg import FedAvg

        # Even with ignore_result_error=False (strict mode), unknown tasks should ignore
        controller = FedAvg(ignore_result_error=False)
        controller._current_failed_clients = set()
        controller._current_num_targets = 5
        controller._current_min_responses = 3
        controller.warning = MagicMock()
        controller.panic = MagicMock()

        result = self._create_mock_result(ReturnCode.EXECUTION_EXCEPTION)
        fl_ctx = self._create_mock_fl_ctx()

        accepted = controller._accept_train_result(
            client_name="site-1", result=result, fl_ctx=fl_ctx, is_unknown_task=True
        )

        assert accepted is False
        # Should warn, NOT panic (because is_unknown_task=True forces ignore)
        controller.warning.assert_called_once()
        controller.panic.assert_not_called()
        # Should NOT set the errored result in context
        fl_ctx.set_prop.assert_not_called()

    def test_unknown_task_uses_empty_tracking_context(self):
        """Test that unknown tasks use empty tracking context to avoid stale data."""
        from unittest.mock import MagicMock

        from nvflare.apis.fl_constant import ReturnCode
        from nvflare.app_common.workflows.fedavg import FedAvg

        controller = FedAvg(ignore_result_error=None)  # Dynamic mode
        # Stale tracking data from previous task
        controller._current_failed_clients = {"site-1", "site-2"}
        controller._current_num_targets = 3
        controller._current_min_responses = 3
        controller.warning = MagicMock()
        controller.panic = MagicMock()

        result = self._create_mock_result(ReturnCode.EXECUTION_EXCEPTION)
        fl_ctx = self._create_mock_fl_ctx()

        # With is_unknown_task=True, should use empty/zero context
        # which means ignore_result_error=True (set in the method)
        accepted = controller._accept_train_result(
            client_name="site-3", result=result, fl_ctx=fl_ctx, is_unknown_task=True
        )

        assert accepted is False
        # Should warn (because unknown task forces ignore_result_error=True)
        controller.warning.assert_called_once()
        controller.panic.assert_not_called()

        # Verify stale _current_failed_clients was NOT modified
        assert "site-3" not in controller._current_failed_clients

    def test_unknown_task_with_ok_result_sets_context(self):
        """Test that unknown task with OK result returns True and sets result in context."""
        from nvflare.apis.fl_constant import ReturnCode
        from nvflare.app_common.app_constant import AppConstants
        from nvflare.app_common.workflows.fedavg import FedAvg

        controller = FedAvg()
        controller._current_failed_clients = set()
        controller._current_num_targets = 5
        controller._current_min_responses = 3

        result = self._create_mock_result(ReturnCode.OK)
        fl_ctx = self._create_mock_fl_ctx()

        accepted = controller._accept_train_result(
            client_name="site-1", result=result, fl_ctx=fl_ctx, is_unknown_task=True
        )

        assert accepted is True
        # OK result should still be set in context
        fl_ctx.set_prop.assert_called_once_with(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)
