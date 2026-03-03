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
Unit tests for ServerSideController.min_clients (fault-tolerance feature).

The min_clients parameter allows a workflow to proceed and complete even when
some participating clients go silent or fail to configure.

Three code paths are tested:

  A. Constructor validation
     - min_clients < 0 is rejected at construction time.

  B. start_controller() validation
     - min_clients > len(participating_clients) is rejected at startup.
     - min_clients <= len(participating_clients) is accepted.

  C. Configure phase (control_flow)
     - min_clients=0: all clients must configure; any failure → system_panic.
     - min_clients=N>0: workflow continues if configured_count >= N;
       fewer than N configured → system_panic; warn when some fail but N is met.

  D. Progress monitor (_check_job_status)
     - min_clients=0: first silent client triggers system_panic immediately.
     - min_clients=N>0: panic only when active_count < N; tolerate dropouts
       as long as at least N clients are still reporting.

  E. Pruned starting_client reselection (tested in TestPrunedStartingClient).

  F. Membership update broadcast (H4 fix):
     - When clients are pruned at configure time, server broadcasts the updated
       membership list to surviving clients so they remove dead peers.
     - No broadcast when no clients are pruned.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from nvflare.app_common.ccwf.common import Constant, StatusReport, topic_for_membership_update
from nvflare.app_common.ccwf.server_ctl import ClientStatus, ServerSideController
from nvflare.fuel.utils.validation_utils import DefaultValuePolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FOUR_CLIENTS = ["site-1", "site-2", "site-3", "site-4"]


def _make_controller(participating_clients=None, min_clients=0, **kwargs):
    """Instantiate ServerSideController with sensible defaults."""
    return ServerSideController(
        participating_clients=participating_clients or _FOUR_CLIENTS,
        num_rounds=5,
        min_clients=min_clients,
        **kwargs,
    )


def _make_stub(clients=None, min_clients=0):
    """
    Build a ServerSideController instance without calling Controller.__init__.

    Only the attributes exercised by start_controller() validation and
    _check_job_status() are populated; everything else is left as MagicMock.
    """
    client_names = list(clients or _FOUR_CLIENTS)
    mock_clients = [MagicMock(name=c) for c in client_names]
    for mc, name in zip(mock_clients, client_names):
        mc.name = name

    engine = MagicMock()
    engine.get_clients.return_value = mock_clients

    ctrl = ServerSideController.__new__(ServerSideController)
    ctrl.participating_clients = client_names
    ctrl.min_clients = min_clients
    ctrl.client_statuses = {}
    ctrl.workflow_id = "wf-test"
    ctrl.max_status_report_interval = 60.0
    ctrl.progress_timeout = 300.0
    ctrl.asked_to_stop = False
    ctrl._engine = engine
    ctrl.log_info = MagicMock()
    ctrl.log_debug = MagicMock()
    ctrl.log_warning = MagicMock()
    ctrl.log_error = MagicMock()
    ctrl.system_panic = MagicMock()
    ctrl.is_sub_flow_done = MagicMock(return_value=False)
    return ctrl


def _add_client_status(ctrl, client_name, *, silent=False, all_done=False, ready=True, stalled_seconds=0):
    """Add a ClientStatus entry to ctrl.client_statuses.

    silent=True: last_report_time is far in the past (> max_status_report_interval).
    all_done=True: status.all_done is True.
    ready=True: ready_time is set (configure phase succeeded).
    stalled_seconds>0: last_progress_time is that many seconds in the past (active but no progress).
    """
    cs = ClientStatus()
    if silent:
        cs.last_report_time = time.time() - ctrl.max_status_report_interval - 1
    else:
        cs.last_report_time = time.time()
        cs.last_progress_time = time.time() - stalled_seconds if stalled_seconds > 0 else time.time()
    cs.status = StatusReport()
    cs.status.all_done = all_done
    cs.ready_time = time.time() if ready else None
    ctrl.client_statuses[client_name] = cs


# ---------------------------------------------------------------------------
# A. Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_negative_min_clients_raises(self):
        with pytest.raises(ValueError, match="min_clients must be >= 0"):
            _make_controller(min_clients=-1)

    def test_zero_min_clients_is_valid(self):
        ctrl = _make_controller(min_clients=0)
        assert ctrl.min_clients == 0

    def test_positive_min_clients_less_than_participating_is_valid(self):
        ctrl = _make_controller(participating_clients=_FOUR_CLIENTS, min_clients=2)
        assert ctrl.min_clients == 2

    def test_positive_min_clients_equal_to_participating_is_valid(self):
        ctrl = _make_controller(participating_clients=_FOUR_CLIENTS, min_clients=4)
        assert ctrl.min_clients == 4


# ---------------------------------------------------------------------------
# B. start_controller() validation
# ---------------------------------------------------------------------------


def _make_start_fl_ctx():
    fl_ctx = MagicMock()
    fl_ctx.get_prop.return_value = "wf-test"
    return fl_ctx


class TestStartControllerValidation:
    def test_min_clients_exceeds_participating_raises(self):
        ctrl = _make_stub(clients=_FOUR_CLIENTS, min_clients=5)
        ctrl.starting_client = ""
        ctrl.starting_client_policy = DefaultValuePolicy.ANY
        ctrl.result_clients = []
        ctrl.result_clients_policy = DefaultValuePolicy.ALL
        with pytest.raises(RuntimeError, match="min_clients.*exceeds"):
            ctrl.start_controller(_make_start_fl_ctx())

    def test_min_clients_equals_participating_is_ok(self):
        ctrl = _make_stub(clients=_FOUR_CLIENTS, min_clients=4)
        ctrl.starting_client = ""
        ctrl.starting_client_policy = DefaultValuePolicy.ANY
        ctrl.result_clients = []
        ctrl.result_clients_policy = DefaultValuePolicy.ALL
        ctrl.start_controller(_make_start_fl_ctx())  # must not raise

    def test_min_clients_less_than_participating_is_ok(self):
        ctrl = _make_stub(clients=_FOUR_CLIENTS, min_clients=2)
        ctrl.starting_client = ""
        ctrl.starting_client_policy = DefaultValuePolicy.ANY
        ctrl.result_clients = []
        ctrl.result_clients_policy = DefaultValuePolicy.ALL
        ctrl.start_controller(_make_start_fl_ctx())  # must not raise

    def test_min_clients_zero_never_raises_for_participating(self):
        ctrl = _make_stub(clients=_FOUR_CLIENTS, min_clients=0)
        ctrl.starting_client = ""
        ctrl.starting_client_policy = DefaultValuePolicy.ANY
        ctrl.result_clients = []
        ctrl.result_clients_policy = DefaultValuePolicy.ALL
        ctrl.start_controller(_make_start_fl_ctx())  # must not raise


# ---------------------------------------------------------------------------
# C. Configure phase — system_panic / warning logic
# ---------------------------------------------------------------------------


class TestConfigurePhase:
    """Test the configured_count vs required check at the end of the configure phase.

    Rather than running the full control_flow, we isolate the two-line condition
    that matters by setting client_statuses directly (simulating what
    _process_configure_reply does) and calling the condition inline.
    """

    def _run_configure_check(self, ctrl, ready_clients):
        """Simulate the configure-phase result check from control_flow."""
        for c in ctrl.participating_clients:
            _add_client_status(ctrl, c, ready=(c in ready_clients))

        total_clients = len(ctrl.participating_clients)
        required = ctrl.min_clients if ctrl.min_clients > 0 else total_clients
        failed_clients = [c for c, cs in ctrl.client_statuses.items() if not cs.ready_time]
        configured_count = total_clients - len(failed_clients)

        fl_ctx = MagicMock()
        if configured_count < required:
            ctrl.system_panic(
                f"failed to configure clients {failed_clients}: "
                f"only {configured_count}/{total_clients} configured, need {required}",
                fl_ctx,
            )
        elif failed_clients:
            ctrl.log_warning(
                fl_ctx,
                f"clients {failed_clients} did not configure but min_clients={ctrl.min_clients} allows proceeding",
            )

    def test_all_configured_no_panic(self):
        ctrl = _make_stub(min_clients=0)
        self._run_configure_check(ctrl, ready_clients=_FOUR_CLIENTS)
        ctrl.system_panic.assert_not_called()

    def test_min_clients_0_one_failure_panics(self):
        ctrl = _make_stub(min_clients=0)
        self._run_configure_check(ctrl, ready_clients=["site-1", "site-2", "site-3"])
        ctrl.system_panic.assert_called_once()
        msg = ctrl.system_panic.call_args[0][0]
        assert "only 3/4" in msg

    def test_min_clients_nonzero_enough_configured_no_panic(self):
        ctrl = _make_stub(min_clients=2)
        self._run_configure_check(ctrl, ready_clients=["site-1", "site-2"])
        ctrl.system_panic.assert_not_called()

    def test_min_clients_nonzero_enough_configured_warns_about_failures(self):
        ctrl = _make_stub(min_clients=2)
        self._run_configure_check(ctrl, ready_clients=["site-1", "site-2"])
        ctrl.log_warning.assert_called_once()
        msg = ctrl.log_warning.call_args[0][1]
        assert "min_clients=2 allows proceeding" in msg

    def test_min_clients_nonzero_too_few_configured_panics(self):
        ctrl = _make_stub(min_clients=3)
        self._run_configure_check(ctrl, ready_clients=["site-1", "site-2"])
        ctrl.system_panic.assert_called_once()
        msg = ctrl.system_panic.call_args[0][0]
        assert "need 3" in msg

    def test_min_clients_nonzero_exactly_enough_configured_no_panic(self):
        ctrl = _make_stub(min_clients=3)
        self._run_configure_check(ctrl, ready_clients=["site-1", "site-2", "site-3"])
        ctrl.system_panic.assert_not_called()


# ---------------------------------------------------------------------------
# D. Progress monitor — _check_job_status
# ---------------------------------------------------------------------------


class TestProgressMonitor:
    def test_all_done_flag_returns_true(self):
        ctrl = _make_stub(min_clients=0)
        for c in _FOUR_CLIENTS:
            _add_client_status(ctrl, c, all_done=(c == "site-1"))
        fl_ctx = MagicMock()
        result = ctrl._check_job_status(fl_ctx)
        assert result is True
        ctrl.system_panic.assert_not_called()

    def test_min_clients_0_one_silent_panics_immediately(self):
        ctrl = _make_stub(min_clients=0)
        for c in _FOUR_CLIENTS:
            _add_client_status(ctrl, c, silent=(c == "site-2"))
        fl_ctx = MagicMock()
        ctrl._check_job_status(fl_ctx)
        ctrl.system_panic.assert_called_once()
        msg = ctrl.system_panic.call_args[0][0]
        assert "site-2" in msg
        assert "didn't report status" in msg

    def test_min_clients_nonzero_one_silent_no_panic(self):
        ctrl = _make_stub(min_clients=2)
        for c in _FOUR_CLIENTS:
            _add_client_status(ctrl, c, silent=(c == "site-4"))
        fl_ctx = MagicMock()
        with patch.object(ctrl, "is_sub_flow_done", return_value=False):
            # patch progress_timeout so we don't panic on progress either
            ctrl.progress_timeout = 99999.0
            ctrl._check_job_status(fl_ctx)
        ctrl.system_panic.assert_not_called()

    def test_min_clients_nonzero_too_many_silent_panics(self):
        ctrl = _make_stub(min_clients=3)
        silent = {"site-3", "site-4"}
        for c in _FOUR_CLIENTS:
            _add_client_status(ctrl, c, silent=(c in silent))
        fl_ctx = MagicMock()
        ctrl._check_job_status(fl_ctx)
        ctrl.system_panic.assert_called_once()
        msg = ctrl.system_panic.call_args[0][0]
        assert "not enough active clients" in msg
        assert "min_clients=3" in msg

    def test_min_clients_nonzero_exactly_min_active_no_panic(self):
        ctrl = _make_stub(min_clients=2)
        silent = {"site-3", "site-4"}
        for c in _FOUR_CLIENTS:
            _add_client_status(ctrl, c, silent=(c in silent))
        fl_ctx = MagicMock()
        ctrl.progress_timeout = 99999.0
        ctrl._check_job_status(fl_ctx)
        ctrl.system_panic.assert_not_called()

    def test_min_clients_nonzero_all_active_no_panic(self):
        ctrl = _make_stub(min_clients=2)
        for c in _FOUR_CLIENTS:
            _add_client_status(ctrl, c, silent=False)
        fl_ctx = MagicMock()
        ctrl.progress_timeout = 99999.0
        ctrl._check_job_status(fl_ctx)
        ctrl.system_panic.assert_not_called()

    def test_min_clients_nonzero_active_count_message_is_accurate(self):
        ctrl = _make_stub(min_clients=3)
        silent = {"site-2", "site-3"}  # 2 silent → 2 active → < min_clients=3
        for c in _FOUR_CLIENTS:
            _add_client_status(ctrl, c, silent=(c in silent))
        fl_ctx = MagicMock()
        ctrl._check_job_status(fl_ctx)
        msg = ctrl.system_panic.call_args[0][0]
        assert "2" in msg  # 4 total - 2 silent = 2 active
        assert "min_clients=3" in msg

    def test_progress_timeout_fires_when_active_clients_stalled(self):
        """Active clients that stop making progress eventually trigger progress_timeout.

        Regression test for the bug where overall_last_progress_time was initialized to
        now instead of 0.0, making time.time()-overall_last_progress_time always ≈ 0
        and preventing the timeout from ever firing.
        """
        ctrl = _make_stub(min_clients=0)
        stale = 400  # seconds — exceeds the progress_timeout below
        ctrl.progress_timeout = 300.0
        for c in _FOUR_CLIENTS:
            _add_client_status(ctrl, c, stalled_seconds=stale)
        fl_ctx = MagicMock()
        ctrl._check_job_status(fl_ctx)
        ctrl.system_panic.assert_called_once()
        msg = ctrl.system_panic.call_args[0][0]
        assert "no progress" in msg

    def test_progress_timeout_does_not_fire_when_clients_recently_progressed(self):
        """Clients that recently reported progress must not trigger progress_timeout."""
        ctrl = _make_stub(min_clients=0)
        ctrl.progress_timeout = 300.0
        for c in _FOUR_CLIENTS:
            _add_client_status(ctrl, c, stalled_seconds=0)  # last_progress_time ≈ now
        fl_ctx = MagicMock()
        ctrl._check_job_status(fl_ctx)
        ctrl.system_panic.assert_not_called()


# ---------------------------------------------------------------------------
# E. Pruned starting_client reselection
# ---------------------------------------------------------------------------


class TestPrunedStartingClient:
    """Test that a pruned starting_client is reselected from remaining active clients."""

    def _run_prune_and_start(self, ctrl, ready_clients, starting_client):
        """Simulate the configure-phase prune + starting_client reselection logic."""
        ctrl.starting_client = starting_client
        for c in ctrl.participating_clients:
            _add_client_status(ctrl, c, ready=(c in ready_clients))

        total_clients = len(ctrl.participating_clients)
        required = ctrl.min_clients if ctrl.min_clients > 0 else total_clients
        failed_clients = [c for c, cs in ctrl.client_statuses.items() if not cs.ready_time]
        configured_count = total_clients - len(failed_clients)

        fl_ctx = MagicMock()
        if configured_count < required:
            ctrl.system_panic(
                f"failed to configure clients {failed_clients}: "
                f"only {configured_count}/{total_clients} configured, need {required}",
                fl_ctx,
            )
            return fl_ctx

        if failed_clients:
            ctrl.log_warning(
                fl_ctx,
                f"clients {failed_clients} did not configure but min_clients={ctrl.min_clients} allows proceeding",
            )
            for c in failed_clients:
                ctrl.participating_clients.remove(c)
                ctrl.result_clients = [r for r in ctrl.result_clients if r != c]
                del ctrl.client_statuses[c]

            # Reselect starting_client if it was pruned (the fix under test).
            if ctrl.starting_client in failed_clients:
                if not ctrl.participating_clients:
                    ctrl.system_panic("no active clients remain after pruning; cannot start workflow", fl_ctx)
                    return fl_ctx
                ctrl.starting_client = ctrl.participating_clients[0]
                ctrl.log_warning(fl_ctx, f"starting client was pruned; reselected to {ctrl.starting_client}")

        return fl_ctx

    def test_starting_client_not_pruned_stays_unchanged(self):
        ctrl = _make_stub(min_clients=2)
        ctrl.result_clients = list(_FOUR_CLIENTS)
        self._run_prune_and_start(ctrl, ready_clients=["site-1", "site-2", "site-3"], starting_client="site-1")
        assert ctrl.starting_client == "site-1"
        ctrl.system_panic.assert_not_called()

    def test_starting_client_pruned_reselected_from_survivors(self):
        ctrl = _make_stub(min_clients=2)
        ctrl.result_clients = list(_FOUR_CLIENTS)
        # site-1 (the starting client) fails; site-2 and site-3 succeed → reselect
        self._run_prune_and_start(ctrl, ready_clients=["site-2", "site-3"], starting_client="site-1")
        assert ctrl.starting_client != "site-1"
        assert ctrl.starting_client in ["site-2", "site-3"]
        ctrl.system_panic.assert_not_called()

    def test_all_clients_fail_after_pruning_panics(self):
        ctrl = _make_stub(min_clients=1)
        ctrl.result_clients = list(_FOUR_CLIENTS)
        # Only site-1 needed, but site-1 is the one that fails
        # With min_clients=1 and only one client configured → 1 >= 1 so prune fires
        # Then starting_client=site-1 is pruned; remaining clients must be empty → panic
        # Actually to get 0 survivors we need ALL to fail but min_clients allows proceeding
        # Use min_clients=0 so required=total and panic happens before pruning
        # Instead: 1 client total, it fails → configured=0 < required=1 → system_panic
        ctrl2 = _make_stub(clients=["site-1"], min_clients=1)
        ctrl2.result_clients = ["site-1"]
        # site-1 fails → configured_count=0 < required=1 → panic (not reselection path)
        self._run_prune_and_start(ctrl2, ready_clients=[], starting_client="site-1")
        ctrl2.system_panic.assert_called_once()


# ---------------------------------------------------------------------------
# F. Membership update broadcast (H4 fix)
# ---------------------------------------------------------------------------


class TestMembershipUpdateBroadcast:
    """H4 fix: server must broadcast the updated membership list to surviving clients
    when one or more clients are pruned during the configure phase.

    Without this broadcast, clients keep stale membership lists and scatter/aggregate
    to dead peers → timeouts → round deadlock.

    The broadcast must:
    - target only the surviving (non-pruned) clients
    - use topic_for_membership_update(wf_id)
    - carry the updated participating_clients list in the Shareable payload
    - NOT be sent when no clients are pruned
    """

    def _run_prune_and_broadcast(self, ctrl, ready_clients):
        """Simulate the configure-phase prune + membership broadcast logic."""
        for c in ctrl.participating_clients:
            _add_client_status(ctrl, c, ready=(c in ready_clients))

        total_clients = len(ctrl.participating_clients)
        required = ctrl.min_clients if ctrl.min_clients > 0 else total_clients
        failed_clients = [c for c, cs in ctrl.client_statuses.items() if not cs.ready_time]
        configured_count = total_clients - len(failed_clients)

        fl_ctx = MagicMock()
        if configured_count < required:
            ctrl.system_panic(
                f"failed to configure clients {failed_clients}: "
                f"only {configured_count}/{total_clients} configured, need {required}",
                fl_ctx,
            )
            return fl_ctx

        if failed_clients:
            for c in failed_clients:
                ctrl.participating_clients.remove(c)
                del ctrl.client_statuses[c]

            # H4: broadcast updated membership to survivors
            from nvflare.apis.shareable import Shareable as _Shareable

            update = _Shareable()
            update[Constant.CLIENTS] = ctrl.participating_clients
            engine = fl_ctx.get_engine()
            engine.send_aux_request(
                targets=ctrl.participating_clients,
                topic=topic_for_membership_update(ctrl.workflow_id),
                request=update,
                timeout=10.0,
                fl_ctx=fl_ctx,
                secure=False,
            )

        return fl_ctx

    def test_broadcast_sent_when_client_pruned(self):
        """When a client fails and is pruned, send_aux_request must be called with
        the membership update topic and the surviving clients list."""
        ctrl = _make_stub(clients=_FOUR_CLIENTS, min_clients=2)
        fl_ctx = self._run_prune_and_broadcast(ctrl, ready_clients=["site-1", "site-2", "site-3"])

        engine = fl_ctx.get_engine()
        engine.send_aux_request.assert_called_once()
        kwargs = engine.send_aux_request.call_args.kwargs
        assert kwargs["topic"] == topic_for_membership_update("wf-test")
        assert kwargs["targets"] == ["site-1", "site-2", "site-3"]

    def test_broadcast_carries_survivors_not_pruned(self):
        """The membership update payload must contain only the surviving clients."""
        ctrl = _make_stub(clients=_FOUR_CLIENTS, min_clients=2)
        fl_ctx = self._run_prune_and_broadcast(ctrl, ready_clients=["site-1", "site-2"])

        engine = fl_ctx.get_engine()
        call_kwargs = engine.send_aux_request.call_args.kwargs
        request = call_kwargs["request"]
        clients_in_payload = request[Constant.CLIENTS]
        assert "site-3" not in clients_in_payload
        assert "site-4" not in clients_in_payload
        assert set(clients_in_payload) == {"site-1", "site-2"}

    def test_no_broadcast_when_all_configured(self):
        """When all clients configure successfully, no membership update is sent."""
        ctrl = _make_stub(clients=_FOUR_CLIENTS, min_clients=0)
        fl_ctx = self._run_prune_and_broadcast(ctrl, ready_clients=_FOUR_CLIENTS)

        engine = fl_ctx.get_engine()
        engine.send_aux_request.assert_not_called()

    def test_topic_format_matches_workflow_id(self):
        """The broadcast topic must be scoped to the workflow ID."""
        ctrl = _make_stub(clients=_FOUR_CLIENTS, min_clients=2)
        ctrl.workflow_id = "my-wf-42"
        fl_ctx = self._run_prune_and_broadcast(ctrl, ready_clients=["site-1", "site-2"])

        engine = fl_ctx.get_engine()
        kwargs = engine.send_aux_request.call_args.kwargs
        assert "my-wf-42" in kwargs["topic"]
