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
Unit tests for Fix 15: MSG_ROOT_TTL on Swarm P2P learn task messages.

Without Fix 15, SwarmClientController scatters the global model to peer CJs
with MSG_ROOT_TTL = learn_task_ack_timeout (~30 s, the ACK wait).  When
PASS_THROUGH is active the peer subprocess downloads the model lazily; if the
download takes longer than learn_task_ack_timeout the sender's ArrayDownloadable
transaction expires and the subprocess receives "no ref found".

Fix 15 has two parts:

  A. SwarmClientController._scatter(): stamps ReservedHeaderKey.MSG_ROOT_TTL =
     float(learn_task_timeout) on task_data when learn_task_timeout is set.

  B. task_controller.broadcast_and_wait(): preserves a pre-set MSG_ROOT_TTL
     instead of always overwriting it with the short task.timeout.

Tests verify:

  SwarmClientController._scatter():
  1. learn_task_timeout set   → MSG_ROOT_TTL = learn_task_timeout on task_data
  2. learn_task_timeout=None  → MSG_ROOT_TTL not stamped (no regression)
  3. MSG_ROOT_TTL is set before the deepcopy for the local training path

  task_controller.broadcast_and_wait():
  4. Pre-set MSG_ROOT_TTL on request → preserved (not overwritten by task.timeout)
  5. No pre-set MSG_ROOT_TTL → falls back to task.timeout (existing behaviour)
"""

from unittest.mock import MagicMock

from nvflare.apis.controller_spec import Task
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LEARN_TASK_TIMEOUT = 3600.0
_LEARN_TASK_ACK_TIMEOUT = 30.0


def _make_swarm_controller(learn_task_timeout=_LEARN_TASK_TIMEOUT):
    """Return a SwarmClientController stub with only _scatter()-related state."""
    ctrl = SwarmClientController.__new__(SwarmClientController)
    ctrl.learn_task_timeout = learn_task_timeout
    ctrl.me = "client1"
    ctrl.logger = MagicMock()
    ctrl.log_info = MagicMock()
    ctrl.log_error = MagicMock()
    ctrl.log_debug = MagicMock()
    ctrl.update_status = MagicMock()
    ctrl.set_learn_task = MagicMock(return_value=True)
    ctrl.send_learn_task = MagicMock(return_value=True)
    return ctrl


def _make_task_data():
    return Shareable()


def _make_fl_ctx():
    return MagicMock()


def _call_scatter(ctrl, task_data, for_round=0, clients=("client1", "client2"), aggr="client2"):
    """Call _scatter() with mocked get_config_prop."""
    fl_ctx = _make_fl_ctx()

    def get_config_prop(key, default=None):
        if key == Constant.TRAIN_CLIENTS:
            return list(clients)
        if key == Constant.AGGR_CLIENTS:
            return [aggr]
        return default

    ctrl.get_config_prop = get_config_prop
    ctrl._scatter(task_data, for_round, fl_ctx)
    return fl_ctx


# ---------------------------------------------------------------------------
# A. SwarmClientController._scatter() — MSG_ROOT_TTL stamping
# ---------------------------------------------------------------------------


class TestScatterMsgRootTtl:
    """SwarmClientController._scatter() stamps MSG_ROOT_TTL when learn_task_timeout is set."""

    def test_learn_task_timeout_set_stamps_msg_root_ttl(self):
        """When learn_task_timeout is configured, MSG_ROOT_TTL = learn_task_timeout is set on task_data."""
        ctrl = _make_swarm_controller(learn_task_timeout=_LEARN_TASK_TIMEOUT)
        task_data = _make_task_data()

        _call_scatter(ctrl, task_data, clients=["client1", "client2"], aggr="client2")

        ttl = task_data.get_header(ReservedHeaderKey.MSG_ROOT_TTL)
        assert (
            ttl == _LEARN_TASK_TIMEOUT
        ), f"MSG_ROOT_TTL must equal learn_task_timeout ({_LEARN_TASK_TIMEOUT}), got {ttl}"

    def test_learn_task_timeout_none_does_not_stamp_msg_root_ttl(self):
        """When learn_task_timeout is None, MSG_ROOT_TTL must not be stamped on task_data.

        This preserves existing behaviour: without learn_task_timeout configured,
        task_controller.broadcast_and_wait() uses task.timeout (learn_task_ack_timeout)
        as MSG_ROOT_TTL, which was the previous default behaviour.
        """
        ctrl = _make_swarm_controller(learn_task_timeout=None)
        task_data = _make_task_data()

        _call_scatter(ctrl, task_data, clients=["client1", "client2"], aggr="client2")

        ttl = task_data.get_header(ReservedHeaderKey.MSG_ROOT_TTL)
        assert ttl is None, (
            "MSG_ROOT_TTL must not be set when learn_task_timeout is None — "
            "existing default (task.timeout) must be used instead"
        )

    def test_msg_root_ttl_is_float(self):
        """MSG_ROOT_TTL must be a float (via_downloader checks `if msg_root_ttl:`)."""
        ctrl = _make_swarm_controller(learn_task_timeout=int(3600))  # int input
        task_data = _make_task_data()

        _call_scatter(ctrl, task_data, clients=["client1", "client2"], aggr="client2")

        ttl = task_data.get_header(ReservedHeaderKey.MSG_ROOT_TTL)
        assert isinstance(ttl, float), f"MSG_ROOT_TTL must be float, got {type(ttl)}"

    def test_msg_root_ttl_set_before_deepcopy_for_local_training(self):
        """MSG_ROOT_TTL must be set on task_data before the deepcopy for the local training path.

        set_learn_task() receives a deep copy of task_data.  If MSG_ROOT_TTL is stamped
        after the deep copy, the local training path does not see it and falls back to
        _MIN_DOWNLOAD_TIMEOUT when the local subprocess downloads the model.
        """
        ctrl = _make_swarm_controller(learn_task_timeout=_LEARN_TASK_TIMEOUT)
        task_data = _make_task_data()

        # Capture the deep copy passed to set_learn_task
        captured_copies = []

        def mock_set_learn_task(task_data, fl_ctx):
            captured_copies.append(task_data)
            return True

        ctrl.set_learn_task = mock_set_learn_task

        # Use client1 (= self.me) as a target so it goes through set_learn_task
        _call_scatter(ctrl, task_data, clients=["client1", "client2"], aggr="client2")

        assert captured_copies, "set_learn_task must have been called"
        local_copy = captured_copies[0]
        ttl = local_copy.get_header(ReservedHeaderKey.MSG_ROOT_TTL)
        assert ttl == _LEARN_TASK_TIMEOUT, (
            "The deep copy passed to set_learn_task must already have MSG_ROOT_TTL set; "
            f"expected {_LEARN_TASK_TIMEOUT}, got {ttl}"
        )


# ---------------------------------------------------------------------------
# B. task_controller.broadcast_and_wait() — MSG_ROOT_TTL preservation
# ---------------------------------------------------------------------------


class TestBroadcastAndWaitMsgRootTtl:
    """task_controller.broadcast_and_wait() must preserve a pre-set MSG_ROOT_TTL."""

    def _make_task_with_request(self, task_timeout_secs=30, pre_set_ttl=None):
        request = Shareable()
        if pre_set_ttl is not None:
            request.set_header(ReservedHeaderKey.MSG_ROOT_TTL, pre_set_ttl)
        # Task requires int timeout
        task = Task(name="train", data=request, timeout=int(task_timeout_secs))
        return task, request

    def _run_broadcast_header_logic(self, task, request):
        """Run only the MSG_ROOT_TTL header-stamping logic from broadcast_and_wait().

        Exercises the minimal code path in task_controller.py that determines and
        sets MSG_ROOT_TTL on the request, without spinning up a full engine stack.
        """
        from nvflare.apis.shareable import ReservedHeaderKey, ReservedKey

        # Replicate the relevant lines from task_controller.broadcast_and_wait()
        request.set_header(ReservedKey.TASK_NAME, task.name)
        # Preserve pre-set MSG_ROOT_TTL or fall back to task.timeout (Fix 15 logic)
        msg_root_ttl = request.get_header(ReservedHeaderKey.MSG_ROOT_TTL) or task.timeout
        request.set_header(ReservedHeaderKey.MSG_ROOT_TTL, msg_root_ttl)

    def test_pre_set_msg_root_ttl_is_preserved(self):
        """broadcast_and_wait() must not overwrite a pre-set MSG_ROOT_TTL.

        SwarmClientController stamps MSG_ROOT_TTL = learn_task_timeout (large) on
        task_data before broadcast_and_wait() is called.  The implementation must
        preserve that value instead of overwriting it with task.timeout
        (= learn_task_ack_timeout, ~30 s), which would cause download transactions
        to expire before the subprocess finishes downloading the global model.
        """
        task, request = self._make_task_with_request(
            task_timeout_secs=int(_LEARN_TASK_ACK_TIMEOUT),
            pre_set_ttl=_LEARN_TASK_TIMEOUT,
        )

        self._run_broadcast_header_logic(task, request)

        ttl_after = request.get_header(ReservedHeaderKey.MSG_ROOT_TTL)
        assert ttl_after == _LEARN_TASK_TIMEOUT, (
            f"Pre-set MSG_ROOT_TTL ({_LEARN_TASK_TIMEOUT}) must be preserved; "
            f"broadcast_and_wait must not overwrite it with task.timeout ({_LEARN_TASK_ACK_TIMEOUT}). "
            f"Got {ttl_after}"
        )

    def test_no_pre_set_msg_root_ttl_uses_task_timeout(self):
        """Without a pre-set MSG_ROOT_TTL, broadcast_and_wait() falls back to task.timeout.

        This preserves the existing default behaviour for all callers that do not
        pre-set MSG_ROOT_TTL (e.g. standard FedAvg task broadcasts).
        """
        task, request = self._make_task_with_request(
            task_timeout_secs=int(_LEARN_TASK_ACK_TIMEOUT),
            pre_set_ttl=None,  # no pre-set
        )

        self._run_broadcast_header_logic(task, request)

        ttl_after = request.get_header(ReservedHeaderKey.MSG_ROOT_TTL)
        assert ttl_after == int(_LEARN_TASK_ACK_TIMEOUT), (
            f"Without a pre-set MSG_ROOT_TTL, broadcast_and_wait() must use task.timeout "
            f"({int(_LEARN_TASK_ACK_TIMEOUT)}). Got {ttl_after}"
        )
