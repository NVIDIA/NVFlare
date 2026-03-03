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

"""Tests for H4 fix: server-to-client membership update broadcast.

H4 fixes the stale membership list problem: when the server prunes failed
clients at configure time (min_clients > 0), surviving clients previously kept
their original trainers/aggrs lists.  _scatter() and result-submission would
then target the pruned (unreachable) clients, causing timeouts and round deadlock.

Fix:
  - server_ctl.py: broadcast updated active-client list after pruning
  - swarm_client_ctl.py: register _handle_membership_update handler in
    process_config(); handler updates self.trainers and self.aggrs

These tests exercise the client-side handler (_handle_membership_update).
Server-side broadcast tests are in test_server_ctl_min_clients.py.
"""
from unittest.mock import MagicMock

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable
from nvflare.app_common.ccwf.common import Constant, topic_for_membership_update
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController


def _make_handler_ctl(me="site-1", trainers=None, aggrs=None):
    """Build a minimal SwarmClientController for testing _handle_membership_update."""
    ctl = SwarmClientController.__new__(SwarmClientController)
    ctl.logger = MagicMock()
    ctl.log_info = MagicMock()
    ctl.log_warning = MagicMock()
    ctl.log_error = MagicMock()
    ctl.log_debug = MagicMock()
    ctl.me = me
    ctl.workflow_id = "wf-test"
    ctl.trainers = list(trainers or [me, "site-2", "site-3"])
    ctl.aggrs = list(aggrs or [me, "site-2", "site-3"])
    return ctl


def _make_update_request(clients):
    """Build the membership-update Shareable as the server sends it."""
    req = Shareable()
    req[Constant.CLIENTS] = list(clients)
    return req


class TestHandleMembershipUpdate:
    """Client-side membership update handler (_handle_membership_update).

    The handler must:
    - update self.trainers and self.aggrs by removing clients absent from new_clients
    - log the update at INFO level
    - return OK
    - handle empty payload gracefully (log warning, return OK)
    """

    def test_dead_trainer_removed_from_trainers(self):
        """Pruned client must be removed from self.trainers."""
        ctl = _make_handler_ctl(trainers=["site-1", "site-2", "site-3"])
        req = _make_update_request(["site-1", "site-2"])  # site-3 pruned
        fl_ctx = MagicMock()

        ctl._handle_membership_update("topic", req, fl_ctx)

        assert "site-3" not in ctl.trainers
        assert set(ctl.trainers) == {"site-1", "site-2"}

    def test_dead_aggr_removed_from_aggrs(self):
        """Pruned client must be removed from self.aggrs."""
        ctl = _make_handler_ctl(aggrs=["site-1", "site-2", "site-3"])
        req = _make_update_request(["site-1", "site-3"])  # site-2 pruned
        fl_ctx = MagicMock()

        ctl._handle_membership_update("topic", req, fl_ctx)

        assert "site-2" not in ctl.aggrs
        assert set(ctl.aggrs) == {"site-1", "site-3"}

    def test_surviving_clients_stay_in_both_lists(self):
        """Clients in the new list must remain in trainers and aggrs."""
        ctl = _make_handler_ctl(
            trainers=["site-1", "site-2", "site-3"],
            aggrs=["site-1", "site-2", "site-3"],
        )
        req = _make_update_request(["site-1", "site-2"])
        fl_ctx = MagicMock()

        ctl._handle_membership_update("topic", req, fl_ctx)

        assert "site-1" in ctl.trainers
        assert "site-2" in ctl.trainers
        assert "site-1" in ctl.aggrs
        assert "site-2" in ctl.aggrs

    def test_returns_ok(self):
        """Handler must return ReturnCode.OK to ACK the server."""
        ctl = _make_handler_ctl()
        req = _make_update_request(["site-1", "site-2"])
        fl_ctx = MagicMock()

        reply = ctl._handle_membership_update("topic", req, fl_ctx)

        rc = reply.get_return_code()
        assert rc == ReturnCode.OK, f"Expected OK, got {rc}"

    def test_log_info_called_on_update(self):
        """Handler must log the membership change at INFO level."""
        ctl = _make_handler_ctl(trainers=["site-1", "site-2", "site-3"])
        req = _make_update_request(["site-1", "site-2"])
        fl_ctx = MagicMock()

        ctl._handle_membership_update("topic", req, fl_ctx)

        ctl.log_info.assert_called()

    def test_empty_payload_logs_warning_and_returns_ok(self):
        """Empty membership update must log a warning and return OK (no crash)."""
        ctl = _make_handler_ctl()
        req = Shareable()  # no CLIENTS key
        fl_ctx = MagicMock()

        reply = ctl._handle_membership_update("topic", req, fl_ctx)

        ctl.log_warning.assert_called()
        assert reply.get_return_code() == ReturnCode.OK

    def test_no_change_when_all_clients_survive(self):
        """When no clients are pruned, trainers and aggrs must be unchanged."""
        original_trainers = ["site-1", "site-2", "site-3"]
        original_aggrs = ["site-1", "site-2", "site-3"]
        ctl = _make_handler_ctl(trainers=list(original_trainers), aggrs=list(original_aggrs))
        # same clients as original — no pruning
        req = _make_update_request(["site-1", "site-2", "site-3"])
        fl_ctx = MagicMock()

        ctl._handle_membership_update("topic", req, fl_ctx)

        assert set(ctl.trainers) == set(original_trainers)
        assert set(ctl.aggrs) == set(original_aggrs)


class TestMembershipUpdateTopicRegistration:
    """process_config() must register the membership-update handler with the correct topic."""

    def test_topic_registered_in_process_config(self):
        """process_config() must register _handle_membership_update for the wf-scoped topic."""
        ctl = SwarmClientController.__new__(SwarmClientController)
        ctl.logger = MagicMock()
        ctl.log_info = MagicMock()
        ctl.log_warning = MagicMock()
        ctl.log_error = MagicMock()
        ctl.me = "site-1"
        ctl.workflow_id = "wf-42"
        ctl.request_to_submit_learn_result_task_name = "req_submit"

        registered_topics = {}
        engine = MagicMock()
        engine.register_aux_message_handler.side_effect = lambda topic, message_handle_func: registered_topics.update(
            {topic: message_handle_func}
        )
        ctl.engine = engine

        def cfg(key, *default):
            return {
                Constant.CLIENTS: ["site-1", "site-2"],
                Constant.TRAIN_CLIENTS: ["site-1", "site-2"],
                Constant.AGGR_CLIENTS: ["site-1", "site-2"],
            }.get(key, default[0] if default else None)

        ctl.get_config_prop = cfg

        fl_ctx = MagicMock()
        ctl.process_config(fl_ctx)

        expected_topic = topic_for_membership_update("wf-42")
        assert expected_topic in registered_topics, (
            f"process_config() must register handler for topic={expected_topic!r}; "
            f"registered: {list(registered_topics.keys())}"
        )
        assert registered_topics[expected_topic] == ctl._handle_membership_update
