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

"""Unit tests for LocalCellPipe.

Each test class creates its own (site_name, token) pair using a UUID suffix so
that the class-level _cells_info cache never collides between test classes.
CoreCell.ALL_CELLS is cleaned up in teardown to prevent FQCN conflicts when the
test suite is run multiple times in the same process.
"""

import threading
import time
import uuid

import pytest

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.utils.attributes_exportable import ExportMode
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.local_cell_pipe import LocalCellPipe, _make_active_fqcn, _make_passive_fqcn
from nvflare.fuel.utils.pipe.pipe import Message, Topic

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
_CONNECT_TIMEOUT = 3.0  # seconds to wait for cell-to-cell TCP connection


def _unique_token() -> str:
    """Return a unique token so that tests don't share cells."""
    return f"job-{uuid.uuid4().hex[:8]}"


def _remove_cells_from_all_cells(*fqcns):
    """Remove test cells from CoreCell.ALL_CELLS to allow re-use of FQCNs."""
    for fqcn in fqcns:
        CoreCell.ALL_CELLS.pop(fqcn, None)


def _make_test_message(topic: str = "data", data: str = "hello") -> Message:
    return Message.new_request(topic=topic, data=data)


# ──────────────────────────────────────────────────────────────────────────────
# FQCN helpers
# ──────────────────────────────────────────────────────────────────────────────
class TestFqcnHelpers:
    """Verify the FQCN helper functions."""

    def test_passive_fqcn_format(self):
        fqcn = _make_passive_fqcn("site-1", "job123")
        assert fqcn.startswith("server.")
        assert "site-1" in fqcn
        assert "job123" in fqcn

    def test_active_fqcn_is_child_of_passive(self):
        site, token = "site-1", "job123"
        passive = _make_passive_fqcn(site, token)
        active = _make_active_fqcn(site, token)
        # active FQCN must be a sub-path of passive FQCN
        assert active.startswith(passive + ".")

    def test_different_tokens_give_different_fqcns(self):
        assert _make_passive_fqcn("s", "t1") != _make_passive_fqcn("s", "t2")
        assert _make_active_fqcn("s", "t1") != _make_active_fqcn("s", "t2")


# ──────────────────────────────────────────────────────────────────────────────
# Placeholder / early-return guard
# ──────────────────────────────────────────────────────────────────────────────
class TestPlaceholderGuard:
    """LocalCellPipe must survive construction when args are SystemVarName placeholders."""

    def test_placeholder_site_name(self):
        pipe = LocalCellPipe(
            mode=Mode.PASSIVE,
            site_name="{SITE_NAME}",
            token="job1",
        )
        # Should be constructed without error; _ci is None
        assert pipe._ci is None

    def test_placeholder_token(self):
        pipe = LocalCellPipe(
            mode=Mode.PASSIVE,
            site_name="site-1",
            token="{JOB_ID}",
        )
        assert pipe._ci is None

    def test_no_placeholder_creates_ci(self):
        token = _unique_token()
        passive = LocalCellPipe(mode=Mode.PASSIVE, site_name="site-x", token=token)
        assert passive._ci is not None
        # Cleanup
        passive._ci.cell.stop()
        LocalCellPipe._cells_info.pop(("passive", "site-x", token), None)
        _remove_cells_from_all_cells(_make_passive_fqcn("site-x", token))


# ──────────────────────────────────────────────────────────────────────────────
# Cell creation and sharing
# ──────────────────────────────────────────────────────────────────────────────
class TestCellSharing:
    """Task pipe and metric pipe must share the same underlying Cell."""

    def setup_method(self):
        self.site = "site-share"
        self.token = _unique_token()

    def teardown_method(self):
        for mode in ("passive", "active"):
            ci = LocalCellPipe._cells_info.pop((mode, self.site, self.token), None)
            if ci:
                try:
                    ci.cell.stop()
                except Exception:
                    pass
        _remove_cells_from_all_cells(
            _make_passive_fqcn(self.site, self.token),
            _make_active_fqcn(self.site, self.token),
        )

    def test_two_passive_pipes_share_cell(self):
        task_pipe = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        metric_pipe = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        # Same _LocalCellInfo object → same Cell
        assert task_pipe._ci is metric_pipe._ci

    def test_passive_cell_info_cached(self):
        pipe1 = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        pipe2 = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        key = ("passive", self.site, self.token)
        assert LocalCellPipe._cells_info[key] is pipe1._ci
        assert LocalCellPipe._cells_info[key] is pipe2._ci


# ──────────────────────────────────────────────────────────────────────────────
# Internal listener (PASSIVE side)
# ──────────────────────────────────────────────────────────────────────────────
class TestPassiveListener:
    """PASSIVE cell must create an internal listener on a random local port."""

    def setup_method(self):
        self.site = "site-listener"
        self.token = _unique_token()
        self.passive = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)

    def teardown_method(self):
        try:
            self.passive.close()
        except Exception:
            pass
        for mode in ("passive",):
            ci = LocalCellPipe._cells_info.pop((mode, self.site, self.token), None)
            if ci:
                try:
                    ci.cell.stop()
                except Exception:
                    pass
        _remove_cells_from_all_cells(_make_passive_fqcn(self.site, self.token))

    def test_internal_listener_created_after_open(self):
        self.passive.open("task")
        url = self.passive._ci.cell.core_cell.get_internal_listener_url()
        assert url is not None
        assert url.startswith("tcp://")

    def test_internal_listener_url_includes_host(self):
        self.passive.open("task")
        url = self.passive._ci.cell.core_cell.get_internal_listener_url()
        # should be a reachable local address
        assert "127.0.0.1" in url or "localhost" in url or "0.0.0.0" in url

    def test_no_listener_before_open(self):
        # Before open() the cell is not started, so no listener yet
        url = self.passive._ci.cell.core_cell.get_internal_listener_url()
        assert url is None

    def test_export_peer_fails_before_open(self):
        with pytest.raises(RuntimeError, match="open()"):
            self.passive.export(ExportMode.PEER)

    def test_export_peer_after_open_returns_active_args(self):
        self.passive.open("task")
        cls_path, args = self.passive.export(ExportMode.PEER)
        assert "LocalCellPipe" in cls_path
        assert args["mode"].upper() == "ACTIVE"
        assert args["site_name"] == self.site
        assert args["token"] == self.token
        assert args["parent_url"].startswith("tcp://")

    def test_export_self_returns_passive_args(self):
        self.passive.open("task")
        cls_path, args = self.passive.export(ExportMode.SELF)
        assert args["mode"] == Mode.PASSIVE or str(args["mode"]).lower() == "passive"
        assert args["site_name"] == self.site
        assert args["token"] == self.token


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end message passing (PASSIVE ↔ ACTIVE)
# ──────────────────────────────────────────────────────────────────────────────
class TestEndToEndMessaging:
    """Messages sent from one side must be received by the other."""

    def setup_method(self):
        self.site = "site-e2e"
        self.token = _unique_token()

        # PASSIVE side
        self.passive = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        self.passive.open("task")

        # Retrieve the parent_url from PASSIVE export
        _, passive_args = self.passive.export(ExportMode.PEER)
        parent_url = passive_args["parent_url"]

        # ACTIVE side
        self.active = LocalCellPipe(
            mode=Mode.ACTIVE,
            site_name=self.site,
            token=self.token,
            parent_url=parent_url,
        )
        self.active.open("task")

        # Give the TCP connection time to establish
        time.sleep(_CONNECT_TIMEOUT)

    def teardown_method(self):
        for pipe in (self.passive, self.active):
            try:
                pipe.close()
            except Exception:
                pass
        for mode in ("passive", "active"):
            ci = LocalCellPipe._cells_info.pop((mode, self.site, self.token), None)
            if ci:
                try:
                    ci.cell.stop()
                except Exception:
                    pass
        _remove_cells_from_all_cells(
            _make_passive_fqcn(self.site, self.token),
            _make_active_fqcn(self.site, self.token),
        )

    # ── positive tests ────────────────────────────────────────────────────────

    def test_active_to_passive_message(self):
        msg = Message.new_request(topic="train", data=b"model_data")
        sent = self.active.send(msg, timeout=5.0)
        assert sent, "send() returned False"
        received = self.passive.receive(timeout=5.0)
        assert received is not None
        assert received.data == b"model_data"
        assert received.topic == "train"

    def test_passive_to_active_message(self):
        msg = Message.new_request(topic="result", data={"loss": 0.42})
        sent = self.passive.send(msg, timeout=5.0)
        assert sent, "send() returned False"
        received = self.active.receive(timeout=5.0)
        assert received is not None
        assert received.data == {"loss": 0.42}

    def test_multiple_messages_in_order(self):
        payloads = [f"payload-{i}" for i in range(5)]
        for p in payloads:
            msg = Message.new_request(topic="data", data=p)
            assert self.active.send(msg, timeout=5.0)

        received = []
        for _ in payloads:
            m = self.passive.receive(timeout=5.0)
            assert m is not None
            received.append(m.data)
        assert received == payloads

    def test_heartbeat_does_not_block(self):
        """Heartbeat messages are fire-and-forget and must not raise."""
        # Heartbeat is a REQUEST with the special HEARTBEAT topic
        hb = Message.new_request(topic=Topic.HEARTBEAT, data=None)
        # fire_and_forget path, must not block or raise
        result = self.active.send(hb, timeout=None)
        assert result is True

    def test_receive_returns_none_when_empty(self):
        result = self.passive.receive(timeout=None)
        assert result is None

    def test_can_resend_is_true(self):
        assert self.passive.can_resend() is True
        assert self.active.can_resend() is True

    def test_clear_empties_queue(self):
        for _ in range(3):
            msg = Message.new_request(topic="x", data="d")
            self.active.send(msg, timeout=5.0)
        # give messages a moment to arrive
        time.sleep(0.5)
        self.passive.clear()
        assert self.passive.receive(timeout=None) is None


# ──────────────────────────────────────────────────────────────────────────────
# Two channels on the same cell (task + metric)
# ──────────────────────────────────────────────────────────────────────────────
class TestTwoChannels:
    """Task and metric pipes share one cell but use distinct channels."""

    def setup_method(self):
        self.site = "site-2ch"
        self.token = _unique_token()

        # PASSIVE side: task + metric pipes
        self.task_passive = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        self.metric_passive = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        self.task_passive.open("task")
        self.metric_passive.open("metric")

        assert self.task_passive._ci is self.metric_passive._ci, "must share same cell"

        # Export from the shared PASSIVE cell
        _, args = self.task_passive.export(ExportMode.PEER)
        parent_url = args["parent_url"]

        # ACTIVE side: task + metric pipes
        self.task_active = LocalCellPipe(mode=Mode.ACTIVE, site_name=self.site, token=self.token, parent_url=parent_url)
        self.metric_active = LocalCellPipe(
            mode=Mode.ACTIVE, site_name=self.site, token=self.token, parent_url=parent_url
        )
        self.task_active.open("task")
        self.metric_active.open("metric")

        assert self.task_active._ci is self.metric_active._ci, "must share same cell"

        time.sleep(_CONNECT_TIMEOUT)

    def teardown_method(self):
        for pipe in (self.task_passive, self.metric_passive, self.task_active, self.metric_active):
            try:
                pipe.close()
            except Exception:
                pass
        for mode in ("passive", "active"):
            ci = LocalCellPipe._cells_info.pop((mode, self.site, self.token), None)
            if ci:
                try:
                    ci.cell.stop()
                except Exception:
                    pass
        _remove_cells_from_all_cells(
            _make_passive_fqcn(self.site, self.token),
            _make_active_fqcn(self.site, self.token),
        )

    def test_task_channel_delivers_to_task_passive(self):
        msg = Message.new_request(topic="train", data="task-payload")
        assert self.task_active.send(msg, timeout=5.0)
        received = self.task_passive.receive(timeout=5.0)
        assert received is not None and received.data == "task-payload"
        # metric pipe queue should be empty
        assert self.metric_passive.receive(timeout=None) is None

    def test_metric_channel_delivers_to_metric_passive(self):
        msg = Message.new_request(topic="metric", data="metric-payload")
        assert self.metric_active.send(msg, timeout=5.0)
        received = self.metric_passive.receive(timeout=5.0)
        assert received is not None and received.data == "metric-payload"
        # task pipe queue should be empty
        assert self.task_passive.receive(timeout=None) is None

    def test_both_channels_simultaneously(self):
        task_msg = Message.new_request(topic="train", data="task")
        metric_msg = Message.new_request(topic="metric", data="metric")
        assert self.task_active.send(task_msg, timeout=5.0)
        assert self.metric_active.send(metric_msg, timeout=5.0)

        t_recv = self.task_passive.receive(timeout=5.0)
        m_recv = self.metric_passive.receive(timeout=5.0)
        assert t_recv is not None and t_recv.data == "task"
        assert m_recv is not None and m_recv.data == "metric"


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle / error cases
# ──────────────────────────────────────────────────────────────────────────────
class TestLifecycleAndErrors:
    """Error and edge-case behaviour."""

    def setup_method(self):
        self.site = "site-lc"
        self.token = _unique_token()

    def teardown_method(self):
        for mode in ("passive", "active"):
            ci = LocalCellPipe._cells_info.pop((mode, self.site, self.token), None)
            if ci:
                try:
                    ci.cell.stop()
                except Exception:
                    pass
        _remove_cells_from_all_cells(
            _make_passive_fqcn(self.site, self.token),
            _make_active_fqcn(self.site, self.token),
        )

    def test_send_after_close_raises(self):
        passive = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        passive.open("task")
        passive.close()
        msg = Message.new_request(topic="x", data="y")
        with pytest.raises(BrokenPipeError):
            passive.send(msg)

    def test_open_after_close_raises(self):
        passive = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        passive.open("task")
        passive.close()
        with pytest.raises(BrokenPipeError):
            passive.open("task2")

    def test_double_close_is_idempotent(self):
        passive = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        passive.open("task")
        passive.close()
        # second close must not raise
        passive.close()

    def test_active_without_parent_url_raises(self):
        with pytest.raises(ValueError, match="parent_url"):
            LocalCellPipe(
                mode=Mode.ACTIVE,
                site_name=self.site,
                token=self.token,
                parent_url="",
            )

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            LocalCellPipe(
                mode="sideways",  # type: ignore[arg-type]
                site_name=self.site,
                token=self.token,
            )

    def test_export_peer_before_open_raises(self):
        passive = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        with pytest.raises(RuntimeError):
            passive.export(ExportMode.PEER)
        passive._ci.cell.stop()
        LocalCellPipe._cells_info.pop(("passive", self.site, self.token), None)
        _remove_cells_from_all_cells(_make_passive_fqcn(self.site, self.token))

    def test_reference_counting_closes_cell_only_when_all_pipes_closed(self):
        """Cell must stay alive while at least one pipe is open."""
        token = _unique_token()
        site = "site-refcount"
        pipe1 = LocalCellPipe(mode=Mode.PASSIVE, site_name=site, token=token)
        pipe2 = LocalCellPipe(mode=Mode.PASSIVE, site_name=site, token=token)
        pipe1.open("task")
        pipe2.open("metric")

        # Closing pipe1 must NOT stop the cell (pipe2 still alive)
        pipe1.close()
        assert pipe2._ci.cell.core_cell.running, "Cell stopped too early"

        # Closing pipe2 stops the cell
        pipe2.close()

        LocalCellPipe._cells_info.pop(("passive", site, token), None)
        _remove_cells_from_all_cells(_make_passive_fqcn(site, token))

    def test_get_last_peer_active_time_returns_zero(self):
        passive = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        passive.open("task")
        assert passive.get_last_peer_active_time() == 0.0
        passive.close()


# ──────────────────────────────────────────────────────────────────────────────
# Concurrency
# ──────────────────────────────────────────────────────────────────────────────
class TestConcurrentMessages:
    """Multiple threads can send concurrently without data corruption."""

    def setup_method(self):
        self.site = "site-concur"
        self.token = _unique_token()

        self.passive = LocalCellPipe(mode=Mode.PASSIVE, site_name=self.site, token=self.token)
        self.passive.open("task")
        _, args = self.passive.export(ExportMode.PEER)

        self.active = LocalCellPipe(
            mode=Mode.ACTIVE,
            site_name=self.site,
            token=self.token,
            parent_url=args["parent_url"],
        )
        self.active.open("task")
        time.sleep(_CONNECT_TIMEOUT)

    def teardown_method(self):
        for pipe in (self.passive, self.active):
            try:
                pipe.close()
            except Exception:
                pass
        for mode in ("passive", "active"):
            ci = LocalCellPipe._cells_info.pop((mode, self.site, self.token), None)
            if ci:
                try:
                    ci.cell.stop()
                except Exception:
                    pass
        _remove_cells_from_all_cells(
            _make_passive_fqcn(self.site, self.token),
            _make_active_fqcn(self.site, self.token),
        )

    def test_concurrent_sends_all_received(self):
        n_threads = 5
        n_msgs_per_thread = 4
        total = n_threads * n_msgs_per_thread
        errors = []

        def sender(thread_id: int):
            for i in range(n_msgs_per_thread):
                msg = Message.new_request(topic="data", data=f"t{thread_id}-m{i}")
                if not self.active.send(msg, timeout=10.0):
                    errors.append(f"thread {thread_id} msg {i} send failed")

        threads = [threading.Thread(target=sender, args=(tid,)) for tid in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert not errors, f"send errors: {errors}"

        received = []
        deadline = time.time() + 10.0
        while len(received) < total and time.time() < deadline:
            m = self.passive.receive(timeout=0.5)
            if m is not None:
                received.append(m.data)

        assert len(received) == total, f"expected {total} messages, got {len(received)}"
