# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.drivers.driver_params import DriverCap
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.sfm.constants import Types
from nvflare.fuel.f3.sfm.heartbeat_monitor import HeartbeatMonitor
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection


class FakeDriver:
    def __init__(self, send_heartbeat_supported=True):
        self.send_heartbeat_supported = send_heartbeat_supported

    def capabilities(self):
        return {DriverCap.SEND_HEARTBEAT.value: self.send_heartbeat_supported}


class FakeConnObj:
    def __init__(self, send_heartbeat_supported=True):
        self.connector = SimpleNamespace(driver=FakeDriver(send_heartbeat_supported))
        self.close = MagicMock()

    def __str__(self):
        return "FakeConn"


class FakeSfmConn:
    def __init__(self, stall_sec, last_activity, send_heartbeat_supported=True):
        self._stall_sec = stall_sec
        self.last_activity = last_activity
        self.conn = FakeConnObj(send_heartbeat_supported=send_heartbeat_supported)
        self.send_heartbeat = MagicMock()

    def get_send_stall_seconds(self):
        return self._stall_sec


class MutableFakeSfmConn(FakeSfmConn):
    def set_stall(self, stall_sec):
        self._stall_sec = stall_sec


class DummySendConn:
    def __init__(self, should_raise=False):
        self.name = "dummy"
        self.should_raise = should_raise
        self.send_calls = 0

    def send_frame(self, _frame):
        self.send_calls += 1
        if self.should_raise:
            raise RuntimeError("send failed")


class BlockingSendConn:
    """Connection whose first send blocks until released/closed."""

    def __init__(self):
        self.name = "blocking"
        self.connector = SimpleNamespace(driver=FakeDriver(True))
        self.send_calls = 0
        self.first_send_entered = threading.Event()
        self.release_send = threading.Event()
        self.close = MagicMock(side_effect=self._on_close)

    def _on_close(self):
        # Simulate close unblocking a blocked send call.
        self.release_send.set()

    def send_frame(self, _frame):
        self.send_calls += 1
        if self.send_calls == 1:
            self.first_send_entered.set()
            self.release_send.wait(timeout=2.0)


class TestSfmStallMonitor:
    def _make_monitor(self, monkeypatch, *, hb_interval, stall_timeout, close_enabled, conns, consecutive_checks=1):
        monkeypatch.setattr(CommConfigurator, "get_heartbeat_interval", lambda self, default: hb_interval)
        monkeypatch.setattr(CommConfigurator, "get_sfm_send_stall_timeout", lambda self, default: stall_timeout)
        monkeypatch.setattr(
            CommConfigurator, "get_sfm_close_stalled_connection", lambda self, default=False: close_enabled
        )
        monkeypatch.setattr(
            CommConfigurator, "get_sfm_send_stall_consecutive_checks", lambda self, default=3: consecutive_checks
        )
        return HeartbeatMonitor(conns)

    def test_monitor_closes_stalled_connection_when_enabled(self, monkeypatch):
        sfm_conn = FakeSfmConn(stall_sec=20.0, last_activity=0.0)
        monitor = self._make_monitor(
            monkeypatch,
            hb_interval=60,
            stall_timeout=10.0,
            close_enabled=True,
            conns={"c1": sfm_conn},
        )

        monitor.curr_time = 100.0
        monitor._check_heartbeat()

        sfm_conn.conn.close.assert_called_once()
        sfm_conn.send_heartbeat.assert_not_called()

    def test_monitor_closes_only_after_consecutive_guard_threshold(self, monkeypatch):
        sfm_conn = FakeSfmConn(stall_sec=20.0, last_activity=0.0)
        monitor = self._make_monitor(
            monkeypatch,
            hb_interval=60,
            stall_timeout=10.0,
            close_enabled=True,
            conns={"c1": sfm_conn},
            consecutive_checks=3,
        )

        monitor.curr_time = 100.0
        monitor._check_heartbeat()
        monitor._check_heartbeat()
        assert sfm_conn.conn.close.call_count == 0

        monitor._check_heartbeat()
        assert sfm_conn.conn.close.call_count == 1

    def test_monitor_logs_warning_on_stall_detection(self, monkeypatch, caplog):
        sfm_conn = FakeSfmConn(stall_sec=20.0, last_activity=0.0)
        monitor = self._make_monitor(
            monkeypatch,
            hb_interval=60,
            stall_timeout=10.0,
            close_enabled=False,
            conns={"c1": sfm_conn},
            consecutive_checks=3,
        )

        caplog.set_level("WARNING")
        monitor.curr_time = 100.0
        monitor._check_heartbeat()

        assert any("Detected stalled send" in rec.message for rec in caplog.records)

    def test_monitor_warn_only_when_close_disabled(self, monkeypatch):
        sfm_conn = FakeSfmConn(stall_sec=20.0, last_activity=0.0)
        monitor = self._make_monitor(
            monkeypatch,
            hb_interval=60,
            stall_timeout=10.0,
            close_enabled=False,
            conns={"c1": sfm_conn},
        )

        monitor.curr_time = 100.0
        monitor._check_heartbeat()

        sfm_conn.conn.close.assert_not_called()
        sfm_conn.send_heartbeat.assert_not_called()

    def test_monitor_sends_ping_when_idle_and_not_stalled(self, monkeypatch):
        sfm_conn = FakeSfmConn(stall_sec=0.0, last_activity=0.0, send_heartbeat_supported=True)
        monitor = self._make_monitor(
            monkeypatch,
            hb_interval=10,
            stall_timeout=30.0,
            close_enabled=False,
            conns={"c1": sfm_conn},
        )

        monitor.curr_time = 100.0
        monitor._check_heartbeat()

        sfm_conn.send_heartbeat.assert_called_once_with(Types.PING)
        sfm_conn.conn.close.assert_not_called()

    def test_monitor_skips_ping_when_driver_disables_heartbeat(self, monkeypatch):
        sfm_conn = FakeSfmConn(stall_sec=0.0, last_activity=0.0, send_heartbeat_supported=False)
        monitor = self._make_monitor(
            monkeypatch,
            hb_interval=10,
            stall_timeout=30.0,
            close_enabled=False,
            conns={"c1": sfm_conn},
        )

        monitor.curr_time = 100.0
        monitor._check_heartbeat()

        sfm_conn.send_heartbeat.assert_not_called()

    def test_monitor_has_no_stall_warning_in_normal_non_stalled_flow(self, monkeypatch, caplog):
        sfm_conn = FakeSfmConn(stall_sec=0.0, last_activity=0.0)
        monitor = self._make_monitor(
            monkeypatch,
            hb_interval=10,
            stall_timeout=30.0,
            close_enabled=True,
            conns={"c1": sfm_conn},
            consecutive_checks=3,
        )

        caplog.set_level("WARNING")
        monitor.curr_time = 100.0
        monitor._check_heartbeat()

        assert not any("Detected stalled send" in rec.message for rec in caplog.records)
        assert sfm_conn.conn.close.call_count == 0

    def test_intermittent_stall_resets_counter_and_suppresses_false_close(self, monkeypatch):
        sfm_conn = MutableFakeSfmConn(stall_sec=20.0, last_activity=0.0)
        monitor = self._make_monitor(
            monkeypatch,
            hb_interval=60,
            stall_timeout=10.0,
            close_enabled=True,
            conns={"c1": sfm_conn},
            consecutive_checks=3,
        )

        monitor.curr_time = 100.0
        monitor._check_heartbeat()  # stall count 1
        sfm_conn.set_stall(0.0)
        monitor._check_heartbeat()  # reset to 0
        sfm_conn.set_stall(20.0)
        monitor._check_heartbeat()  # stall count 1 again
        monitor._check_heartbeat()  # stall count 2

        # Should not close yet because we did not hit 3 consecutive stalled checks after reset.
        assert sfm_conn.conn.close.call_count == 0


class TestSfmConnectionSendState:
    def test_send_state_resets_after_successful_send(self):
        conn = DummySendConn(should_raise=False)
        sfm_conn = SfmConnection(conn=conn, local_endpoint=Endpoint("local"))

        sfm_conn.send_dict(Types.PING, 1, {"k": "v"})

        assert conn.send_calls == 1
        assert sfm_conn.get_send_stall_seconds() == 0.0

    def test_send_state_resets_after_send_exception(self):
        conn = DummySendConn(should_raise=True)
        sfm_conn = SfmConnection(conn=conn, local_endpoint=Endpoint("local"))

        with pytest.raises(RuntimeError, match="send failed"):
            sfm_conn.send_dict(Types.PING, 1, {"k": "v"})

        # Ensure finally block clears tracking state even on failure.
        assert sfm_conn.get_send_stall_seconds() == 0.0


class TestHolBlockingBehavior:
    def test_without_mitigation_send_lock_can_cause_hol_blocking(self):
        """Baseline behavior: blocked send on one thread blocks a second sender on same connection lock."""
        conn = BlockingSendConn()
        sfm_conn = SfmConnection(conn=conn, local_endpoint=Endpoint("local"))

        errors = []

        def do_send():
            try:
                sfm_conn.send_dict(Types.PING, 1, {"k": "v"})
            except Exception as ex:
                errors.append(ex)

        t1 = threading.Thread(target=do_send, daemon=True)
        t1.start()
        assert conn.first_send_entered.wait(timeout=0.5)

        t2 = threading.Thread(target=do_send, daemon=True)
        t2.start()

        # While the first send is blocked, the second cannot enter send_frame due to SfmConnection.lock.
        time.sleep(0.05)
        assert conn.send_calls == 1

        conn.release_send.set()
        t1.join(timeout=0.5)
        t2.join(timeout=0.5)
        assert not t1.is_alive()
        assert not t2.is_alive()
        assert conn.send_calls == 2
        assert errors == []


class TestFix3OnlyRecovery:
    def test_fix3_close_enabled_unblocks_stalled_send(self, monkeypatch):
        """Fix #3 only: monitor closes stalled connection and blocked send returns."""
        conn = BlockingSendConn()
        sfm_conn = SfmConnection(conn=conn, local_endpoint=Endpoint("local"))

        def do_send():
            sfm_conn.send_dict(Types.PING, 1, {"k": "v"})

        t = threading.Thread(target=do_send, daemon=True)
        t.start()
        assert conn.first_send_entered.wait(timeout=0.5)

        # Give send state a moment to register non-zero stall time.
        time.sleep(0.03)

        monkeypatch.setattr(CommConfigurator, "get_heartbeat_interval", lambda self, default: 60)
        monkeypatch.setattr(CommConfigurator, "get_sfm_send_stall_timeout", lambda self, default: 0.01)
        monkeypatch.setattr(CommConfigurator, "get_sfm_close_stalled_connection", lambda self, default=False: True)
        monkeypatch.setattr(CommConfigurator, "get_sfm_send_stall_consecutive_checks", lambda self, default=3: 1)

        monitor = HeartbeatMonitor({"c1": sfm_conn})
        monitor.curr_time = time.time()
        monitor._check_heartbeat()

        assert conn.close.called
        t.join(timeout=0.5)
        assert not t.is_alive()

    def test_fix3_close_disabled_keeps_stalled_send_blocked(self, monkeypatch):
        """Negative case for #3: warn-only mode does not close/unblock the stalled send."""
        conn = BlockingSendConn()
        sfm_conn = SfmConnection(conn=conn, local_endpoint=Endpoint("local"))

        def do_send():
            sfm_conn.send_dict(Types.PING, 1, {"k": "v"})

        t = threading.Thread(target=do_send, daemon=True)
        t.start()
        assert conn.first_send_entered.wait(timeout=0.5)
        time.sleep(0.03)

        monkeypatch.setattr(CommConfigurator, "get_heartbeat_interval", lambda self, default: 60)
        monkeypatch.setattr(CommConfigurator, "get_sfm_send_stall_timeout", lambda self, default: 0.01)
        monkeypatch.setattr(CommConfigurator, "get_sfm_close_stalled_connection", lambda self, default=False: False)
        monkeypatch.setattr(CommConfigurator, "get_sfm_send_stall_consecutive_checks", lambda self, default=3: 1)

        monitor = HeartbeatMonitor({"c1": sfm_conn})
        monitor.curr_time = time.time()
        monitor._check_heartbeat()

        assert not conn.close.called
        assert t.is_alive()

        conn.release_send.set()
        t.join(timeout=0.5)
        assert not t.is_alive()
