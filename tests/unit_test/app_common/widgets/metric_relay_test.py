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

from unittest.mock import MagicMock, patch

from nvflare.apis.dxo import DXO
from nvflare.apis.event_type import EventType
from nvflare.app_common.widgets.metric_relay import MetricRelay
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.pipe import Message, Pipe, Topic


class _FakePipe(Pipe):
    """Minimal no-op Pipe stub that passes isinstance checks."""

    def __init__(self):
        super().__init__(mode=Mode.ACTIVE)

    def open(self, name):
        pass

    def close(self):
        pass

    def send(self, msg, timeout=None):
        return True

    def receive(self, timeout=None):
        return None

    def get_last_peer_active_time(self):
        return 0.0

    def clear(self):
        pass

    def can_resend(self):
        return False


def _make_fl_ctx(pipe):
    engine = MagicMock()
    engine.get_component.return_value = pipe
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    return fl_ctx


def _start_run(relay, pipe=None):
    """Fire ABOUT_TO_START_RUN so the relay has an open pipe."""
    if pipe is None:
        pipe = _FakePipe()
    fl_ctx = _make_fl_ctx(pipe)
    relay.handle_event(EventType.ABOUT_TO_START_RUN, fl_ctx)
    return fl_ctx


class TestMetricRelayHandlerLifecycle:
    """Handler recreation and close_pipe=False fixes."""

    def test_before_task_execution_creates_fresh_handler_each_round(self):
        """Each BEFORE_TASK_EXECUTION must produce a new PipeHandler, not reuse a stopped one."""
        relay = MetricRelay(pipe_id="pipe")
        fl_ctx = _start_run(relay)

        h1 = MagicMock()
        h2 = MagicMock()
        with patch("nvflare.app_common.widgets.metric_relay.PipeHandler", side_effect=[h1, h2]):
            relay.handle_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)
            first_handler = relay.pipe_handler

            relay.handle_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)
            second_handler = relay.pipe_handler

        assert first_handler is not second_handler
        assert first_handler is h1
        assert second_handler is h2

    def test_before_task_execution_starts_handler(self):
        """BEFORE_TASK_EXECUTION must call start() on the new handler."""
        relay = MetricRelay(pipe_id="pipe")
        fl_ctx = _start_run(relay)

        mock_handler = MagicMock()
        with patch("nvflare.app_common.widgets.metric_relay.PipeHandler", return_value=mock_handler):
            relay.handle_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)

        mock_handler.start.assert_called_once()

    def test_second_before_task_execution_stops_old_handler_without_closing_pipe(self):
        """On the second BEFORE_TASK_EXECUTION, the old handler is stopped with close_pipe=False."""
        relay = MetricRelay(pipe_id="pipe")
        fl_ctx = _start_run(relay)

        h1 = MagicMock()
        h2 = MagicMock()
        with patch("nvflare.app_common.widgets.metric_relay.PipeHandler", side_effect=[h1, h2]):
            relay.handle_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)
            relay.handle_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)

        h1.stop.assert_called_once_with(close_pipe=False)


class TestMetricRelayStatusCallback:
    """Bound-closure status callback correctness."""

    def _get_status_cb(self, relay, fl_ctx):
        """Fire BEFORE_TASK_EXECUTION and return the registered status callback."""
        mock_handler = MagicMock()
        with patch("nvflare.app_common.widgets.metric_relay.PipeHandler", return_value=mock_handler):
            relay.handle_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)
        return mock_handler, mock_handler.set_status_cb.call_args[0][0]

    def test_status_cb_stops_handler_with_close_pipe_false(self):
        """Status callback must call stop(close_pipe=False), never close the pipe."""
        relay = MetricRelay(pipe_id="pipe")
        fl_ctx = _start_run(relay)
        mock_handler, status_cb = self._get_status_cb(relay, fl_ctx)

        msg = Message.new_request(Topic.PEER_GONE, "heartbeat timeout")
        status_cb(msg)

        mock_handler.stop.assert_called_once_with(close_pipe=False)

    def test_stale_status_cb_does_not_stop_new_handler(self):
        """A status callback from a previous handler must be ignored after handler replacement."""
        relay = MetricRelay(pipe_id="pipe")
        fl_ctx = _start_run(relay)

        h1 = MagicMock()
        h2 = MagicMock()
        with patch("nvflare.app_common.widgets.metric_relay.PipeHandler", side_effect=[h1, h2]):
            relay.handle_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)
            stale_cb = h1.set_status_cb.call_args[0][0]

            # Replace handler — h2 is now current
            relay.handle_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)

        # Fire the stale callback from h1
        msg = Message.new_request(Topic.PEER_GONE, "late arrival")
        stale_cb(msg)

        h2.stop.assert_not_called()

    def test_current_status_cb_does_stop_current_handler(self):
        """Status callback from the current handler must stop it."""
        relay = MetricRelay(pipe_id="pipe")
        fl_ctx = _start_run(relay)
        mock_handler, status_cb = self._get_status_cb(relay, fl_ctx)

        msg = Message.new_request(Topic.PEER_GONE, "heartbeat timeout")
        status_cb(msg)

        mock_handler.stop.assert_called_once_with(close_pipe=False)


class TestMetricRelayMessageCallback:
    """Bad payload drop fix."""

    def test_bad_payload_dropped_send_not_called(self):
        """_pipe_msg_cb must drop non-DXO data without calling send_analytic_dxo."""
        relay = MetricRelay(pipe_id="pipe")
        relay._fl_ctx = MagicMock()

        msg = Message.new_request("metric", data="not_a_dxo")
        with patch("nvflare.app_common.widgets.metric_relay.send_analytic_dxo") as mock_send:
            relay._pipe_msg_cb(msg)

        mock_send.assert_not_called()

    def test_bad_payload_does_not_raise(self):
        """_pipe_msg_cb must not propagate an exception on bad payload."""
        relay = MetricRelay(pipe_id="pipe")
        relay._fl_ctx = MagicMock()

        msg = Message.new_request("metric", data={"not": "a dxo"})
        with patch("nvflare.app_common.widgets.metric_relay.send_analytic_dxo"):
            relay._pipe_msg_cb(msg)  # must not raise

    def test_valid_dxo_payload_forwarded(self):
        """_pipe_msg_cb must forward a valid DXO to send_analytic_dxo."""
        relay = MetricRelay(pipe_id="pipe")
        relay._fl_ctx = MagicMock()

        dxo = MagicMock(spec=DXO)
        msg = Message.new_request("metric", data=dxo)
        with patch("nvflare.app_common.widgets.metric_relay.send_analytic_dxo") as mock_send:
            relay._pipe_msg_cb(msg)

        mock_send.assert_called_once_with(relay, dxo, relay._fl_ctx, relay._event_type, fire_fed_event=relay._fed_event)
