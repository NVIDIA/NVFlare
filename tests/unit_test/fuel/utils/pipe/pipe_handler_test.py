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

import time

from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.pipe import Pipe, Topic
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler


class _BrokenPipe(Pipe):
    """Minimal Pipe stub whose receive() always raises BrokenPipeError."""

    def __init__(self, error_msg):
        super().__init__(mode=Mode.ACTIVE)
        self._error_msg = error_msg

    def open(self, name):
        pass

    def close(self):
        pass

    def send(self, msg, timeout=None):
        return True

    def receive(self, timeout=None):
        raise BrokenPipeError(self._error_msg)

    def get_last_peer_active_time(self):
        return 0.0

    def clear(self):
        pass

    def can_resend(self) -> bool:
        return False


class TestPipeHandlerBrokenPipe:
    """PipeHandler._try_read must emit PEER_GONE and stop when receive() raises BrokenPipeError."""

    def _make_handler(self, pipe):
        return PipeHandler(
            pipe=pipe,
            read_interval=0.01,
            heartbeat_interval=5.0,
            heartbeat_timeout=30.0,
        )

    def _drain_messages(self, handler, timeout=1.0):
        messages = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = handler.get_next()
            if msg:
                messages.append(msg)
            time.sleep(0.01)
        return messages

    def test_pipe_closed_emits_peer_gone_and_stops(self):
        """When receive() raises BrokenPipeError('pipe is not open'), PEER_GONE is emitted and the reader stops."""
        handler = self._make_handler(_BrokenPipe("pipe is not open"))
        handler.start()

        messages = self._drain_messages(handler, timeout=1.0)
        handler.stop()

        assert any(m.topic == Topic.PEER_GONE for m in messages)
        assert handler.reader is None

    def test_pipe_dir_gone_emits_peer_gone_and_stops(self):
        """When receive() raises BrokenPipeError('error reading from ...'), PEER_GONE is emitted and the reader stops."""
        handler = self._make_handler(_BrokenPipe("error reading from /some/dir"))
        handler.start()

        messages = self._drain_messages(handler, timeout=1.0)
        handler.stop()

        assert any(m.topic == Topic.PEER_GONE for m in messages)
        assert handler.reader is None

    def test_heartbeat_passes_600s_timeout(self):
        """_heartbeat() must call send_to_peer with timeout=600.0 so FilePipe keeps the file alive."""
        from unittest.mock import patch

        pipe = _BrokenPipe("pipe is not open")
        handler = PipeHandler(
            pipe=pipe,
            read_interval=0.01,
            heartbeat_interval=0.05,  # fast so we capture quickly
            heartbeat_timeout=30.0,
        )

        captured = []
        original = handler.send_to_peer

        def capturing_send(msg, timeout=None, abort_signal=None):
            if msg.topic == Topic.HEARTBEAT:
                captured.append(timeout)
            return True

        with patch.object(handler, "send_to_peer", side_effect=capturing_send):
            handler.start()
            deadline = time.time() + 0.5
            while not captured and time.time() < deadline:
                time.sleep(0.01)
            handler.stop()

        assert captured, "no heartbeat was sent"
        assert all(t == 600.0 for t in captured), f"expected timeout=600.0, got {captured}"

    def test_graceful_stop_does_not_emit_peer_gone(self):
        """BrokenPipeError raised after stop() is called must not emit PEER_GONE."""
        pipe = _BrokenPipe("pipe is not open")
        handler = self._make_handler(pipe)

        received = []
        handler.set_status_cb(lambda msg: received.append(msg))

        handler.start()
        # Stop immediately — asked_to_stop=True before _try_read can emit anything.
        handler.stop()

        # Give the reader thread time to finish.
        deadline = time.time() + 1.0
        while handler.reader is not None and time.time() < deadline:
            time.sleep(0.01)

        assert not any(m.topic == Topic.PEER_GONE for m in received)
