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

import threading
from unittest.mock import MagicMock

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message


class _Reply:
    def get_header(self, key):
        if key == MessageHeaderKey.RETURN_CODE:
            return ReturnCode.OK
        return None


class _FakeCell:
    def __init__(self):
        self.last_request = None

    def send_request(self, channel, topic, target, request, timeout=None, optional=False):
        _ = channel
        _ = topic
        _ = target
        _ = timeout
        _ = optional
        self.last_request = request
        return _Reply()


def _make_pipe(fake_cell):
    pipe = CellPipe.__new__(CellPipe)
    pipe.pipe_lock = threading.Lock()
    pipe.closed = False
    pipe.channel = "test_channel"
    pipe.peer_fqcn = "peer_fqcn"
    pipe.cell = fake_cell
    pipe.logger = MagicMock()
    pipe.hb_seq = 1
    return pipe


def test_reply_sets_minimum_msg_root_ttl():
    fake_cell = _FakeCell()
    pipe = _make_pipe(fake_cell)
    msg = Message.new_reply(topic="train", data={}, req_msg_id="req-1", msg_id="msg-1")

    sent = pipe.send(msg, timeout=60.0)

    assert sent is True
    assert fake_cell.last_request.get_header(MessageHeaderKey.MSG_ROOT_ID) == "msg-1"
    assert fake_cell.last_request.get_header(MessageHeaderKey.MSG_ROOT_TTL) == 300.0


def test_reply_preserves_larger_timeout_as_msg_root_ttl():
    fake_cell = _FakeCell()
    pipe = _make_pipe(fake_cell)
    msg = Message.new_reply(topic="train", data={}, req_msg_id="req-1", msg_id="msg-2")

    sent = pipe.send(msg, timeout=900.0)

    assert sent is True
    assert fake_cell.last_request.get_header(MessageHeaderKey.MSG_ROOT_ID) == "msg-2"
    assert fake_cell.last_request.get_header(MessageHeaderKey.MSG_ROOT_TTL) == 900.0


def test_reply_with_none_timeout_sets_minimum_msg_root_ttl():
    fake_cell = _FakeCell()
    pipe = _make_pipe(fake_cell)
    msg = Message.new_reply(topic="train", data={}, req_msg_id="req-1", msg_id="msg-4")

    sent = pipe.send(msg, timeout=None)

    assert sent is True
    assert fake_cell.last_request.get_header(MessageHeaderKey.MSG_ROOT_ID) == "msg-4"
    assert fake_cell.last_request.get_header(MessageHeaderKey.MSG_ROOT_TTL) == 300.0


def test_request_does_not_set_msg_root_ttl():
    fake_cell = _FakeCell()
    pipe = _make_pipe(fake_cell)
    msg = Message.new_request(topic="train", data={}, msg_id="msg-3")

    sent = pipe.send(msg, timeout=60.0)

    assert sent is True
    assert fake_cell.last_request.get_header(MessageHeaderKey.MSG_ROOT_ID) == "msg-3"
    assert fake_cell.last_request.get_header(MessageHeaderKey.MSG_ROOT_TTL) is None
