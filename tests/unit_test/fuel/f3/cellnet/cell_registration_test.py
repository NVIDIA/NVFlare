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

import logging
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.private.defs import CellChannel


def _cell():
    cell = Cell.__new__(Cell)  # no network: only registration is under test
    cell.logger = logging.getLogger(__name__)
    cell.core_cell = MagicMock()
    cell.register_blob_cb = MagicMock()
    return cell


@pytest.mark.parametrize("positional", [True, False], ids=["positional", "keyword"])
def test_a_stream_channel_callback_also_receives_streamed_requests_however_it_is_registered(positional):
    cell, cb = _cell(), MagicMock()
    if positional:
        cell.register_request_cb("task_worker", "task_worker.log", cb, session_topic="log")
    else:
        cell.register_request_cb(channel="task_worker", topic="task_worker.log", cb=cb, session_topic="log")

    cell.core_cell.register_request_cb.assert_called_once_with(
        "task_worker", "task_worker.log", cb, session_topic="log"
    )
    channel, topic = cell.register_blob_cb.call_args[0][:2]
    assert (channel, topic) == ("task_worker", "task_worker.log")


def test_an_excluded_channel_callback_is_framed_only():
    cell, cb = _cell(), MagicMock()
    cell.register_request_cb(CellChannel.RETURN_ONLY, "any", cb)
    cell.core_cell.register_request_cb.assert_called_once()
    cell.register_blob_cb.assert_not_called()
