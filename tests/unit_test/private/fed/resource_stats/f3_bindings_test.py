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

from nvflare.fuel.f3.message import Message
from nvflare.private.fed.resource_stats.f3_bindings import attach_f3_context
from nvflare.private.fed.resource_stats.f3_counter import F3Counter, F3TrafficClass


def test_attach_f3_context_is_process_local_and_preserves_wire_headers():
    message = Message(headers={"existing": "value"}, payload=b"payload")
    counter = F3Counter()

    assert attach_f3_context(message, counter, F3TrafficClass.TASK_RESULT, recipient_name="server")

    context = message.get_logical_send_context()
    assert context is not None
    assert context._accounting is counter
    assert context._traffic_class is F3TrafficClass.TASK_RESULT
    assert context._recipient_name == "server"
    assert message.headers == {"existing": "value"}


def test_attach_failure_marks_counter_gap_without_raising_into_send_path():
    class RejectingMessage:
        def set_logical_send_context(self, _context):
            raise RuntimeError("instrumentation unavailable")

    counter = F3Counter()

    assert not attach_f3_context(RejectingMessage(), counter, F3TrafficClass.TASK_RESULT, recipient_name="server")
    assert counter.snapshot() == {
        "status": "partial",
        "issues": ["counter_gap"],
        "sent_to": [],
    }


def test_attach_failure_still_does_not_raise_when_gap_marker_is_broken():
    class RejectingMessage:
        def set_logical_send_context(self, _context):
            raise RuntimeError("instrumentation unavailable")

    class BrokenAccounting:
        def mark_counter_gap(self):
            raise RuntimeError("counter unavailable")

    assert not attach_f3_context(
        RejectingMessage(), BrokenAccounting(), F3TrafficClass.TASK_RESULT, recipient_name="server"
    )
