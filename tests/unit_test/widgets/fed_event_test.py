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

from nvflare.apis.fl_constant import EventScope, FedEventHeader, FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.widgets.fed_event import FedEventRunner


def _make_incoming_event(timestamp, event_id=None, peer_name="site-1"):
    event = Shareable()
    event.set_peer_props({ReservedKey.IDENTITY_NAME: peer_name})
    event.set_header(FedEventHeader.TIMESTAMP, timestamp)
    event.set_header(FedEventHeader.EVENT_TYPE, "fed.test")
    if event_id is not None:
        event.set_header(FedEventHeader.EVENT_ID, event_id)
    return event


def _make_outgoing_context(event):
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.EVENT_SCOPE, EventScope.FEDERATION, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.EVENT_DATA, event, private=True, sticky=False)
    return fl_ctx


def _make_receiver():
    receiver = FedEventRunner()
    receiver.poster = MagicMock()  # prevent _receive from starting the posting thread
    return receiver


class TestFedEventRunner:
    def test_outgoing_events_have_unique_event_ids(self):
        runner = FedEventRunner()
        first_event = Shareable()
        second_event = Shareable()

        with patch.object(runner, "fire_and_forget_request"):
            runner.handle_event("fed.test", _make_outgoing_context(first_event))
            runner.handle_event("fed.test", _make_outgoing_context(second_event))

        first_event_id = first_event.get_header(FedEventHeader.EVENT_ID)
        second_event_id = second_event.get_header(FedEventHeader.EVENT_ID)
        assert first_event_id
        assert second_event_id
        assert first_event_id != second_event_id

    def test_distinct_events_with_equal_timestamps_are_queued(self):
        receiver = _make_receiver()

        first_reply = receiver._receive("fed.event", _make_incoming_event(1.0, "event-1"), FLContext())
        second_reply = receiver._receive("fed.event", _make_incoming_event(1.0, "event-2"), FLContext())

        assert first_reply.get_return_code() == ReturnCode.OK
        assert second_reply.get_return_code() == ReturnCode.OK
        assert len(receiver.in_events) == 2

    def test_rapid_metrics_from_five_sites_are_all_queued(self):
        receiver = _make_receiver()

        for site_num in range(5):
            for metric_num in range(8):
                event_id = f"site-{site_num}-metric-{metric_num}"
                event = _make_incoming_event(1.0, event_id, f"site-{site_num}")
                receiver._receive("fed.event", event, FLContext())

        assert len(receiver.in_events) == 40

    def test_distinct_out_of_order_events_are_queued(self):
        receiver = _make_receiver()

        receiver._receive("fed.event", _make_incoming_event(2.0, "event-2"), FLContext())
        receiver._receive("fed.event", _make_incoming_event(1.0, "event-1"), FLContext())

        assert len(receiver.in_events) == 2

    def test_duplicate_event_id_from_same_peer_is_not_queued_twice(self):
        receiver = _make_receiver()

        receiver._receive("fed.event", _make_incoming_event(1.0, "event-1"), FLContext())
        duplicate_reply = receiver._receive("fed.event", _make_incoming_event(2.0, "event-1"), FLContext())

        assert duplicate_reply.get_return_code() == ReturnCode.OK
        assert len(receiver.in_events) == 1

    def test_same_event_id_from_different_peers_is_queued(self):
        receiver = _make_receiver()

        receiver._receive("fed.event", _make_incoming_event(1.0, "event-1", "site-1"), FLContext())
        receiver._receive("fed.event", _make_incoming_event(1.0, "event-1", "site-2"), FLContext())

        assert len(receiver.in_events) == 2

    def test_invalid_event_ids_are_rejected(self):
        receiver = _make_receiver()

        with patch.object(receiver, "log_error") as log_error:
            for invalid_event_id in ("", 123):
                reply = receiver._receive("fed.event", _make_incoming_event(1.0, invalid_event_id), FLContext())

        assert log_error.call_count == 2
        assert reply.get_return_code() == ReturnCode.BAD_REQUEST_DATA
        assert receiver.in_events == []

    def test_event_id_cache_evicts_oldest_entry(self):
        receiver = _make_receiver()

        with patch("nvflare.widgets.fed_event.MAX_EVENT_ID_CACHE_SIZE", 2):
            for event_id in ("event-1", "event-2", "event-3"):
                receiver._receive("fed.event", _make_incoming_event(1.0, event_id), FLContext())

        assert list(receiver.received_event_ids["site-1"]) == ["event-2", "event-3"]

    def test_legacy_events_without_event_ids_are_not_timestamp_filtered(self):
        receiver = _make_receiver()

        receiver._receive("fed.event", _make_incoming_event(1.0), FLContext())
        receiver._receive("fed.event", _make_incoming_event(1.0), FLContext())

        assert len(receiver.in_events) == 2
