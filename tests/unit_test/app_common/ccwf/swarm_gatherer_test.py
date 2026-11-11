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
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.swarm_client_ctl import Gatherer
from tests.timing_utils import ManualClock, ObservedLock, ThreadGroup, wait_for


def _make_gatherer(aggregator=None, trainers=None, min_responses=2, timeout=0):
    trainers = trainers or ["site-1", "site-2"]
    if aggregator is None:
        aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)
        aggregator.handle_event(EventType.START_RUN, FLContext())
    gatherer = Gatherer(
        task_data=Shareable(),
        fl_ctx=FLContext(),
        for_round=0,
        executor=Mock(best_metric=None),
        aggregator=aggregator,
        metric_comparator=Mock(),
        all_clients=trainers,
        trainers=trainers,
        min_responses_required=min_responses,
        wait_time_after_min_resps_received=1,
        timeout=timeout,
    )
    gatherer.fire_event = Mock()
    gatherer.fire_event_with_data = Mock()
    gatherer.lock = ObservedLock()
    return gatherer


def _submit(gatherer, site, value):
    result = DXO(DataKind.WEIGHTS, {"weight": value}).to_shareable()
    result.set_header(AppConstants.CURRENT_ROUND, gatherer.for_round)
    result.add_cookie(AppConstants.CONTRIBUTION_ROUND, gatherer.for_round)
    result.set_peer_props({ReservedKey.IDENTITY_NAME: site})
    fl_ctx = FLContext()
    fl_ctx.put(ReservedKey.ENGINE, Mock(), private=True, sticky=False)
    reply = gatherer.gather(site, result, fl_ctx)
    assert reply.get_return_code() == ReturnCode.OK
    return reply


@pytest.mark.parametrize("completion", ["all", "minimum", "timeout", "aggregate"])
def test_completion_waits_for_inflight_acceptance(completion, monkeypatch):
    clock = ManualClock()
    monkeypatch.setattr("nvflare.app_common.ccwf.swarm_client_ctl.time", clock)
    trainers = ["site-1", "site-2"] if completion == "all" else ["site-1", "site-2", "site-3"]
    gatherer = _make_gatherer(
        trainers=trainers,
        min_responses=1 if completion == "minimum" else 2,
        timeout=5 if completion == "timeout" else 0,
    )
    assert _submit(gatherer, "site-1", 1.0).get_return_code() == ReturnCode.OK
    accepting = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    results = []
    accept = gatherer.aggregator.accept

    def blocked_accept(result, fl_ctx):
        accepting.set()
        release.wait()
        return accept(result, fl_ctx)

    def finish():
        try:
            if completion != "aggregate":
                assert gatherer.is_done()
            results.append(from_shareable(gatherer.aggregate()).data["weight"])
        finally:
            finished.set()

    with ThreadGroup(release) as threads, patch.object(gatherer.aggregator, "accept", side_effect=blocked_accept):
        threads.start(_submit, gatherer, "site-2", 2.0)
        assert accepting.wait(5)
        clock.advance(10)
        threads.start(finish)
        wait_for(lambda: gatherer.lock.blocked.is_set() or finished.is_set())
        assert gatherer.lock.blocked.is_set()
        assert not finished.is_set()
    assert finished.is_set()
    assert results == [1.5]


def test_stale_submission_cannot_contaminate_next_round():
    gatherer = _make_gatherer()
    assert _submit(gatherer, "site-1", 1.0).get_return_code() == ReturnCode.OK
    assert _submit(gatherer, "site-2", 2.0).get_return_code() == ReturnCode.OK
    aggregating = threading.Event()
    release = threading.Event()
    submitted = threading.Event()
    results = []
    aggregate = gatherer.aggregator.aggregate

    def blocked_aggregate(fl_ctx):
        aggregating.set()
        release.wait()
        return aggregate(fl_ctx)

    def finish():
        results.append(from_shareable(gatherer.aggregate()).data["weight"])

    def stale_submit():
        try:
            _submit(gatherer, "site-1", 100.0)
        finally:
            submitted.set()

    with ThreadGroup(release) as threads, patch.object(gatherer.aggregator, "aggregate", side_effect=blocked_aggregate):
        threads.start(finish)
        assert aggregating.wait(5)
        threads.start(stale_submit)
        wait_for(lambda: gatherer.lock.blocked.is_set() or submitted.is_set())
        assert gatherer.lock.blocked.is_set()
        assert not submitted.is_set()
    assert results == [1.5]
    assert submitted.is_set()

    # Also cover a caller that resumes only after the old round has closed.
    assert _submit(gatherer, "site-1", 100.0).get_return_code() == ReturnCode.OK
    next_round = _make_gatherer(aggregator=gatherer.aggregator)
    next_round.for_round = 1
    assert _submit(next_round, "site-1", 3.0).get_return_code() == ReturnCode.OK
    assert _submit(next_round, "site-2", 5.0).get_return_code() == ReturnCode.OK
    assert from_shareable(next_round.aggregate()).data["weight"] == 4.0
