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
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.swarm_client_ctl import Gatherer, SwarmClientController
from tests.timing_utils import ManualClock, ObservedLock, ThreadGroup, join_thread, wait_for


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


def _submit(gatherer, site, value, expected_rc=ReturnCode.OK):
    result = DXO(DataKind.WEIGHTS, {"weight": value}).to_shareable()
    result.set_header(AppConstants.CURRENT_ROUND, gatherer.for_round)
    result.add_cookie(AppConstants.CONTRIBUTION_ROUND, gatherer.for_round)
    result.set_peer_props({ReservedKey.IDENTITY_NAME: site})
    fl_ctx = FLContext()
    fl_ctx.put(ReservedKey.ENGINE, Mock(), private=True, sticky=False)
    reply = gatherer.gather(site, result, fl_ctx)
    assert reply.get_return_code() == expected_rc
    return reply


@pytest.mark.parametrize("completion", ["all", "minimum", "timeout", "aggregate"])
def test_completion_waits_for_inflight_acceptance(completion, monkeypatch):
    clock = ManualClock()
    monkeypatch.setattr("nvflare.app_common.ccwf.swarm_client_ctl.time", clock)
    trainers = ["site-1", "site-2"] if completion == "all" else ["site-1", "site-2", "site-3"]
    gatherer = _make_gatherer(
        trainers=trainers,
        min_responses=1 if completion == "minimum" else 2,
        timeout=5 if completion in ["timeout", "aggregate"] else 0,
    )
    assert _submit(gatherer, "site-1", 1.0).get_return_code() == ReturnCode.OK
    accepting = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    results = []
    checked = threading.Event()
    decisions = []
    accept = gatherer.aggregator.accept

    def blocked_accept(result, fl_ctx):
        accepting.set()
        release.wait()
        return accept(result, fl_ctx)

    def finish():
        try:
            results.append(from_shareable(gatherer.aggregate()).data["weight"])
        finally:
            finished.set()

    def check_done():
        try:
            decisions.append(gatherer.is_done())
        finally:
            checked.set()

    if completion == "aggregate":
        # A successful poll does not exclude an acceptance starting before closure.
        clock.advance(10)
        assert gatherer.is_done()

    with ThreadGroup(release) as threads, patch.object(gatherer.aggregator, "accept", side_effect=blocked_accept):
        threads.start(_submit, gatherer, "site-2", 2.0)
        assert accepting.wait(5)
        clock.advance(10)
        if completion == "aggregate":
            threads.start(finish)
            wait_for(lambda: gatherer.lock.blocked.is_set() or finished.is_set())
            assert gatherer.lock.blocked.is_set()
            assert not finished.is_set()
        else:
            threads.start(check_done)
            assert checked.wait(5)
            assert decisions == [False]
            assert gatherer.lock.blocked.is_set()
    if completion != "aggregate":
        assert gatherer.is_done()
        finish()
    assert finished.is_set()
    assert results == [1.5]


def test_stale_submission_cannot_contaminate_next_round(caplog):
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

    caplog.set_level("INFO", logger=gatherer.logger.name)
    with (
        patch.object(gatherer.aggregator, "aggregate", side_effect=blocked_aggregate),
        patch.object(gatherer.aggregator, "accept", wraps=gatherer.aggregator.accept) as late_accept,
        ThreadGroup(release) as threads,
    ):
        threads.start(finish)
        assert aggregating.wait(5)
        threads.start(stale_submit)
        assert submitted.wait(5)
        assert not gatherer.lock.blocked.is_set()
        late_accept.assert_not_called()
    assert results == [1.5]
    assert submitted.is_set()

    # Also cover a caller that resumes only after the old round has closed.
    with patch.object(gatherer.aggregator, "accept", wraps=gatherer.aggregator.accept) as late_accept:
        assert _submit(gatherer, "site-1", 100.0).get_return_code() == ReturnCode.OK
        late_accept.assert_not_called()
    assert "dropped result from late site-1 for closed round 0" in caplog.text
    next_round = _make_gatherer(aggregator=gatherer.aggregator)
    next_round.for_round = 1
    assert _submit(next_round, "site-1", 3.0).get_return_code() == ReturnCode.OK
    assert _submit(next_round, "site-2", 5.0).get_return_code() == ReturnCode.OK
    assert from_shareable(next_round.aggregate()).data["weight"] == 4.0


def test_monitor_can_stop_while_acceptance_is_blocked(monkeypatch):
    clock = ManualClock()
    monkeypatch.setattr("nvflare.app_common.ccwf.swarm_client_ctl.time", clock)
    gatherer = _make_gatherer()
    _submit(gatherer, "site-1", 1.0)
    controller = SwarmClientController()
    controller.gatherer = gatherer
    controller._end_gather = Mock()
    accepting = threading.Event()
    release_accept = threading.Event()
    polling = threading.Event()
    release_poll = threading.Event()
    accept = gatherer.aggregator.accept

    def blocked_accept(result, fl_ctx):
        accepting.set()
        release_accept.wait()
        return accept(result, fl_ctx)

    def monitor_sleep(_):
        polling.set()
        release_poll.wait()

    monkeypatch.setattr(clock, "sleep", monitor_sleep)
    with (
        ThreadGroup(release_accept, release_poll) as threads,
        patch.object(gatherer.aggregator, "accept", side_effect=blocked_accept),
    ):
        try:
            threads.start(_submit, gatherer, "site-2", 2.0)
            assert accepting.wait(5)
            monitor = threads.start(controller._monitor_gather)
            assert polling.wait(5)
            assert gatherer.lock.blocked.is_set()
            controller.asked_to_stop = True
            release_poll.set()
            join_thread(monitor)
            assert not release_accept.is_set()
            controller._end_gather.assert_not_called()
        finally:
            controller.asked_to_stop = True


def test_aggregation_callbacks_run_without_gather_lock():
    gatherer = _make_gatherer()
    _submit(gatherer, "site-1", 1.0)
    _submit(gatherer, "site-2", 2.0)
    events = []

    def on_aggregate(event_type, *args, **kwargs):
        if event_type not in [AppEventType.BEFORE_AGGREGATION, AppEventType.AFTER_AGGREGATION]:
            return
        assert gatherer.lock.acquire(blocking=False)
        gatherer.lock.release()
        assert gatherer.is_done()
        # A callback can safely acknowledge a late result without changing the model.
        _submit(gatherer, "site-1", 100.0)
        events.append(event_type)

    gatherer.fire_event.side_effect = on_aggregate
    gatherer.fire_event_with_data.side_effect = on_aggregate
    assert from_shareable(gatherer.aggregate()).data["weight"] == 1.5
    assert events == [AppEventType.BEFORE_AGGREGATION, AppEventType.AFTER_AGGREGATION]


@pytest.mark.parametrize("error", [False, True])
def test_unsuccessful_acceptance_does_not_leave_lock_held(error):
    gatherer = _make_gatherer()
    _submit(gatherer, "site-1", 1.0)
    submission = Shareable()
    submission.set_header(AppConstants.CURRENT_ROUND, 0)
    assert gatherer.can_accept_submission("site-2", submission, FLContext()) == ReturnCode.OK
    expected_rc = ReturnCode.EXECUTION_EXCEPTION if error else ReturnCode.OK
    with patch.object(
        gatherer.aggregator, "accept", side_effect=RuntimeError("accept failed") if error else None
    ) as accept:
        accept.return_value = False
        _submit(gatherer, "site-2", 2.0, expected_rc)

    assert gatherer.lock.acquire(blocking=False)
    gatherer.lock.release()
    assert not gatherer.trainer_statuses["site-2"].busy
    # Preserve existing response accounting for failed/rejected results.
    assert gatherer.is_done()
    assert from_shareable(gatherer.aggregate()).data["weight"] == 1.0


@pytest.mark.parametrize("submit_before_close", [True, False])
def test_submission_ordered_before_or_after_round_closure(submit_before_close, monkeypatch):
    clock = ManualClock()
    monkeypatch.setattr("nvflare.app_common.ccwf.swarm_client_ctl.time", clock)
    gatherer = _make_gatherer(timeout=5)
    _submit(gatherer, "site-1", 1.0)
    clock.advance(10)
    assert gatherer.is_done()

    if submit_before_close:
        _submit(gatherer, "site-2", 2.0)
    result = from_shareable(gatherer.aggregate()).data["weight"]
    if not submit_before_close:
        _submit(gatherer, "site-2", 2.0)
    assert result == (1.5 if submit_before_close else 1.0)

    next_round = _make_gatherer(aggregator=gatherer.aggregator)
    next_round.for_round = 1
    _submit(next_round, "site-1", 3.0)
    _submit(next_round, "site-2", 5.0)
    assert from_shareable(next_round.aggregate()).data["weight"] == 4.0


@pytest.mark.parametrize("policy,deadline", [("minimum", 1), ("timeout", 5)])
def test_completion_deadlines_are_preserved(policy, deadline, monkeypatch):
    clock = ManualClock()
    monkeypatch.setattr("nvflare.app_common.ccwf.swarm_client_ctl.time", clock)
    gatherer = _make_gatherer(min_responses=1 if policy == "minimum" else 2, timeout=5 if policy == "timeout" else 0)
    _submit(gatherer, "site-1", 1.0)
    assert gatherer.is_done() is False
    clock.advance(deadline)
    assert gatherer.is_done() is False
    clock.advance(0.01)
    assert gatherer.is_done()
    assert from_shareable(gatherer.aggregate()).data["weight"] == 1.0


def test_multiple_submissions_remain_serialized():
    gatherer = _make_gatherer(trainers=["site-1", "site-2", "site-3"], min_responses=3)
    _submit(gatherer, "site-1", 1.0)
    accepting = threading.Event()
    release = threading.Event()
    third_accepted = threading.Event()
    accept = gatherer.aggregator.accept

    def blocked_accept(result, fl_ctx):
        if result.get_peer_prop(ReservedKey.IDENTITY_NAME, None) == "site-2":
            accepting.set()
            release.wait()
        else:
            third_accepted.set()
        return accept(result, fl_ctx)

    with patch.object(gatherer.aggregator, "accept", side_effect=blocked_accept), ThreadGroup(release) as threads:
        threads.start(_submit, gatherer, "site-2", 2.0)
        assert accepting.wait(5)
        threads.start(_submit, gatherer, "site-3", 3.0)
        assert gatherer.lock.blocked.wait(5)
        assert not third_accepted.is_set()
    assert third_accepted.is_set()
    assert gatherer.is_done()
    assert from_shareable(gatherer.aggregate()).data["weight"] == 2.0


@pytest.mark.parametrize("event", [AppEventType.BEFORE_AGGREGATION, AppEventType.AFTER_AGGREGATION])
def test_aggregation_callback_failure_keeps_old_gatherer_closed(event):
    gatherer = _make_gatherer()
    _submit(gatherer, "site-1", 1.0)
    _submit(gatherer, "site-2", 2.0)

    def fail_callback(event_type, *args, **kwargs):
        if event_type == event:
            raise RuntimeError("aggregation callback failed")

    gatherer.fire_event.side_effect = fail_callback
    gatherer.fire_event_with_data.side_effect = fail_callback
    with pytest.raises(RuntimeError, match="aggregation callback failed"):
        gatherer.aggregate()
    assert gatherer.lock.acquire(blocking=False)
    gatherer.lock.release()
    with patch.object(gatherer.aggregator, "accept") as accept:
        _submit(gatherer, "site-1", 100.0)
        accept.assert_not_called()


@pytest.mark.parametrize("already_permitted", [False, True])
def test_closed_gatherer_rejects_upload_permission(already_permitted):
    gatherer = _make_gatherer()
    _submit(gatherer, "site-1", 1.0)
    _submit(gatherer, "site-2", 2.0)
    request = Shareable()
    request.set_header(AppConstants.CURRENT_ROUND, 0)
    if already_permitted:
        assert gatherer.can_accept_submission("site-2", request, FLContext()) == ReturnCode.OK
    gatherer.aggregate()
    assert gatherer.can_accept_submission("site-2", request, FLContext()) == ReturnCode.MODEL_UNRECOGNIZED
    assert gatherer.trainer_statuses["site-2"].busy == already_permitted


def test_permission_check_remains_responsive_during_acceptance():
    gatherer = _make_gatherer()
    _submit(gatherer, "site-1", 1.0)
    request = Shareable()
    request.set_header(AppConstants.CURRENT_ROUND, 0)
    assert gatherer.can_accept_submission("site-2", request, FLContext()) == ReturnCode.OK
    accepting = threading.Event()
    release = threading.Event()
    checked = threading.Event()
    replies = []
    accept = gatherer.aggregator.accept

    def blocked_accept(result, fl_ctx):
        accepting.set()
        release.wait()
        return accept(result, fl_ctx)

    def check_permission():
        try:
            replies.append(gatherer.can_accept_submission("site-1", request, FLContext()))
        finally:
            checked.set()

    with patch.object(gatherer.aggregator, "accept", side_effect=blocked_accept), ThreadGroup(release) as threads:
        threads.start(_submit, gatherer, "site-2", 2.0)
        assert accepting.wait(5)
        threads.start(check_permission)
        assert checked.wait(5)
        assert replies == [ReturnCode.SERVICE_UNAVAILABLE]
        assert not gatherer.lock.blocked.is_set()
    assert gatherer.can_accept_submission("site-1", request, FLContext()) == ReturnCode.OK
    assert from_shareable(gatherer.aggregate()).data["weight"] == 1.5
