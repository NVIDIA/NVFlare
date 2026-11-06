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

from unittest.mock import MagicMock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.job_def import JobMetaKey
from nvflare.app_common.app_event_type import AppEventType
from nvflare.metrics.job_metrics_collector import JobMetricsCollector
from nvflare.metrics.metrics_collector import MetricsCollector
from nvflare.metrics.metrics_keys import MetricKeys
from tests.timing_utils import isolate_time


class _ReorderedPairMetricsCollector(MetricsCollector):
    def __init__(self):
        super().__init__(tags={})
        self.single_events = []
        self.pair_events = {
            "END": "_custom",
            "START": "_custom",
        }
        self.pair_start_events = ["START"]

    def get_single_events(self):
        return self.single_events

    def get_pair_events(self):
        return self.pair_events

    def get_pair_start_events(self):
        return self.pair_start_events


class _LegacyMultiPairMetricsCollector(MetricsCollector):
    def __init__(self):
        super().__init__(tags={})
        self.single_events = []
        self.pair_events = {
            "A": "_legacy",
            "B": "_legacy",
        }

    def get_single_events(self):
        return self.single_events

    def get_pair_events(self):
        return self.pair_events


def _collector_with_captured_metrics():
    collector = JobMetricsCollector(tags={})
    published_metrics = []

    def capture(metrics: dict, metric_name: str, tags: dict, fl_ctx):
        published_metrics.append((metric_name, dict(metrics)))

    collector.publish_metrics = capture
    return collector, published_metrics


def _duration_values(published_metrics, metric_name: str):
    return [
        metrics[MetricKeys.time_taken]
        for name, metrics in published_metrics
        if name == metric_name and MetricKeys.time_taken in metrics
    ]


@pytest.fixture(autouse=True)
def _isolate_module_clocks(monkeypatch):
    isolate_time(monkeypatch, "nvflare.metrics.metrics_collector")


def test_repeated_task_execution_begin_refreshes_start_time(monkeypatch):
    event_times = iter([100.0, 101.0, 109.0])
    monkeypatch.setattr("nvflare.metrics.metrics_collector.time.time", lambda: next(event_times))
    collector, published_metrics = _collector_with_captured_metrics()

    collector.collect_event_metrics(EventType.BEFORE_TASK_EXECUTION, {}, MagicMock())
    collector.collect_event_metrics(EventType.BEFORE_TASK_EXECUTION, {}, MagicMock())
    collector.collect_event_metrics(EventType.AFTER_TASK_EXECUTION, {}, MagicMock())

    assert _duration_values(published_metrics, "_task_execution") == [8.0]
    assert collector.event_start_time.get("_task_execution") is None


def test_explicit_start_events_do_not_depend_on_pair_event_order(monkeypatch):
    event_times = iter([20.0, 24.5])
    monkeypatch.setattr("nvflare.metrics.metrics_collector.time.time", lambda: next(event_times))
    collector = _ReorderedPairMetricsCollector()
    published_metrics = []

    def capture(metrics: dict, metric_name: str, tags: dict, fl_ctx):
        published_metrics.append((metric_name, dict(metrics)))

    collector.publish_metrics = capture

    collector.collect_event_metrics("START", {}, MagicMock())
    collector.collect_event_metrics("END", {}, MagicMock())

    assert _duration_values(published_metrics, "_custom") == [4.5]
    assert collector.event_start_time.get("_custom") is None


def test_multi_event_pair_without_explicit_start_preserves_legacy_toggle(monkeypatch):
    event_times = iter([30.0, 34.5])
    monkeypatch.setattr("nvflare.metrics.metrics_collector.time.time", lambda: next(event_times))
    collector = _LegacyMultiPairMetricsCollector()
    published_metrics = []

    def capture(metrics: dict, metric_name: str, tags: dict, fl_ctx):
        published_metrics.append((metric_name, dict(metrics)))

    collector.publish_metrics = capture

    collector.collect_event_metrics("B", {}, MagicMock())
    collector.collect_event_metrics("A", {}, MagicMock())

    assert _duration_values(published_metrics, "_legacy") == [4.5]
    assert collector.event_start_time.get("_legacy") is None


def test_task_execution_end_without_start_does_not_publish_duration(monkeypatch):
    monkeypatch.setattr("nvflare.metrics.metrics_collector.time.time", lambda: 109.0)
    collector, published_metrics = _collector_with_captured_metrics()

    collector.collect_event_metrics(EventType.AFTER_TASK_EXECUTION, {}, MagicMock())
    collector.collect_event_metrics(EventType.ABORT_TASK, {}, MagicMock())

    assert _duration_values(published_metrics, "_task_execution") == []
    assert collector.event_start_time.get("_task_execution") is None


def test_abort_task_ends_task_execution_pair(monkeypatch):
    event_times = iter([200.0, 203.25])
    monkeypatch.setattr("nvflare.metrics.metrics_collector.time.time", lambda: next(event_times))
    collector, published_metrics = _collector_with_captured_metrics()

    collector.collect_event_metrics(EventType.BEFORE_TASK_EXECUTION, {}, MagicMock())
    collector.collect_event_metrics(EventType.ABORT_TASK, {}, MagicMock())

    assert _duration_values(published_metrics, "_task_execution") == [3.25]
    assert collector.event_start_time.get("_task_execution") is None


def test_legacy_singleton_pair_event_toggle_is_preserved(monkeypatch):
    event_times = iter([10.0, 13.5])
    monkeypatch.setattr("nvflare.metrics.metrics_collector.time.time", lambda: next(event_times))
    collector, published_metrics = _collector_with_captured_metrics()

    collector.collect_event_metrics(AppEventType.RECEIVE_BEST_MODEL, {}, MagicMock())
    collector.collect_event_metrics(AppEventType.RECEIVE_BEST_MODEL, {}, MagicMock())

    assert _duration_values(published_metrics, "_receive_best_model") == [3.5]
    assert collector.event_start_time.get("_receive_best_model") is None


@pytest.mark.parametrize("end_event", [EventType.JOB_COMPLETED, EventType.JOB_ABORTED, EventType.JOB_CANCELLED])
def test_overlapping_job_metrics_keep_identity_and_duration_per_job(monkeypatch, end_event):
    event_times = iter([100.0, 110.0, 120.0, 121.0, 135.0])
    monkeypatch.setattr("nvflare.metrics.metrics_collector.time.time", lambda: next(event_times))
    manager = FLContextManager()
    contexts = []
    for job_id in ("job-A", "job-B"):
        ctx = manager.new_context()
        ctx.set_prop(FLContextKey.EVENT_DATA, {JobMetaKey.JOB_ID.value: job_id}, sticky=False)
        contexts.append(ctx)
    # Explicit lifecycle IDs must take precedence over any legacy shared value.
    manager.new_context().set_prop(FLContextKey.CURRENT_JOB_ID, "wrong-job")
    collector = JobMetricsCollector(tags={"site": "server"})
    collector.publish_metrics = MagicMock()

    collector.handle_event(EventType.JOB_STARTED, contexts[0])
    collector.handle_event(EventType.JOB_STARTED, contexts[1])
    collector.handle_event(end_event, contexts[0])
    collector.handle_event(EventType.JOB_COMPLETED, contexts[0])
    collector.handle_event(EventType.JOB_COMPLETED, contexts[1])

    calls = collector.publish_metrics.call_args_list
    durations = [
        (tags["job_id"], metrics[MetricKeys.time_taken])
        for metrics, name, tags, _ctx in (c.args for c in calls)
        if name == "_job"
    ]
    assert durations == [("job-A", 20.0), ("job-B", 25.0)]
    assert calls[0].args[2] == {"site": "server", "job_id": "job-A"}
    assert calls[1].args[2] == {"site": "server", "job_id": "job-B"}
    assert collector.tags == {"site": "server"}
    assert not collector.event_start_time


@pytest.mark.parametrize("current_job_id", [None, "job-1"])
def test_job_metrics_fall_back_to_context_job_id(current_job_id):
    ctx = FLContextManager(job_id="job-1" if current_job_id is None else "").new_context()
    if current_job_id:
        ctx.set_prop(FLContextKey.CURRENT_JOB_ID, current_job_id, sticky=False)
    collector = JobMetricsCollector(tags={})
    collector.publish_metrics = MagicMock()

    collector.handle_event(EventType.DEPLOY_JOB_TO_SERVER, ctx)

    assert collector.publish_metrics.call_args.args[2] == {"job_id": "job-1"}


def test_legacy_singleton_pairing_is_isolated_by_job(monkeypatch):
    event_times = iter([10.0, 20.0, 30.0, 50.0])
    monkeypatch.setattr("nvflare.metrics.metrics_collector.time.time", lambda: next(event_times))
    collector, published_metrics = _collector_with_captured_metrics()
    ctx_a = FLContextManager(job_id="job-A").new_context()
    ctx_b = FLContextManager(job_id="job-B").new_context()

    for ctx in (ctx_a, ctx_b, ctx_a, ctx_b):
        collector.handle_event(AppEventType.RECEIVE_BEST_MODEL, ctx)

    assert _duration_values(published_metrics, "_receive_best_model") == [20.0, 30.0]
    assert not collector.event_start_time
