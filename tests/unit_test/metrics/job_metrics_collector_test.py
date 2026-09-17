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

from nvflare.apis.event_type import EventType
from nvflare.app_common.app_event_type import AppEventType
from nvflare.metrics.job_metrics_collector import JobMetricsCollector
from nvflare.metrics.metrics_collector import MetricsCollector
from nvflare.metrics.metrics_keys import MetricKeys


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


def test_repeated_task_execution_begin_refreshes_start_time(monkeypatch):
    event_times = iter([100.0, 101.0, 109.0])
    monkeypatch.setattr("nvflare.metrics.metrics_collector.time.time", lambda: next(event_times))
    collector, published_metrics = _collector_with_captured_metrics()

    collector.collect_event_metrics(EventType.BEFORE_TASK_EXECUTION, {}, MagicMock())
    collector.collect_event_metrics(EventType.BEFORE_TASK_EXECUTION, {}, MagicMock())
    collector.collect_event_metrics(EventType.AFTER_TASK_EXECUTION, {}, MagicMock())

    assert _duration_values(published_metrics, "_task_execution") == [8.0]
    assert collector.event_start_time["_task_execution"] is None


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
    assert collector.event_start_time["_custom"] is None


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
    assert collector.event_start_time["_legacy"] is None


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
    assert collector.event_start_time["_task_execution"] is None


def test_legacy_singleton_pair_event_toggle_is_preserved(monkeypatch):
    event_times = iter([10.0, 13.5])
    monkeypatch.setattr("nvflare.metrics.metrics_collector.time.time", lambda: next(event_times))
    collector, published_metrics = _collector_with_captured_metrics()

    collector.collect_event_metrics(AppEventType.RECEIVE_BEST_MODEL, {}, MagicMock())
    collector.collect_event_metrics(AppEventType.RECEIVE_BEST_MODEL, {}, MagicMock())

    assert _duration_values(published_metrics, "_receive_best_model") == [3.5]
    assert collector.event_start_time["_receive_best_model"] is None
