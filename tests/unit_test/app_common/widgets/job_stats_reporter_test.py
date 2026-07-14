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

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.aggregators.weighted_aggregation_helper import AggregationStatsKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.widgets.job_stats_reporter import JobStatsReporter, JobStatusCode

JOB_ID = "test_job"


def make_engine(tmp_path, client_names):
    engine = Mock()
    engine.get_clients.return_value = [SimpleNamespace(name=n) for n in client_names]
    workspace = Mock()
    workspace.get_run_dir.return_value = str(tmp_path)
    engine.get_workspace.return_value = workspace
    return engine


def make_ctx(engine, peer_name=None, props=None):
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    ctx.set_prop(ReservedKey.RUN_NUM, JOB_ID, private=True, sticky=False)
    if peer_name:
        peer_ctx = FLContext()
        peer_ctx.set_prop(ReservedKey.IDENTITY_NAME, peer_name, private=False, sticky=False)
        ctx.set_peer_context(peer_ctx)
    if props:
        for k, v in props.items():
            ctx.set_prop(k, v, private=True, sticky=False)
    return ctx


def make_aggr_stats(round_num, contributors, keys_aggregated, keys_seen, fully, partially, skipped):
    return {
        AggregationStatsKey.ROUND: round_num,
        AggregationStatsKey.ACCEPTED_CONTRIBUTIONS: len(contributors),
        AggregationStatsKey.CONTRIBUTORS: list(contributors),
        AggregationStatsKey.KEYS_AGGREGATED: keys_aggregated,
        AggregationStatsKey.KEYS_SEEN: keys_seen,
        AggregationStatsKey.FULLY_MATCHED_KEYS: fully,
        AggregationStatsKey.PARTIALLY_MATCHED_KEYS: partially,
        AggregationStatsKey.SKIPPED_KEYS: skipped,
    }


def fire_round(reporter, engine, round_num, contributing_clients, aggr_stats, error_clients=None):
    error_clients = error_clients or []
    reporter.handle_event(AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: round_num}))
    for client in contributing_clients + error_clients:
        task_id = f"task-{round_num}-{client}"
        reporter.handle_event(
            EventType.AFTER_TASK_DATA_FILTER,
            make_ctx(engine, peer_name=client, props={FLContextKey.TASK_NAME: "train", FLContextKey.TASK_ID: task_id}),
        )
        if client in error_clients:
            result = make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            result = Shareable()
        reporter.handle_event(
            EventType.BEFORE_PROCESS_SUBMISSION,
            make_ctx(
                engine,
                peer_name=client,
                props={
                    FLContextKey.TASK_NAME: "train",
                    FLContextKey.TASK_ID: task_id,
                    FLContextKey.TASK_RESULT: result,
                },
            ),
        )
        if client not in error_clients:
            reporter.handle_event(
                AppEventType.AFTER_CONTRIBUTION_ACCEPT,
                make_ctx(
                    engine,
                    peer_name=client,
                    props={AppConstants.CURRENT_ROUND: round_num, AppConstants.AGGREGATION_ACCEPTED: True},
                ),
            )
    agg_ctx = make_ctx(
        engine, props={AppConstants.CURRENT_ROUND: round_num, AppConstants.AGGREGATION_STATS: aggr_stats}
    )
    reporter.handle_event(AppEventType.AFTER_AGGREGATION, agg_ctx)
    reporter.handle_event(AppEventType.ROUND_DONE, agg_ctx)


class TestJobStatsReporter:
    def test_successful_run_writes_report(self, tmp_path):
        clients = ["site-1", "site-2"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter(json_filename="job_stats_run_summary.json")

        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        for round_num in range(3):
            fire_round(reporter, engine, round_num, clients, make_aggr_stats(round_num, clients, 10, 12, 8, 2, 2))
        end_ctx = make_ctx(engine)
        reporter.handle_event(EventType.END_RUN, end_ctx)

        summary = reporter.get_summary(end_ctx)
        assert summary["status"] == JobStatusCode.SUCCESS
        assert summary["participation"]["total_rounds"] == 3
        assert summary["participation"]["total_clients"] == 2
        assert summary["participation"]["failure_rate_percent"] == 0.0
        assert summary["consistency"]["stable_rounds"] == 3
        assert summary["consistency"]["inconsistent_rounds"] == []
        assert [r["keys_aggregated"] for r in summary["rounds"]] == [10, 10, 10]
        assert summary["timing"]["task_breakdown"].keys() == {"train"}

        report_file = os.path.join(str(tmp_path), "job_stats_run_summary.log")
        assert os.path.isfile(report_file)
        with open(report_file, encoding="utf-8") as f:
            report = f.read()
        assert "SUCCESS" in report
        assert "10 / 12" in report
        assert "2/2" in report
        assert os.path.isfile(os.path.join(str(tmp_path), "job_stats_run_summary.json"))

    def test_partial_success_with_missing_client_and_error(self, tmp_path):
        clients = ["site-1", "site-2", "site-3"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()

        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        fire_round(reporter, engine, 0, clients, make_aggr_stats(0, clients, 10, 10, 10, 0, 0))
        fire_round(reporter, engine, 1, clients, make_aggr_stats(1, clients, 10, 10, 10, 0, 0))
        # round 2: site-3 fails with an error result and does not contribute
        fire_round(
            reporter,
            engine,
            2,
            ["site-1", "site-2"],
            make_aggr_stats(2, ["site-1", "site-2"], 8, 10, 6, 2, 2),
            error_clients=["site-3"],
        )
        end_ctx = make_ctx(engine)
        reporter.handle_event(EventType.END_RUN, end_ctx)

        summary = reporter.get_summary(end_ctx)
        assert summary["status"] == JobStatusCode.PARTIAL_SUCCESS
        assert summary["participation"]["failed_clients"] == ["site-3"]
        assert summary["participation"]["client_errors"] == {"site-3": [ReturnCode.EXECUTION_EXCEPTION]}
        assert summary["consistency"]["stable_rounds"] == 2
        assert summary["consistency"]["inconsistent_rounds"] == [2]
        # 1 missing contribution out of 9 client-round slots
        assert summary["participation"]["failure_rate_percent"] == pytest.approx(100.0 / 9)

        report = reporter.format_report(summary)
        assert "PARTIAL SUCCESS" in report
        assert "Round 2" in report
        assert "site-3" in report

    def test_failure_on_fatal_system_error(self, tmp_path):
        clients = ["site-1"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()

        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        fire_round(reporter, engine, 0, clients, make_aggr_stats(0, clients, 5, 5, 5, 0, 0))
        reporter.handle_event(
            EventType.FATAL_SYSTEM_ERROR, make_ctx(engine, props={FLContextKey.EVENT_DATA: "bad task result"})
        )
        end_ctx = make_ctx(engine)
        reporter.handle_event(EventType.END_RUN, end_ctx)

        summary = reporter.get_summary(end_ctx)
        assert summary["status"] == JobStatusCode.FAILURE
        assert summary["status_reason"] == "bad task result"
        assert "FAILURE" in reporter.format_report(summary)

    def test_client_disconnect_is_partial_success(self, tmp_path):
        clients = ["site-1", "site-2"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()

        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        fire_round(reporter, engine, 0, clients, make_aggr_stats(0, clients, 5, 5, 5, 0, 0))
        reporter.handle_event(
            EventType.CLIENT_DISCONNECTED, make_ctx(engine, props={FLContextKey.DISCONNECTED_CLIENT_NAME: "site-2"})
        )
        end_ctx = make_ctx(engine)
        reporter.handle_event(EventType.END_RUN, end_ctx)

        summary = reporter.get_summary(end_ctx)
        assert summary["status"] == JobStatusCode.PARTIAL_SUCCESS
        assert summary["participation"]["disconnected_clients"] == ["site-2"]

    def test_no_aggregation_stats_reports_na(self, tmp_path):
        clients = ["site-1", "site-2"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()

        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        # a round without any published aggregation stats (e.g. custom aggregator)
        fire_round(reporter, engine, 0, clients, aggr_stats=None)
        end_ctx = make_ctx(engine)
        reporter.handle_event(EventType.END_RUN, end_ctx)

        summary = reporter.get_summary(end_ctx)
        assert summary["rounds"][0]["keys_aggregated"] is None
        assert summary["consistency"]["stable_rounds"] is None
        report = reporter.format_report(summary)
        assert "No aggregation key stats available" in report

    def test_round_timing_recorded(self, tmp_path):
        clients = ["site-1"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()

        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        fire_round(reporter, engine, 0, clients, make_aggr_stats(0, clients, 5, 5, 5, 0, 0))
        end_ctx = make_ctx(engine)
        reporter.handle_event(EventType.END_RUN, end_ctx)

        summary = reporter.get_summary(end_ctx)
        assert summary["timing"]["total_job_time"] is not None
        assert summary["rounds"][0]["duration"] is not None
        assert "train" in summary["timing"]["task_breakdown"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
