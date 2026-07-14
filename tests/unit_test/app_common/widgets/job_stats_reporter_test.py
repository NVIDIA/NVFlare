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

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ProcessType, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.aggregators.weighted_aggregation_helper import AggregationStatsKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.widgets.job_stats_reporter import JobStatsReporter, JobStatusCode, _ResourceSampler

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
            make_ctx(
                engine,
                peer_name=client,
                props={
                    FLContextKey.TASK_NAME: "train",
                    FLContextKey.TASK_ID: task_id,
                    FLContextKey.TASK_DATA: Shareable(),
                    AppConstants.CURRENT_ROUND: round_num,
                },
            ),
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
        reporter = JobStatsReporter()

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
        assert summary["final_aggregation"]["round"] == "Final"
        assert summary["final_aggregation"]["source_round"] == 2
        assert summary["final_aggregation"]["keys_aggregated"] == 10
        assert summary["final_aggregation"]["aggregation_stats"] == summary["rounds"][-1]["aggregation_stats"]
        assert summary["timing"]["task_breakdown"].keys() == {"train"}

        report_file = os.path.join(str(tmp_path), "job_stats_run_summary.log")
        assert os.path.isfile(report_file)
        with open(report_file, encoding="utf-8") as f:
            report = f.read()
        assert "SUCCESS" in report
        assert "Aggregation Stats Summary (3 rounds + final)" in report
        final_rows = [line for line in report.splitlines() if line.startswith("Final")]
        assert len(final_rows) == 1
        assert "10 / 12" in final_rows[0]
        assert "10 / 12" in report
        assert "2/2" in report
        # per-round key-match columns
        assert "Fully Matched Keys" in report
        assert "Partially Matched Keys" in report
        assert "Skipped Keys" in report
        # participation and timing rows from the sample layout
        assert "Total Clients" in report
        assert "Avg Clients Participated / Round" in report
        assert "Avg Client Computation Time" in report
        assert "Avg Download Size per Client" in report
        # consistency section
        assert "Aggregation Consistency Across Rounds" in report
        assert "Stable Rounds" in report
        assert "3 / 3" in report
        assert "Inconsistent Rounds" in report
        # per-task-name timing breakdown
        assert "Task Timing Breakdown" in report
        assert "  train" in report
        json_path = os.path.join(str(tmp_path), "job_stats_run_summary.json")
        assert os.path.isfile(json_path)
        with open(json_path, encoding="utf-8") as f:
            json_summary = json.load(f)
        assert {
            "status",
            "rounds",
            "final_aggregation",
            "consistency",
            "participation",
            "timing",
            "communication",
            "resources",
            "errors",
            "report_files",
        }.issubset(json_summary)
        assert json_summary["final_aggregation"]["source_round"] == 2
        assert len(json_summary["timing"]["tasks"]) == 6
        assert json_summary["job_start_time"] is not None
        assert json_summary["job_end_time"] is not None
        assert json_summary["rounds"][0]["started_at"] is not None
        assert json_summary["timing"]["tasks"][0]["assigned_at"] is not None
        assert json_summary["report_files"]["json"] == "job_stats_run_summary.json"

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
        assert summary["status"] == "PARTIAL"
        assert summary["participation"]["failed_clients"] == ["site-3"]
        assert summary["participation"]["client_errors"] == {"site-3": [ReturnCode.EXECUTION_EXCEPTION]}
        assert summary["consistency"]["stable_rounds"] == 2
        assert summary["consistency"]["inconsistent_rounds"] == [2]
        # 1 missing contribution out of 9 client-round slots
        assert summary["participation"]["failure_rate_percent"] == pytest.approx(100.0 / 9)

        report = reporter.format_report(summary)
        assert "PARTIAL SUCCESS" in report
        assert "round=2" in report
        assert "[site-3] (1/3)" in report
        # the inconsistent round is called out in the consistency section
        assert "Inconsistent Rounds" in report
        assert "Round 2" in report

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

    def test_zero_response_round_is_failure(self, tmp_path):
        clients = ["site-1", "site-2"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        reporter.handle_event(AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: 0}))
        for client in clients:
            reporter.handle_event(
                EventType.AFTER_TASK_DATA_FILTER,
                make_ctx(
                    engine,
                    peer_name=client,
                    props={
                        FLContextKey.TASK_NAME: "train",
                        FLContextKey.TASK_ID: f"task-{client}",
                        FLContextKey.TASK_DATA: Shareable(),
                        AppConstants.CURRENT_ROUND: 0,
                    },
                ),
            )
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        summary = reporter.get_summary(make_ctx(engine))
        assert summary["status"] == JobStatusCode.FAILURE
        assert summary["rounds"][0]["participation_rate_percent"] == 0.0
        assert summary["participation"]["failure_rate_percent"] == 100.0
        assert summary["participation"]["incomplete_tasks"] == 2

    def test_participation_denominator_is_round_targets_not_connected_clients(self, tmp_path):
        all_clients = ["site-1", "site-2", "site-3"]
        targeted = ["site-1", "site-2"]
        engine = make_engine(tmp_path, all_clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        fire_round(reporter, engine, 0, targeted, make_aggr_stats(0, targeted, 5, 5, 5, 0, 0))

        summary = reporter.get_summary(make_ctx(engine))
        assert summary["status"] == JobStatusCode.SUCCESS
        assert summary["rounds"][0]["total_clients"] == 2
        assert summary["rounds"][0]["participation_rate_percent"] == 100.0
        assert summary["participation"]["failure_rate_percent"] == 0.0

    def test_client_telemetry_produces_per_task_client_round_comm_and_resources(self, tmp_path):
        engine = make_engine(tmp_path, ["site-1"])
        server = JobStatsReporter(json_filename="summary.json")
        server.handle_event(EventType.START_RUN, make_ctx(engine))
        server.handle_event(AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: 4}))

        task_data = Shareable({"model": b"server-model"})
        task_data.set_header(AppConstants.CURRENT_ROUND, 4)
        server.handle_event(
            EventType.AFTER_TASK_DATA_FILTER,
            make_ctx(
                engine,
                peer_name="site-1",
                props={
                    FLContextKey.TASK_NAME: "train",
                    FLContextKey.TASK_ID: "task-4",
                    FLContextKey.TASK_DATA: task_data,
                    AppConstants.CURRENT_ROUND: 4,
                },
            ),
        )

        client = JobStatsReporter(collect_resources=False)
        client.handle_event(
            EventType.START_RUN,
            make_ctx(engine, props={FLContextKey.PROCESS_TYPE: ProcessType.CLIENT_JOB}),
        )
        client_ctx = make_ctx(
            engine,
            props={
                FLContextKey.TASK_NAME: "train",
                FLContextKey.TASK_ID: "task-4",
                FLContextKey.TASK_DATA: task_data,
            },
        )
        client.handle_event(EventType.BEFORE_TASK_DATA_FILTER, client_ctx)
        client.handle_event(EventType.BEFORE_TASK_EXECUTION, client_ctx)
        # the runner sets TASK_RESULT before firing AFTER_TASK_EXECUTION (pre-result-filter),
        # which is where the telemetry header is attached
        result = Shareable({"update": b"client-update"})
        client_ctx.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
        client.handle_event(EventType.AFTER_TASK_EXECUTION, client_ctx)

        telemetry = result.get_header(AppConstants.JOB_STATS_CLIENT_TELEMETRY)
        assert telemetry["round"] == 4
        assert telemetry["execution_time"] is not None
        assert telemetry["client_processing_time"] >= telemetry["execution_time"]
        assert telemetry["client_framework_overhead"] >= 0.0
        assert telemetry["update_size_mb"] > 0

        # Supply deterministic resource samples to verify server aggregation/rendering.
        telemetry["resources"] = {
            "cpu_percent": {"count": 2, "mean": 30.0, "stddev": 10.0, "min": 20.0, "max": 40.0},
            "memory_rss_mb": {"count": 2, "mean": 101.0, "stddev": 1.0, "min": 100.0, "max": 102.0},
            "gpu_percent": {"count": 2, "mean": 50.0, "stddev": 5.0, "min": 45.0, "max": 55.0},
            "gpu_memory_mb": {"count": 2, "mean": 512.0, "stddev": 0.0, "min": 512.0, "max": 512.0},
            "battery_start_percent": 90.0,
            "battery_end_percent": 88.0,
            "battery_used_percent": 2.0,
        }
        server.handle_event(
            EventType.BEFORE_PROCESS_SUBMISSION,
            make_ctx(
                engine,
                peer_name="site-1",
                props={
                    FLContextKey.TASK_NAME: "train",
                    FLContextKey.TASK_ID: "task-4",
                    FLContextKey.TASK_RESULT: result,
                },
            ),
        )
        server.handle_event(
            AppEventType.AFTER_CONTRIBUTION_ACCEPT,
            make_ctx(
                engine,
                peer_name="site-1",
                props={AppConstants.CURRENT_ROUND: 4, AppConstants.AGGREGATION_ACCEPTED: True},
            ),
        )
        server.handle_event(AppEventType.ROUND_DONE, make_ctx(engine, props={AppConstants.CURRENT_ROUND: 4}))

        summary = server.get_summary(make_ctx(engine))
        task = summary["timing"]["tasks"][0]
        assert task["round"] == 4
        assert task["client"] == "site-1"
        assert task["client_computation_time"] is not None
        assert task["client_framework_overhead"] is not None
        assert task["communication_time"] is not None
        assert summary["timing"]["client_rounds"][0]["round"] == 4
        assert summary["timing"]["communication_time_per_client_round"]["count"] == 1
        assert summary["timing"]["communication_overhead_percent"]["mean"] >= 0.0
        assert summary["communication"]["server"]["total_sent_mb"] > 0
        assert summary["communication"]["server"]["total_received_mb"] > 0
        assert summary["communication"]["clients"]["site-1"]["model_update_size_mb"]["count"] == 1
        assert summary["communication"]["clients"]["site-1"]["download_size_mb"]["count"] == 1
        cpu = summary["resources"]["site-1"]["cpu_percent"]
        assert cpu["count"] == 2
        assert cpu["mean"] == pytest.approx(30.0)
        assert cpu["stddev"] == pytest.approx(10.0)
        assert cpu["max"] == 40.0
        assert summary["resources"]["site-1"]["battery_used_percent"] == 2.0

        report = server.format_report(summary)
        assert "Every Task Execution" in report
        assert "Per-Client Per-Round Time" in report
        assert "Avg CPU % (mean ± std)" in report
        assert "30.0 ± 10.0" in report
        assert "Time in Communication / Round" in report
        assert "% of round time" in report

    def test_optional_error_file_contains_details(self, tmp_path):
        engine = make_engine(tmp_path, ["site-1"])
        reporter = JobStatsReporter(
            filename="reports/job_stats_run_summary.log",
            json_filename="reports/job_stats_run_summary.json",
            error_filename="errors/job_stats_errors.log",
        )
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        fire_round(
            reporter,
            engine,
            0,
            [],
            make_aggr_stats(0, [], 0, 0, 0, 0, 0),
            error_clients=["site-1"],
        )
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        error_path = os.path.join(str(tmp_path), "errors", "job_stats_errors.log")
        assert os.path.isfile(error_path)
        with open(error_path, encoding="utf-8") as f:
            error_report = f.read()
        assert "site-1" in error_report
        assert ReturnCode.EXECUTION_EXCEPTION in error_report

        # the main report points at the error log
        with open(os.path.join(str(tmp_path), "reports", "job_stats_run_summary.log"), encoding="utf-8") as f:
            report = f.read()
        assert "Error Log Location" in report
        assert "errors/job_stats_errors.log" in report
        assert os.path.isfile(os.path.join(str(tmp_path), "reports", "job_stats_run_summary.json"))

    def test_resource_sampler_collects_cpu_gpu_memory_and_battery(self):
        process = MagicMock()
        process.cpu_percent.side_effect = [0.0, 25.0]
        process.memory_info.return_value = SimpleNamespace(rss=100 * 1024 * 1024)
        fake_psutil = SimpleNamespace(
            Process=Mock(return_value=process),
            sensors_battery=MagicMock(
                side_effect=[
                    SimpleNamespace(percent=90.0, power_plugged=False),
                    SimpleNamespace(percent=88.0, power_plugged=False),
                ]
            ),
        )
        fake_pynvml = SimpleNamespace(
            nvmlInit=Mock(),
            nvmlShutdown=Mock(),
            nvmlDeviceGetCount=Mock(return_value=1),
            nvmlDeviceGetHandleByIndex=Mock(return_value="gpu-0"),
            nvmlDeviceGetProcessUtilization=Mock(
                return_value=[
                    SimpleNamespace(pid=os.getpid(), smUtil=50.0, timeStamp=2),
                    SimpleNamespace(pid=999999, smUtil=90.0, timeStamp=2),
                ]
            ),
            nvmlDeviceGetComputeRunningProcesses=Mock(
                return_value=[
                    SimpleNamespace(pid=os.getpid(), usedGpuMemory=512 * 1024 * 1024),
                    SimpleNamespace(pid=999999, usedGpuMemory=2048 * 1024 * 1024),
                ]
            ),
            nvmlDeviceGetGraphicsRunningProcesses=Mock(return_value=[]),
            nvmlDeviceGetUtilizationRates=Mock(return_value=SimpleNamespace(gpu=99.0)),
            nvmlDeviceGetMemoryInfo=Mock(return_value=SimpleNamespace(used=4096 * 1024 * 1024)),
        )

        with patch.dict("sys.modules", {"psutil": fake_psutil, "pynvml": fake_pynvml}):
            sampler = _ResourceSampler(interval=60.0)
            sampler.start()
            resources = sampler.stop()

        assert resources["cpu_percent"]["mean"] == 25.0
        assert resources["memory_rss_mb"]["max"] == 100.0
        assert resources["gpu_percent"]["mean"] == 50.0
        assert resources["gpu_memory_mb"]["mean"] == 512.0
        assert resources["battery_used_percent"] == 2.0
        fake_pynvml.nvmlShutdown.assert_called_once()
        fake_pynvml.nvmlDeviceGetUtilizationRates.assert_not_called()
        fake_pynvml.nvmlDeviceGetMemoryInfo.assert_not_called()


def fire_relay_leg(reporter, engine, round_num, client, task_id=None, result=None):
    """Simulate a cyclic/relay leg: task assigned then completed, with NO acceptance events."""
    task_id = task_id or f"relay-{round_num}-{client}"
    task_data = Shareable()
    task_data.set_header(AppConstants.CURRENT_ROUND, round_num)
    reporter.handle_event(
        EventType.AFTER_TASK_DATA_FILTER,
        make_ctx(
            engine,
            peer_name=client,
            props={FLContextKey.TASK_NAME: "relay", FLContextKey.TASK_ID: task_id, FLContextKey.TASK_DATA: task_data},
        ),
    )
    reporter.handle_event(
        EventType.BEFORE_PROCESS_SUBMISSION,
        make_ctx(
            engine,
            peer_name=client,
            props={
                FLContextKey.TASK_NAME: "relay",
                FLContextKey.TASK_ID: task_id,
                FLContextKey.TASK_RESULT: result if result is not None else Shareable(),
            },
        ),
    )


class TestReviewFindingRegressions:
    def test_non_aggregation_relay_job_is_success(self, tmp_path):
        # F1: relay/cyclic workflows never fire acceptance events; completed OK tasks must
        # count as participation instead of the job being declared a FAILURE.
        clients = ["site-1", "site-2"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        for round_num in range(2):
            reporter.handle_event(
                AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: round_num})
            )
            for client in clients:
                fire_relay_leg(reporter, engine, round_num, client)
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        summary = reporter.get_summary(make_ctx(engine))
        assert summary["status"] == JobStatusCode.SUCCESS
        assert summary["participation"]["participation_rate_percent"] == 100.0
        assert summary["participation"]["participation_basis"] == "completed_tasks"

    def test_reconnected_client_does_not_downgrade_status(self, tmp_path):
        # F4: a transient disconnect followed by CLIENT_RECONNECTED must not force PARTIAL.
        clients = ["site-1", "site-2"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        reporter.handle_event(
            EventType.CLIENT_DISCONNECTED, make_ctx(engine, props={FLContextKey.DISCONNECTED_CLIENT_NAME: "site-2"})
        )
        reporter.handle_event(
            EventType.CLIENT_RECONNECTED, make_ctx(engine, props={FLContextKey.RECONNECTED_CLIENT_NAME: "site-2"})
        )
        fire_round(reporter, engine, 0, clients, make_aggr_stats(0, clients, 5, 5, 5, 0, 0))
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        summary = reporter.get_summary(make_ctx(engine))
        assert summary["status"] == JobStatusCode.SUCCESS
        assert summary["participation"]["disconnected_clients"] == []
        assert summary["errors"] == []

    def test_second_workflow_rounds_do_not_merge(self, tmp_path):
        # F5: rounds restarting at 0 in a later workflow must not merge with the first workflow's.
        clients = ["site-1"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        reporter.handle_event(EventType.START_WORKFLOW, make_ctx(engine))
        fire_round(reporter, engine, 0, clients, make_aggr_stats(0, clients, 5, 5, 5, 0, 0))
        reporter.handle_event(EventType.END_WORKFLOW, make_ctx(engine))
        reporter.handle_event(EventType.START_WORKFLOW, make_ctx(engine))
        fire_round(reporter, engine, 0, clients, make_aggr_stats(0, clients, 7, 7, 7, 0, 0))
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        summary = reporter.get_summary(make_ctx(engine))
        assert len(summary["rounds"]) == 2
        assert [r["keys_aggregated"] for r in summary["rounds"]] == [5, 7]
        assert summary["rounds"][0]["workflow"] != summary["rounds"][1]["workflow"]
        assert len(summary["consistency"]["workflows"]) == 2
        assert [item["stable_rounds"] for item in summary["consistency"]["workflows"]] == [1, 1]
        assert summary["consistency"]["inconsistent_rounds"] == []
        report = reporter.format_report(summary)
        assert "Workflow 1" in report
        assert "Workflow 2" in report
        assert "Workflow" in report.split("Aggregation Stats Summary", 1)[1]

    def test_malformed_telemetry_does_not_break_summary(self, tmp_path):
        # F6: a bad client telemetry header must degrade gracefully, not destroy the report.
        clients = ["site-1"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        result = Shareable()
        result.set_header(
            AppConstants.JOB_STATS_CLIENT_TELEMETRY,
            {
                "round": 0,
                "update_size_mb": "n/a",
                "execution_time": {"bad": "type"},
                "resources": {"battery_start_percent": 90.0, "cpu_percent": "oops"},
            },
        )
        reporter.handle_event(AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: 0}))
        fire_relay_leg(reporter, engine, 0, "site-1", result=result)
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        report_file = os.path.join(str(tmp_path), "job_stats_run_summary.log")
        assert os.path.isfile(report_file)
        summary = reporter.get_summary(make_ctx(engine))
        assert summary["status"] == JobStatusCode.SUCCESS
        task = summary["timing"]["tasks"][0]
        assert task["model_update_mb"] is not None  # falls back to server-side estimation
        assert task["client_computation_time"] is None

    def test_nonfinite_telemetry_is_dropped_and_json_remains_standard(self, tmp_path):
        clients = ["site-1"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        reporter.handle_event(AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: 0}))
        result = Shareable()
        result.set_header(
            AppConstants.JOB_STATS_CLIENT_TELEMETRY,
            {
                "round": 0,
                "execution_time": "nan",
                "update_size_mb": "inf",
                "resources": {
                    "cpu_percent": {"count": "inf", "mean": 25.0},
                    "memory_rss_mb": {"count": 1, "mean": "nan"},
                },
            },
        )
        fire_relay_leg(reporter, engine, 0, "site-1", result=result)
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        summary = reporter.get_summary(make_ctx(engine))
        assert summary["status"] == JobStatusCode.SUCCESS
        assert summary["participation"]["incomplete_tasks"] == 0
        assert summary["timing"]["tasks"][0]["client_computation_time"] is None
        with open(os.path.join(str(tmp_path), "job_stats_run_summary.json"), encoding="utf-8") as f:
            json.load(f, parse_constant=lambda value: pytest.fail(f"non-standard JSON constant: {value}"))

    def test_size_estimate_counts_repeated_shared_objects(self):
        # F7: [1] * n holds one shared int object; every occurrence must still be counted.
        from nvflare.app_common.widgets.job_stats_reporter import _estimate_size_bytes

        small = _estimate_size_bytes({"labels": [1]})
        big = _estimate_size_bytes({"labels": [1] * 1000})
        assert big > small * 100

    def test_round_zero_duplicate_submission_not_double_counted(self, tmp_path):
        # F14: telemetry round 0 must not be discarded as falsy, and a duplicate resend of an
        # already-processed result must not create a phantom task record.
        clients = ["site-1"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        reporter.handle_event(AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: 0}))
        result = Shareable()
        result.set_header(AppConstants.JOB_STATS_CLIENT_TELEMETRY, {"round": 0, "task_name": "train"})
        submission_ctx = dict(
            peer_name="site-1",
            props={FLContextKey.TASK_NAME: "train", FLContextKey.TASK_ID: "t0", FLContextKey.TASK_RESULT: result},
        )
        # unmatched submission (no assignment observed): must still attribute round 0 from telemetry
        reporter.handle_event(EventType.BEFORE_PROCESS_SUBMISSION, make_ctx(engine, **submission_ctx))
        # duplicate resend of the same task result
        reporter.handle_event(EventType.BEFORE_PROCESS_SUBMISSION, make_ctx(engine, **submission_ctx))
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        summary = reporter.get_summary(make_ctx(engine))
        assert len(summary["timing"]["tasks"]) == 1
        assert summary["timing"]["tasks"][0]["round"] == 0

    def test_client_rounds_sorted_numerically(self, tmp_path):
        # F15: rounds 0..11 must order numerically, not lexicographically (0,1,10,11,2,...).
        clients = ["site-1"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        for round_num in range(12):
            reporter.handle_event(
                AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: round_num})
            )
            fire_relay_leg(reporter, engine, round_num, "site-1")
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        summary = reporter.get_summary(make_ctx(engine))
        assert [row["round"] for row in summary["timing"]["client_rounds"]] == list(range(12))
        assert [row["round"] for row in summary["rounds"]] == list(range(12))

    def test_final_aggregation_skips_round_without_stats(self, tmp_path):
        # F13: an aborted/unfinished last round without aggregation stats must not be the Final row.
        clients = ["site-1"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        fire_round(reporter, engine, 0, clients, make_aggr_stats(0, clients, 5, 5, 5, 0, 0))
        # round 1 starts and gets a task assignment but never aggregates
        reporter.handle_event(AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: 1}))
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        summary = reporter.get_summary(make_ctx(engine))
        assert summary["final_aggregation"]["source_round"] == 0
        assert summary["final_aggregation"]["keys_aggregated"] == 5

    def test_task_assignment_ignores_stale_sticky_round(self, tmp_path):
        # F3: a task whose data carries no round must not inherit the sticky CURRENT_ROUND of a
        # previous training workflow once that round is closed.
        clients = ["site-1"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        fire_round(reporter, engine, 3, clients, make_aggr_stats(3, clients, 5, 5, 5, 0, 0))
        # cross-site-eval style task: no round header/cookie, but ctx still has sticky round 3
        reporter.handle_event(
            EventType.AFTER_TASK_DATA_FILTER,
            make_ctx(
                engine,
                peer_name="site-1",
                props={
                    FLContextKey.TASK_NAME: "validate",
                    FLContextKey.TASK_ID: "eval-1",
                    FLContextKey.TASK_DATA: Shareable(),
                    AppConstants.CURRENT_ROUND: 3,
                },
            ),
        )
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        summary = reporter.get_summary(make_ctx(engine))
        # round 3 must not gain the eval task as a targeted-but-missing contribution
        assert summary["status"] == JobStatusCode.PARTIAL  # only due to the never-completed eval task
        eval_tasks = [t for t in summary["timing"]["tasks"] if t["task_name"] == "validate"]
        assert eval_tasks[0]["round"] is None


class TestSecondReviewRegressions:
    def test_report_filenames_must_stay_under_run_dir(self):
        # absolute or traversal filenames must be rejected at config time, not silently escape
        with pytest.raises(ValueError):
            JobStatsReporter(filename="/tmp/steal.log")
        with pytest.raises(ValueError):
            JobStatsReporter(error_filename="../outside/errors.log")
        JobStatsReporter(filename="reports/summary.log")  # relative subdir is fine

    def test_malformed_aggregation_stats_degrade_gracefully(self, tmp_path):
        # string CONTRIBUTORS must not explode into per-character clients, and non-numeric
        # counts must not abort report generation
        clients = ["site-1"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        reporter.handle_event(AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: 0}))
        fire_relay_leg(reporter, engine, 0, "site-1")
        bad_stats = {
            AggregationStatsKey.ROUND: 0,
            AggregationStatsKey.CONTRIBUTORS: "site-1",  # bare string, not a list
            AggregationStatsKey.KEYS_AGGREGATED: "N/A",  # non-numeric
            AggregationStatsKey.FULLY_MATCHED_KEYS: [1, 2],  # unhashable if used raw
        }
        reporter.handle_event(
            AppEventType.AFTER_AGGREGATION,
            make_ctx(engine, props={AppConstants.CURRENT_ROUND: 0, AppConstants.AGGREGATION_STATS: bad_stats}),
        )
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        assert os.path.isfile(os.path.join(str(tmp_path), "job_stats_run_summary.log"))
        summary = reporter.get_summary(make_ctx(engine))
        assert summary["status"] == JobStatusCode.SUCCESS
        assert summary["rounds"][0]["accepted_client_names"] == ["site-1"]  # not ['1','e','i','s','t','-']
        assert summary["rounds"][0]["keys_aggregated"] is None
        report = reporter.format_report(summary)
        assert "Aggregation Consistency Across Rounds" in report

    def test_stats_without_contributor_names_fall_back_to_completed_tasks(self, tmp_path):
        # a custom aggregator publishing counts but no names must not zero out participation
        clients = ["site-1", "site-2"]
        engine = make_engine(tmp_path, clients)
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        reporter.handle_event(AppEventType.ROUND_STARTED, make_ctx(engine, props={AppConstants.CURRENT_ROUND: 0}))
        for client in clients:
            fire_relay_leg(reporter, engine, 0, client)
        stats = make_aggr_stats(0, [], 5, 5, 5, 0, 0)
        stats[AggregationStatsKey.ACCEPTED_CONTRIBUTIONS] = 2  # counts known, names unknown
        reporter.handle_event(
            AppEventType.AFTER_AGGREGATION,
            make_ctx(engine, props={AppConstants.CURRENT_ROUND: 0, AppConstants.AGGREGATION_STATS: stats}),
        )
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        summary = reporter.get_summary(make_ctx(engine))
        assert summary["status"] == JobStatusCode.SUCCESS
        assert summary["participation"]["participation_rate_percent"] == 100.0

    def test_summary_generation_failure_still_writes_report(self, tmp_path):
        # a bug in summary generation must degrade the report, never lose it entirely
        engine = make_engine(tmp_path, ["site-1"])
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        reporter.get_summary = Mock(side_effect=RuntimeError("boom"))
        reporter.handle_event(EventType.END_RUN, make_ctx(engine))

        report_file = os.path.join(str(tmp_path), "job_stats_run_summary.log")
        assert os.path.isfile(report_file)
        with open(report_file, encoding="utf-8") as f:
            report = f.read()
        assert "Report generation failed" in report
        assert "RuntimeError" in report
        assert not os.path.isfile(os.path.join(str(tmp_path), "job_stats_run_summary.json"))

    def test_bool_telemetry_values_are_dropped(self):
        from nvflare.app_common.widgets.job_stats_reporter import _sanitize_telemetry

        telemetry = _sanitize_telemetry(
            {
                "execution_time": True,
                "update_size_mb": False,
                "resources": {
                    "cpu_percent": {"count": True, "mean": True},
                    "battery_start_percent": True,
                    "battery_plugged": True,
                },
            }
        )
        assert "execution_time" not in telemetry
        assert "update_size_mb" not in telemetry
        assert "cpu_percent" not in telemetry["resources"]
        assert "battery_start_percent" not in telemetry["resources"]
        # battery_plugged is a genuine boolean field and must pass through
        assert telemetry["resources"]["battery_plugged"] is True

    def test_lazy_ref_payload_reports_unknown_size_not_zero(self, tmp_path):
        from nvflare.app_common.widgets.job_stats_reporter import _estimate_payload_mb

        class _FakeLazyRef:
            pass  # exposes neither size nor file_path

        # unknown -> None (rendered N/A), never a misleading few-bytes estimate
        assert _estimate_payload_mb({"w": _FakeLazyRef()}) is None

        class _FakeFileLazyRef:
            def __init__(self, file_path):
                self.file_path = file_path

        blob = tmp_path / "tensor.bin"
        blob.write_bytes(b"x" * 2048)
        size_mb = _estimate_payload_mb({"w": _FakeFileLazyRef(str(blob))})
        # 2048 bytes from the file stat + 1 byte for the "w" key
        assert size_mb == pytest.approx(2049 / (1024.0 * 1024.0))

    def test_reconnect_scrubs_client_errors_entry(self, tmp_path):
        # disconnects now record into _client_errors; reconnect must scrub them symmetrically
        engine = make_engine(tmp_path, ["site-1", "site-2"])
        reporter = JobStatsReporter()
        reporter.handle_event(EventType.START_RUN, make_ctx(engine))
        reporter.handle_event(
            EventType.CLIENT_DISCONNECTED, make_ctx(engine, props={FLContextKey.DISCONNECTED_CLIENT_NAME: "site-2"})
        )
        summary = reporter.get_summary(make_ctx(engine))
        assert summary["participation"]["client_errors"] == {"site-2": ["CLIENT_DISCONNECTED"]}

        reporter.handle_event(
            EventType.CLIENT_RECONNECTED, make_ctx(engine, props={FLContextKey.RECONNECTED_CLIENT_NAME: "site-2"})
        )
        summary = reporter.get_summary(make_ctx(engine))
        assert summary["participation"]["client_errors"] == {}
        assert summary["errors"] == []
        assert summary["participation"]["failed_clients"] == []

    def test_gpu_sampler_includes_child_process_usage(self):
        # subprocess-mode training does GPU work in a launched child pid
        child_pid = 4242
        process = MagicMock()
        process.cpu_percent.side_effect = [0.0, 25.0]
        process.memory_info.return_value = SimpleNamespace(rss=100 * 1024 * 1024)
        process.children.return_value = [SimpleNamespace(pid=child_pid)]
        fake_psutil = SimpleNamespace(Process=Mock(return_value=process), sensors_battery=Mock(return_value=None))
        fake_pynvml = SimpleNamespace(
            nvmlInit=Mock(),
            nvmlShutdown=Mock(),
            nvmlDeviceGetCount=Mock(return_value=1),
            nvmlDeviceGetHandleByIndex=Mock(return_value="gpu-0"),
            nvmlDeviceGetProcessUtilization=Mock(
                return_value=[SimpleNamespace(pid=child_pid, smUtil=40.0, timeStamp=1)]
            ),
            nvmlDeviceGetComputeRunningProcesses=Mock(
                return_value=[SimpleNamespace(pid=child_pid, usedGpuMemory=512 * 1024 * 1024)]
            ),
        )

        with patch.dict("sys.modules", {"psutil": fake_psutil, "pynvml": fake_pynvml}):
            sampler = _ResourceSampler(interval=60.0)
            sampler.start()
            resources = sampler.stop()

        assert resources["gpu_percent"]["mean"] == 40.0
        assert resources["gpu_memory_mb"]["mean"] == 512.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
