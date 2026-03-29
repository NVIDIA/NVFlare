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

from nvflare.apis.event_type import EventType
from nvflare.app_common.app_event_type import AppEventType
from nvflare.metrics.quantum_proof_metrics_collector import QuantumProofMetricsCollector


class _Ctx:
    def __init__(self, job_id="job-1"):
        self._job_id = job_id

    def get_job_id(self):
        return self._job_id


def test_start_run_publishes_quantum_posture(monkeypatch):
    calls = []

    def _capture(comp, streaming_to_server, metrics, metric_name, tags, data_bus, fl_ctx):
        calls.append({"metric_name": metric_name, "metrics": metrics, "tags": tags})

    monkeypatch.setattr("nvflare.metrics.quantum_proof_metrics_collector.collect_metrics", _capture)

    collector = QuantumProofMetricsCollector(
        tags={"site": "server"},
        expected_attestation_mode="xmss",
        expected_kex_mode="x25519-mlkem768-hybrid",
        pqc_migration_enabled=True,
        pqc_lock_legacy_transfers=True,
    )

    collector.handle_event(EventType.START_RUN, _Ctx())

    names = [c["metric_name"] for c in calls]
    assert "quantum_path" in names
    assert "quantum_pqc_controls" in names

    posture = [c for c in calls if c["metric_name"] == "quantum_path"][0]
    assert posture["metrics"]["ready"] == 1
    assert posture["tags"]["attestation_mode"] == "xmss"
    assert posture["tags"]["kex_mode"] == "x25519-mlkem768-hybrid"


def test_proof_flow_publishes_counts_and_duration(monkeypatch):
    calls = []

    def _capture(comp, streaming_to_server, metrics, metric_name, tags, data_bus, fl_ctx):
        calls.append({"metric_name": metric_name, "metrics": metrics, "tags": tags})

    monkeypatch.setattr("nvflare.metrics.quantum_proof_metrics_collector.collect_metrics", _capture)

    collector = QuantumProofMetricsCollector(tags={"site": "site-1"})
    ctx = _Ctx("job-xyz")

    collector.handle_event(EventType.BEFORE_TASK_EXECUTION, ctx)
    collector.handle_event(EventType.AFTER_TASK_EXECUTION, ctx)
    collector.handle_event(EventType.ABORT_TASK, ctx)

    names = [c["metric_name"] for c in calls]
    assert "quantum_proof_verify" in names
    assert "quantum_proof_verify_success" in names
    assert "quantum_proof_verify_failure" in names

    duration = [c for c in calls if c["metric_name"] == "quantum_proof_verify" and "time_taken" in c["metrics"]]
    assert len(duration) == 1
    assert duration[0]["tags"]["job_id"] == "job-xyz"


def test_aggregation_flow_publishes_metrics(monkeypatch):
    calls = []

    def _capture(comp, streaming_to_server, metrics, metric_name, tags, data_bus, fl_ctx):
        calls.append({"metric_name": metric_name, "metrics": metrics})

    monkeypatch.setattr("nvflare.metrics.quantum_proof_metrics_collector.collect_metrics", _capture)

    collector = QuantumProofMetricsCollector(tags={"site": "server"})
    ctx = _Ctx()

    collector.handle_event(AppEventType.BEFORE_AGGREGATION, ctx)
    collector.handle_event(AppEventType.END_AGGREGATION, ctx)

    names = [c["metric_name"] for c in calls]
    assert "quantum_proof_aggregation" in names
    assert "quantum_proof_aggregation_success" in names
    assert any(c["metric_name"] == "quantum_proof_aggregation" and "time_taken" in c["metrics"] for c in calls)


def test_abort_task_clears_proof_timer(monkeypatch):
    calls = []

    def _capture(comp, streaming_to_server, metrics, metric_name, tags, data_bus, fl_ctx):
        calls.append({"metric_name": metric_name, "metrics": metrics})

    monkeypatch.setattr("nvflare.metrics.quantum_proof_metrics_collector.collect_metrics", _capture)

    collector = QuantumProofMetricsCollector(tags={"site": "site-1"})
    ctx = _Ctx("job-xyz")

    collector.handle_event(EventType.BEFORE_TASK_EXECUTION, ctx)
    collector.handle_event(EventType.ABORT_TASK, ctx)
    collector.handle_event(EventType.AFTER_TASK_EXECUTION, ctx)

    verify_elapsed = [c for c in calls if c["metric_name"] == "quantum_proof_verify" and "time_taken" in c["metrics"]]
    assert len(verify_elapsed) == 0
