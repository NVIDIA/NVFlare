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

import importlib.util
from pathlib import Path


def _load_gate_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "examples" / "advanced" / "monitoring" / "sovereign" / "nvflare_quantum_readiness_gate.py"
    spec = importlib.util.spec_from_file_location("nvflare_quantum_readiness_gate", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_targets_reports_missing_instance(monkeypatch):
    gate = _load_gate_module()

    payload = {
        "data": {
            "activeTargets": [
                {"health": "up", "labels": {"instance": "statsd-exporter:9102"}},
                {"health": "down", "labels": {"instance": "other:9100"}},
            ]
        }
    }
    monkeypatch.setattr(gate, "fetch_json", lambda url: payload)

    failures = gate.check_targets("http://localhost:9090", ["statsd-exporter:9102", "other:9100"])
    assert len(failures) == 1
    assert "other:9100" in failures[0]


def test_check_metric_names_reports_missing_metric(monkeypatch):
    gate = _load_gate_module()
    monkeypatch.setattr(gate, "fetch_json", lambda url: {"data": ["quantum_path_ready"]})

    failures = gate.check_metric_names("http://localhost:9090", ["quantum_path_ready", "quantum_proof_verify_count"])
    assert failures == ["required metric missing: quantum_proof_verify_count"]


def test_check_quantum_path_detects_empty_query(monkeypatch):
    gate = _load_gate_module()

    def _query(prom_url, expr):
        if "quantum_proof_verify_count" in expr:
            return []
        return [{"metric": {}, "value": [0, "1"]}]

    monkeypatch.setattr(gate, "query_vector", _query)

    failures = gate.check_quantum_path("http://localhost:9090")
    assert any("proof_verify_requests" in f for f in failures)


def test_main_returns_pass_and_writes_report(monkeypatch, tmp_path):
    gate = _load_gate_module()

    monkeypatch.setattr(
        gate,
        "wait_json",
        lambda url, expect_key, retries, delay_seconds: {"status": "success", "database": "ok"},
    )
    monkeypatch.setattr(gate, "check_targets", lambda prom_url, required_instances: [])
    monkeypatch.setattr(gate, "check_metric_names", lambda prom_url, required_metrics: [])
    monkeypatch.setattr(gate, "check_quantum_path", lambda prom_url: [])
    monkeypatch.setattr(gate, "check_congestion_risk", lambda *args, **kwargs: [])

    out_file = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "nvflare_quantum_readiness_gate.py",
            "--output",
            str(out_file),
        ],
    )

    rc = gate.main()
    assert rc == 0
    assert out_file.exists()
    assert '"status": "PASS"' in out_file.read_text(encoding="utf-8")


def test_build_basic_auth_header():
    gate = _load_gate_module()
    auth_header = gate.build_basic_auth_header("user", "pass")
    assert auth_header == "Basic dXNlcjpwYXNz"


def test_check_congestion_risk_reports_threshold_breach(monkeypatch):
    gate = _load_gate_module()

    def _query_scalar(prom_url, expr, auth_header=""):
        if "verify_count" in expr:
            return 10.0
        if "failure_count" in expr:
            return 2.0
        if "verify_time_taken" in expr:
            return 0.1
        if "aggregation_time_taken" in expr:
            return 0.8
        return 0.0

    monkeypatch.setattr(gate, "query_scalar", _query_scalar)
    failures = gate.check_congestion_risk("http://localhost:9090", max_failure_ratio=0.1, max_latency_ratio=4.0)
    assert len(failures) == 2
    assert any("failure ratio too high" in f for f in failures)
    assert any("latency ratio too high" in f for f in failures)
