#!/usr/bin/env python3
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

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List


def fetch_json(url: str, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_json(url: str, expect_key: str, retries: int, delay_seconds: float) -> dict:
    last_error = None
    for _ in range(retries):
        try:
            payload = fetch_json(url)
            if not expect_key:
                return payload
            if expect_key in payload:
                return payload
            last_error = f"missing key {expect_key!r} in response from {url}"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            last_error = str(e)
        time.sleep(delay_seconds)
    raise RuntimeError(last_error or f"failed waiting for {url}")


def query_vector(prom_url: str, expr: str) -> List[dict]:
    query = urllib.parse.urlencode({"query": expr})
    payload = fetch_json(f"{prom_url}/api/v1/query?{query}")
    if payload.get("status") != "success":
        return []
    return payload.get("data", {}).get("result", [])


def check_targets(prom_url: str, required_instances: List[str]) -> List[str]:
    payload = fetch_json(f"{prom_url}/api/v1/targets")
    active_targets = payload.get("data", {}).get("activeTargets", [])

    healthy = set()
    for t in active_targets:
        if t.get("health") == "up":
            labels = t.get("labels") or {}
            instance = labels.get("instance") or t.get("scrapeUrl", "")
            healthy.add(instance)

    failures = []
    for inst in required_instances:
        if inst not in healthy:
            failures.append(f"required target is not healthy: {inst}")
    return failures


def check_metric_names(prom_url: str, required_metrics: List[str]) -> List[str]:
    payload = fetch_json(f"{prom_url}/api/v1/label/__name__/values")
    metric_names = set(payload.get("data", []))

    failures = []
    for metric_name in required_metrics:
        if metric_name not in metric_names:
            failures.append(f"required metric missing: {metric_name}")
    return failures


def check_quantum_path(prom_url: str) -> List[str]:
    checks = {
        "quantum_path_ready": "max(quantum_path_ready)",
        "pqc_controls_enabled": "min(quantum_pqc_controls_migration_enabled * quantum_pqc_controls_legacy_lock_enabled)",
        "proof_verify_requests": "sum(rate(quantum_proof_verify_count[5m]))",
    }

    failures = []
    for check_name, expr in checks.items():
        try:
            result = query_vector(prom_url, expr)
        except Exception as e:
            failures.append(f"query failed for {check_name}: {e}")
            continue

        if not result:
            failures.append(f"query returned no data for {check_name}: {expr}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Readiness gate for NVFlare Sovereign quantum-proof observability profile."
    )
    parser.add_argument("--prom-url", default="http://localhost:9090", help="Prometheus base URL")
    parser.add_argument("--grafana-url", default="http://localhost:3000", help="Grafana base URL")
    parser.add_argument(
        "--required-target",
        action="append",
        dest="required_targets",
        default=["statsd-exporter:9102"],
        help="Prometheus target instance required to be healthy (repeatable).",
    )
    parser.add_argument(
        "--required-metric",
        action="append",
        dest="required_metrics",
        default=[
            "quantum_path_ready",
            "quantum_pqc_controls_migration_enabled",
            "quantum_pqc_controls_legacy_lock_enabled",
            "quantum_proof_verify_count",
            "quantum_proof_verify_success_count",
            "quantum_proof_verify_time_taken",
        ],
        help="Metric name required to exist in Prometheus label set (repeatable).",
    )
    parser.add_argument("--retries", type=int, default=30, help="Retries per health check")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between retries")
    parser.add_argument("--output", default="-", help="Output report path, or '-' for stdout")
    args = parser.parse_args()

    report = {
        "profile": "nvflare-sovereign-quantum",
        "checks": {},
        "details": {
            "required_targets": args.required_targets,
            "required_metrics": args.required_metrics,
        },
    }

    failures = []

    try:
        _ = wait_json(
            f"{args.prom_url}/api/v1/status/buildinfo",
            expect_key="status",
            retries=args.retries,
            delay_seconds=args.delay,
        )
        report["checks"]["prometheus_health"] = True
    except Exception as e:
        report["checks"]["prometheus_health"] = False
        failures.append(f"prometheus health check failed: {e}")

    try:
        grafana_health = wait_json(
            f"{args.grafana_url}/api/health",
            expect_key="database",
            retries=args.retries,
            delay_seconds=args.delay,
        )
        report["checks"]["grafana_health"] = grafana_health.get("database") == "ok"
        if not report["checks"]["grafana_health"]:
            failures.append(f"grafana database not ok: {grafana_health}")
    except Exception as e:
        report["checks"]["grafana_health"] = False
        failures.append(f"grafana health check failed: {e}")

    try:
        target_failures = check_targets(args.prom_url, args.required_targets)
        report["checks"]["targets_up"] = len(target_failures) == 0
        failures.extend(target_failures)
    except Exception as e:
        report["checks"]["targets_up"] = False
        failures.append(f"target checks failed: {e}")

    try:
        metric_failures = check_metric_names(args.prom_url, args.required_metrics)
        report["checks"]["required_metrics_present"] = len(metric_failures) == 0
        failures.extend(metric_failures)
    except Exception as e:
        report["checks"]["required_metrics_present"] = False
        failures.append(f"metric checks failed: {e}")

    try:
        quantum_failures = check_quantum_path(args.prom_url)
        report["checks"]["quantum_path_queries"] = len(quantum_failures) == 0
        failures.extend(quantum_failures)
    except Exception as e:
        report["checks"]["quantum_path_queries"] = False
        failures.append(f"quantum path checks failed: {e}")

    report["status"] = "PASS" if len(failures) == 0 else "FAIL"
    report["failures"] = failures

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)

    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
