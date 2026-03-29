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
import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

CONGESTION_PROFILES = {
    "dev": {"max_failure_ratio": 0.20, "max_latency_ratio": 10.0},
    "staging": {"max_failure_ratio": 0.10, "max_latency_ratio": 6.0},
    "prod": {"max_failure_ratio": 0.05, "max_latency_ratio": 3.0},
}


def build_basic_auth_header(username: str = "", password: str = "") -> str:
    if not username:
        return ""
    token = base64.b64encode(f"{username}:{password or ''}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def fetch_json(url: str, timeout: float = 5.0, auth_header: str = "") -> dict:
    req = urllib.request.Request(url, method="GET")
    if auth_header:
        req.add_header("Authorization", auth_header)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_json(url: str, expect_key: str, retries: int, delay_seconds: float, auth_header: str = "") -> dict:
    last_error = None
    for _ in range(retries):
        try:
            if auth_header:
                payload = fetch_json(url, auth_header=auth_header)
            else:
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


def query_vector(prom_url: str, expr: str, auth_header: str = "", query_time: Optional[float] = None) -> List[dict]:
    params = {"query": expr}
    if query_time is not None:
        params["time"] = str(query_time)
    query = urllib.parse.urlencode(params)
    if auth_header:
        payload = fetch_json(f"{prom_url}/api/v1/query?{query}", auth_header=auth_header)
    else:
        payload = fetch_json(f"{prom_url}/api/v1/query?{query}")
    if payload.get("status") != "success":
        return []
    return payload.get("data", {}).get("result", [])


def query_scalar(prom_url: str, expr: str, auth_header: str = "", query_time: Optional[float] = None) -> float:
    result = query_vector(prom_url, expr, auth_header=auth_header, query_time=query_time)
    if not result:
        return 0.0
    return float(result[0]["value"][1])


def resolve_thresholds(profile: str, max_failure_ratio: float, max_latency_ratio: float) -> tuple[float, float]:
    if profile in CONGESTION_PROFILES:
        p = CONGESTION_PROFILES[profile]
        return p["max_failure_ratio"], p["max_latency_ratio"]
    return max_failure_ratio, max_latency_ratio


def check_targets(prom_url: str, required_instances: List[str], auth_header: str = "") -> List[str]:
    if auth_header:
        payload = fetch_json(f"{prom_url}/api/v1/targets", auth_header=auth_header)
    else:
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


def check_metric_names(prom_url: str, required_metrics: List[str], auth_header: str = "") -> List[str]:
    if auth_header:
        payload = fetch_json(f"{prom_url}/api/v1/label/__name__/values", auth_header=auth_header)
    else:
        payload = fetch_json(f"{prom_url}/api/v1/label/__name__/values")
    metric_names = set(payload.get("data", []))

    failures = []
    for metric_name in required_metrics:
        if metric_name not in metric_names:
            failures.append(f"required metric missing: {metric_name}")
    return failures


def check_quantum_path(prom_url: str, auth_header: str = "") -> List[str]:
    checks = {
        "quantum_path_ready": "max(quantum_path_ready)",
        "pqc_controls_enabled": "min(quantum_pqc_controls_migration_enabled * quantum_pqc_controls_legacy_lock_enabled)",
        "proof_verify_requests": "sum(rate(quantum_proof_verify_count[5m]))",
    }

    failures = []
    for check_name, expr in checks.items():
        try:
            if auth_header:
                result = query_vector(prom_url, expr, auth_header=auth_header)
            else:
                result = query_vector(prom_url, expr)
        except Exception as e:
            failures.append(f"query failed for {check_name}: {e}")
            continue

        if not result:
            failures.append(f"query returned no data for {check_name}: {expr}")
    return failures


def check_congestion_risk(
    prom_url: str,
    max_failure_ratio: float,
    max_latency_ratio: float,
    consecutive_breaches: int = 1,
    sample_step_minutes: int = 5,
    auth_header: str = "",
) -> List[str]:
    failures = []

    consecutive_breaches = max(1, consecutive_breaches)
    sample_step_minutes = max(1, sample_step_minutes)

    failure_breaches = 0
    latency_breaches = 0
    latest_failure_ratio = 0.0
    latest_latency_ratio = 0.0

    for i in range(consecutive_breaches):
        sample_time = time.time() - (i * sample_step_minutes * 60)

        verify_rate = query_scalar(
            prom_url,
            "sum(rate(quantum_proof_verify_count[5m]))",
            auth_header=auth_header,
            query_time=sample_time,
        )
        failure_rate = query_scalar(
            prom_url,
            "sum(rate(quantum_proof_verify_failure_count[5m]))",
            auth_header=auth_header,
            query_time=sample_time,
        )
        verify_latency = query_scalar(
            prom_url,
            "max(avg_over_time(quantum_proof_verify_time_taken[5m]) or vector(0))",
            auth_header=auth_header,
            query_time=sample_time,
        )
        agg_latency = query_scalar(
            prom_url,
            "max(avg_over_time(quantum_proof_aggregation_time_taken[5m]) or vector(0))",
            auth_header=auth_header,
            query_time=sample_time,
        )

        if verify_rate <= 0:
            continue

        failure_ratio = failure_rate / max(verify_rate, 1e-9)
        latency_ratio = 0.0
        if verify_latency > 0:
            latency_ratio = agg_latency / verify_latency

        if i == 0:
            latest_failure_ratio = failure_ratio
            latest_latency_ratio = latency_ratio

        if failure_ratio > max_failure_ratio:
            failure_breaches += 1
        if latency_ratio > max_latency_ratio:
            latency_breaches += 1

    if failure_breaches >= consecutive_breaches:
        failures.append(
            "failure ratio too high for "
            f"{consecutive_breaches} consecutive samples: latest={latest_failure_ratio:.4f}, "
            f"threshold={max_failure_ratio:.4f}"
        )

    if latency_breaches >= consecutive_breaches:
        failures.append(
            "aggregation/verify latency ratio too high for "
            f"{consecutive_breaches} consecutive samples: latest={latest_latency_ratio:.4f}, "
            f"threshold={max_latency_ratio:.4f}"
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Readiness gate for NVFlare Sovereign quantum-proof observability profile."
    )
    parser.add_argument("--prom-url", default="http://localhost:9090", help="Prometheus base URL")
    parser.add_argument("--grafana-url", default="http://localhost:3000", help="Grafana base URL")
    parser.add_argument("--prom-user", default="", help="Prometheus basic-auth username (optional)")
    parser.add_argument("--prom-password", default="", help="Prometheus basic-auth password (optional)")
    parser.add_argument("--grafana-user", default="", help="Grafana basic-auth username (optional)")
    parser.add_argument("--grafana-password", default="", help="Grafana basic-auth password (optional)")
    parser.add_argument(
        "--profile",
        choices=["custom", "dev", "staging", "prod"],
        default="custom",
        help="Congestion profile preset. If set, profile thresholds override explicit max-ratio args.",
    )
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
    parser.add_argument(
        "--max-failure-ratio",
        type=float,
        default=0.10,
        help="Maximum allowed failure ratio over 5m before congestion check fails.",
    )
    parser.add_argument(
        "--max-latency-ratio",
        type=float,
        default=4.0,
        help="Maximum allowed aggregation/verify latency ratio over 5m before congestion check fails.",
    )
    parser.add_argument(
        "--consecutive-breaches",
        type=int,
        default=1,
        help="Require this many consecutive breached samples before congestion check fails.",
    )
    parser.add_argument(
        "--congestion-sample-step-minutes",
        type=int,
        default=5,
        help="Minutes between historical samples used for consecutive breach detection.",
    )
    parser.add_argument("--output", default="-", help="Output report path, or '-' for stdout")
    args = parser.parse_args()

    effective_failure_ratio, effective_latency_ratio = resolve_thresholds(
        args.profile,
        args.max_failure_ratio,
        args.max_latency_ratio,
    )

    prom_url = args.prom_url
    grafana_url = args.grafana_url
    prom_auth = build_basic_auth_header(args.prom_user, args.prom_password)
    grafana_auth = build_basic_auth_header(args.grafana_user, args.grafana_password)

    report = {
        "profile": "nvflare-sovereign-quantum",
        "checks": {},
        "details": {
            "required_targets": args.required_targets,
            "required_metrics": args.required_metrics,
            "profile": args.profile,
            "max_failure_ratio": effective_failure_ratio,
            "max_latency_ratio": effective_latency_ratio,
            "consecutive_breaches": args.consecutive_breaches,
            "congestion_sample_step_minutes": args.congestion_sample_step_minutes,
        },
    }

    failures = []

    try:
        if prom_auth:
            _ = wait_json(
                f"{prom_url}/api/v1/status/buildinfo",
                expect_key="status",
                retries=args.retries,
                delay_seconds=args.delay,
                auth_header=prom_auth,
            )
        else:
            _ = wait_json(
                f"{prom_url}/api/v1/status/buildinfo",
                expect_key="status",
                retries=args.retries,
                delay_seconds=args.delay,
            )
        report["checks"]["prometheus_health"] = True
    except Exception as e:
        report["checks"]["prometheus_health"] = False
        failures.append(f"prometheus health check failed: {e}")

    try:
        if grafana_auth:
            grafana_health = wait_json(
                f"{grafana_url}/api/health",
                expect_key="database",
                retries=args.retries,
                delay_seconds=args.delay,
                auth_header=grafana_auth,
            )
        else:
            grafana_health = wait_json(
                f"{grafana_url}/api/health",
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
        if prom_auth:
            target_failures = check_targets(prom_url, args.required_targets, auth_header=prom_auth)
        else:
            target_failures = check_targets(prom_url, args.required_targets)
        report["checks"]["targets_up"] = len(target_failures) == 0
        failures.extend(target_failures)
    except Exception as e:
        report["checks"]["targets_up"] = False
        failures.append(f"target checks failed: {e}")

    try:
        if prom_auth:
            metric_failures = check_metric_names(prom_url, args.required_metrics, auth_header=prom_auth)
        else:
            metric_failures = check_metric_names(prom_url, args.required_metrics)
        report["checks"]["required_metrics_present"] = len(metric_failures) == 0
        failures.extend(metric_failures)
    except Exception as e:
        report["checks"]["required_metrics_present"] = False
        failures.append(f"metric checks failed: {e}")

    try:
        if prom_auth:
            quantum_failures = check_quantum_path(prom_url, auth_header=prom_auth)
        else:
            quantum_failures = check_quantum_path(prom_url)
        report["checks"]["quantum_path_queries"] = len(quantum_failures) == 0
        failures.extend(quantum_failures)
    except Exception as e:
        report["checks"]["quantum_path_queries"] = False
        failures.append(f"quantum path checks failed: {e}")

    try:
        if prom_auth:
            congestion_failures = check_congestion_risk(
                prom_url,
                effective_failure_ratio,
                effective_latency_ratio,
                consecutive_breaches=args.consecutive_breaches,
                sample_step_minutes=args.congestion_sample_step_minutes,
                auth_header=prom_auth,
            )
        else:
            congestion_failures = check_congestion_risk(
                prom_url,
                effective_failure_ratio,
                effective_latency_ratio,
                consecutive_breaches=args.consecutive_breaches,
                sample_step_minutes=args.congestion_sample_step_minutes,
            )
        report["checks"]["congestion_within_threshold"] = len(congestion_failures) == 0
        failures.extend(congestion_failures)
    except Exception as e:
        report["checks"]["congestion_within_threshold"] = False
        failures.append(f"congestion checks failed: {e}")

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
