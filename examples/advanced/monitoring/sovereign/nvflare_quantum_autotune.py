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
import math
import time
import urllib.parse
import urllib.request
from typing import List

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


def fetch_json(url: str, auth_header: str = "", timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    if auth_header:
        req.add_header("Authorization", auth_header)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def query_range(prom_url: str, expr: str, start: int, end: int, step: int, auth_header: str = "") -> List[float]:
    params = urllib.parse.urlencode({"query": expr, "start": start, "end": end, "step": step})
    payload = fetch_json(f"{prom_url}/api/v1/query_range?{params}", auth_header=auth_header)
    if payload.get("status") != "success":
        return []
    result = payload.get("data", {}).get("result", [])
    if not result:
        return []

    values = []
    for item in result:
        for _, v in item.get("values", []):
            try:
                values.append(float(v))
            except Exception:
                continue
    return values


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    arr = sorted(values)
    k = (len(arr) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    d0 = arr[f] * (c - k)
    d1 = arr[c] * (k - f)
    return d0 + d1


def summarize(values: List[float]) -> dict:
    return {
        "samples": len(values),
        "p50": round(percentile(values, 50), 6),
        "p90": round(percentile(values, 90), 6),
        "p95": round(percentile(values, 95), 6),
        "p99": round(percentile(values, 99), 6),
        "max": round(max(values), 6) if values else 0.0,
    }


def recommend_profile(max_failure_ratio: float, max_latency_ratio: float) -> str:
    # Pick the strictest preset that still covers the recommended thresholds.
    for profile in ["prod", "staging", "dev"]:
        p = CONGESTION_PROFILES[profile]
        if p["max_failure_ratio"] >= max_failure_ratio and p["max_latency_ratio"] >= max_latency_ratio:
            return profile
    return "custom"


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-tune congestion thresholds from Prometheus history.")
    parser.add_argument("--prom-url", default="http://localhost:9090", help="Prometheus base URL")
    parser.add_argument("--prom-user", default="", help="Prometheus basic-auth username")
    parser.add_argument("--prom-password", default="", help="Prometheus basic-auth password")
    parser.add_argument("--lookback-minutes", type=int, default=60, help="History window in minutes")
    parser.add_argument("--step-seconds", type=int, default=30, help="Query range step in seconds")
    parser.add_argument(
        "--min-throughput-rps",
        type=float,
        default=1.0,
        help="Ignore low-traffic samples below this verify throughput",
    )
    parser.add_argument(
        "--headroom-factor",
        type=float,
        default=1.2,
        help="Multiply p95 by this factor for threshold recommendation",
    )
    parser.add_argument(
        "--print-shell-export",
        action="store_true",
        help="Print shell export lines for CI (stdout). If set, JSON still writes to --output when --output is a file.",
    )
    parser.add_argument("--output", default="-", help="Output file path or '-' for stdout")
    args = parser.parse_args()

    auth_header = build_basic_auth_header(args.prom_user, args.prom_password)
    end = int(time.time())
    start = end - (args.lookback_minutes * 60)

    verify_rate_expr = "max(sum(rate(quantum_proof_verify_count[5m])) or vector(0))"
    failure_ratio_expr = "max((sum(rate(quantum_proof_verify_failure_count[5m])) / clamp_min(sum(rate(quantum_proof_verify_count[5m])), 1)) or vector(0))"
    latency_ratio_expr = "max((max(avg_over_time(quantum_proof_aggregation_time_taken[5m]) or vector(0)) / clamp_min(max(avg_over_time(quantum_proof_verify_time_taken[5m]) or vector(0)), 0.001)) or vector(0))"

    verify_rates = query_range(args.prom_url, verify_rate_expr, start, end, args.step_seconds, auth_header=auth_header)
    failure_ratios = query_range(
        args.prom_url, failure_ratio_expr, start, end, args.step_seconds, auth_header=auth_header
    )
    latency_ratios = query_range(
        args.prom_url, latency_ratio_expr, start, end, args.step_seconds, auth_header=auth_header
    )

    # Align lengths conservatively by shortest list.
    n = min(len(verify_rates), len(failure_ratios), len(latency_ratios))
    verify_rates = verify_rates[:n]
    failure_ratios = failure_ratios[:n]
    latency_ratios = latency_ratios[:n]

    filtered_failure = []
    filtered_latency = []
    for i in range(n):
        if verify_rates[i] >= args.min_throughput_rps:
            filtered_failure.append(failure_ratios[i])
            filtered_latency.append(latency_ratios[i])

    fail_summary = summarize(filtered_failure)
    lat_summary = summarize(filtered_latency)

    rec_fail = max(0.01, fail_summary["p95"] * args.headroom_factor)
    rec_lat = max(1.5, lat_summary["p95"] * args.headroom_factor)
    profile = recommend_profile(rec_fail, rec_lat)

    base_gate_cmd = [
        "python3",
        "nvflare_quantum_readiness_gate.py",
        "--prom-url",
        args.prom_url,
    ]
    if args.prom_user:
        base_gate_cmd.extend(["--prom-user", args.prom_user])
    if args.prom_password:
        # Avoid embedding plaintext credentials in JSON reports or CI logs.
        base_gate_cmd.extend(["--prom-password", "${NVFLARE_PROM_PASSWORD}"])

    profile_apply_cmd = " ".join(base_gate_cmd + ["--profile", profile])
    custom_apply_cmd = " ".join(
        base_gate_cmd
        + [
            "--profile",
            "custom",
            "--max-failure-ratio",
            f"{rec_fail:.4f}",
            "--max-latency-ratio",
            f"{rec_lat:.4f}",
        ]
    )

    report = {
        "window": {
            "lookback_minutes": args.lookback_minutes,
            "step_seconds": args.step_seconds,
            "min_throughput_rps": args.min_throughput_rps,
        },
        "coverage": {
            "total_samples": n,
            "traffic_filtered_samples": len(filtered_failure),
        },
        "failure_ratio": fail_summary,
        "latency_ratio": lat_summary,
        "recommendation": {
            "recommended_profile": profile,
            "max_failure_ratio": round(rec_fail, 6),
            "max_latency_ratio": round(rec_lat, 6),
            "shell_exports": {
                "NVFLARE_RECOMMENDED_PROFILE": profile,
                "NVFLARE_MAX_FAILURE_RATIO": f"{rec_fail:.6f}",
                "NVFLARE_MAX_LATENCY_RATIO": f"{rec_lat:.6f}",
            },
            "apply_commands": {
                "profile_mode": profile_apply_cmd,
                "custom_mode": custom_apply_cmd,
            },
        },
    }

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output != "-":
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    elif not args.print_shell_export:
        print(output)

    if args.print_shell_export:
        print(f"export NVFLARE_RECOMMENDED_PROFILE={profile}")
        print(f"export NVFLARE_MAX_FAILURE_RATIO={rec_fail:.6f}")
        print(f"export NVFLARE_MAX_LATENCY_RATIO={rec_lat:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
