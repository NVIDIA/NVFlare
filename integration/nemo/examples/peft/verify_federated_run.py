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
"""Independently verify adapter continuity and FP32 FedAvg results."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import OrderedDict
from typing import Mapping

import adapter_checkpoint
import torch


def weighted_average(states: list[Mapping[str, torch.Tensor]], weights: list[float]) -> OrderedDict[str, torch.Tensor]:
    if not states or len(states) != len(weights):
        raise ValueError("States and weights must have the same nonzero length.")
    reference = adapter_checkpoint.align_adapter_state_strict(states[0], states[0])
    total_weight = sum(weights)
    if not total_weight > 0:
        raise ValueError("The total aggregation weight must be positive.")
    result = OrderedDict()
    for key, reference_value in reference.items():
        accumulator = torch.zeros(reference_value.shape, dtype=torch.float32)
        for state, weight in zip(states, weights):
            aligned = adapter_checkpoint.align_adapter_state_strict(state, reference)
            accumulator.add_(aligned[key].float(), alpha=weight)
        result[key] = accumulator.div_(total_weight)
    return result


def verify_aggregate(
    client_states: list[Mapping[str, torch.Tensor]],
    weights: list[float],
    aggregate_state: Mapping[str, torch.Tensor],
) -> dict:
    aggregate = adapter_checkpoint.align_adapter_state_strict(aggregate_state, client_states[0])
    expected = weighted_average(client_states, weights)
    mismatches = []
    max_abs_error = 0.0
    for key, value in aggregate.items():
        rounded_expected = expected[key].to(value.dtype)
        difference = (value.cpu() - rounded_expected.cpu()).abs().float()
        tensor_max = float(difference.max().item()) if difference.numel() else 0.0
        max_abs_error = max(max_abs_error, tensor_max)
        if not torch.equal(value.cpu(), rounded_expected.cpu()):
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            f"Server aggregate differs from independent FP32 FedAvg for {len(mismatches)} tensors; "
            f"examples={mismatches[:5]}, max_abs_error={max_abs_error}."
        )
    return {"tensor_count": len(aggregate), "max_abs_error": max_abs_error, "weights": weights}


def _one(path_pattern: str) -> str:
    matches = glob.glob(path_pattern, recursive=True)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one match for {path_pattern}, found {len(matches)}: {matches[:5]}")
    return matches[0]


def verify_run(client_work_dir: str, server_root: str, num_clients: int, num_rounds: int) -> dict:
    report = {"schema_version": 1, "rounds": []}
    previous_aggregate_hash = None
    for round_idx in range(num_rounds):
        states = []
        weights = []
        client_reports = []
        for site_idx in range(1, num_clients + 1):
            round_dir = os.path.join(client_work_dir, f"site-{site_idx}", f"site-{site_idx}_round_{round_idx}")
            with open(os.path.join(round_dir, "round_manifest.json")) as f:
                client_manifest = json.load(f)
            load_report = client_manifest.get("automodel_report", {})
            load_matches = load_report.get(
                "loaded_matches_received_after_dtype_cast",
                client_manifest["received_adapter_hash"] == client_manifest["loaded_adapter_hash"],
            )
            if not load_matches:
                raise ValueError(f"site-{site_idx} round {round_idx} did not load the received adapter exactly.")
            if previous_aggregate_hash and client_manifest["received_adapter_hash"] != previous_aggregate_hash:
                raise ValueError(f"site-{site_idx} round {round_idx} received a stale global adapter.")
            state = adapter_checkpoint.load_adapter_state(client_manifest["checkpoint_location"])
            if adapter_checkpoint.state_hash(state) != client_manifest["outgoing_adapter_hash"]:
                raise ValueError(f"site-{site_idx} round {round_idx} outgoing checkpoint hash mismatch.")
            states.append(state)
            weights.append(float(client_manifest["actual_optimizer_steps"]))
            client_reports.append(client_manifest)

        server_manifest_path = _one(
            os.path.join(server_root, "**", "server_rounds", f"round_{round_idx}", "round_manifest.json")
        )
        with open(server_manifest_path) as f:
            server_manifest = json.load(f)
        aggregate = adapter_checkpoint.strip_model_prefix(
            adapter_checkpoint.load_adapter_state(server_manifest["checkpoint_location"])
        )
        aggregate_report = verify_aggregate(states, weights, aggregate)
        aggregate_hash = adapter_checkpoint.state_hash(aggregate)
        if aggregate_hash != server_manifest["aggregate_adapter_hash"]:
            raise ValueError(f"Server round {round_idx} manifest hash mismatch.")
        previous_aggregate_hash = aggregate_hash
        report["rounds"].append(
            {
                "round": round_idx,
                "clients": client_reports,
                "aggregate": server_manifest,
                "independent_fp32_verification": aggregate_report,
            }
        )
    report["all_checks_passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client_work_dir", required=True)
    parser.add_argument("--server_root", required=True)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify_run(args.client_work_dir, args.server_root, args.num_clients, args.num_rounds)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
