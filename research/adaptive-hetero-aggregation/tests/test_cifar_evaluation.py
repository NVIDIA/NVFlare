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
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_DIR / "cifar10_evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from eval_split import _integer_allocation  # noqa: E402
from evaluate_result import checkpoint_state_dict, find_server_checkpoint  # noqa: E402
from run_campaign import _completed_keys  # noqa: E402
from summarize_results import summarize  # noqa: E402


def test_find_server_checkpoint_prefers_final_round_model(tmp_path):
    server_dir = tmp_path / "server" / "simulate_job" / "app_server"
    server_dir.mkdir(parents=True)
    best = server_dir / "best_FL_global_model.pt"
    final = server_dir / "FL_global_model.pt"
    best.write_bytes(b"best")
    final.write_bytes(b"final")

    assert find_server_checkpoint(str(tmp_path)) == final


def test_checkpoint_state_dict_supports_nvflare_pt_format(tmp_path):
    path = tmp_path / "FL_global_model.pt"
    expected = {"weight": torch.tensor([1.0, 2.0])}
    torch.save({"model": expected, "meta": {}}, path)

    loaded = checkpoint_state_dict(path)

    assert set(loaded) == {"weight"}
    assert torch.equal(loaded["weight"], expected["weight"])


def test_integer_allocation_preserves_total_and_nonnegativity():
    allocated = _integer_allocation(17, np.asarray([0.1, 0.2, 0.7]))

    assert allocated.sum() == 17
    assert np.all(allocated >= 0)


def test_summary_reports_ci_and_paired_split_seed_deltas():
    rows = []
    for seed, adaptive, fedavg in ((7, 0.80, 0.75), (19, 0.82, 0.78), (31, 0.81, 0.77)):
        rows.extend(
            [
                {
                    "method": "adaptive",
                    "alpha": 0.1,
                    "participation_rate": 1.0,
                    "seed": seed,
                    "global_accuracy": adaptive,
                    "worst_client_accuracy": adaptive - 0.10,
                },
                {
                    "method": "fedavg",
                    "alpha": 0.1,
                    "participation_rate": 1.0,
                    "seed": seed,
                    "global_accuracy": fedavg,
                    "worst_client_accuracy": fedavg - 0.12,
                },
            ]
        )

    result = summarize(rows)

    adaptive_summary = next(item for item in result["summaries"] if item["method"] == "adaptive")
    assert adaptive_summary["metrics"]["global_accuracy"]["n"] == 3
    assert adaptive_summary["metrics"]["global_accuracy"]["ci95_low"] is not None
    paired = result["paired_comparisons"][0]
    assert paired["seeds"] == [7, 19, 31]
    assert paired["delta_reference_minus_baseline"]["global_accuracy"]["mean"] > 0.0


def test_campaign_resume_uses_method_condition_and_seed(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps(
            {
                "method": "adaptive",
                "alpha": 0.1,
                "participation_rate": 0.75,
                "seed": 19,
            }
        )
        + "\n"
    )

    assert _completed_keys(path) == {("adaptive", 0.1, 0.75, 19)}
