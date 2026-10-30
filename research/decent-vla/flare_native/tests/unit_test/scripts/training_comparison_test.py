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

"""Comparison must reject missing work and expose the chosen weight-check coverage."""

import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "comparison", Path(__file__).resolve().parents[3] / "scripts/compare_training.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_nested_client_count_overrides_and_reference_config_is_unchanged():
    raw = {
        "backend": {"name": "mock"},
        "data": {"partitioning": {"num_clients": 8}},
        "fl": {"rounds": 100, "execution": "local", "nvflare": {"gpu": "0,1"}},
    }
    original = copy.deepcopy(raw)
    cfg = comparison.prepare_config(
        raw, num_clients=2, rounds=3, strategy="fedadam", wire_dtype="fp32", checkpoint_every=0
    )
    assert raw == original
    assert cfg["data"]["partitioning"]["num_clients"] == cfg["data"]["num_clients"] == cfg["fl"]["min_clients"] == 2
    assert cfg["fl"]["num_rounds"] == 3 and "rounds" not in cfg["fl"]
    assert cfg["fl"]["mode"] == "nvflare" and "execution" not in cfg["fl"]
    assert cfg["fl"]["save_every_n_rounds"] == 0
    assert cfg["data"]["num_workers"] == 0
    assert cfg["backend"]["extra"]["deterministic"] is True


@pytest.mark.parametrize("gpu,threads", [(None, 1), ("0", 1), ("0,1", 8), ("0,1,2,3,4,5,6,7,8", 8)])
def test_strict_comparison_rejects_swapping_or_excess_gpu_groups(gpu, threads):
    with pytest.raises(ValueError, match="resident clients"):
        comparison.prepare_config({"data": {"num_clients": 8}, "fl": {}}, gpu=gpu, threads=threads)


def test_repeated_gpu_groups_keep_one_resident_slot_per_client_even_with_threads_one():
    cfg = comparison.prepare_config({"data": {"num_clients": 4}, "fl": {}}, gpu="0,1,0,1", threads=1)
    assert cfg["fl"]["nvflare"]["gpu"] == "0,1,0,1"


@pytest.fixture
def outputs(tmp_path):
    for label in ("reference", "native"):
        folder = tmp_path / label
        (folder / "checkpoints").mkdir(parents=True)
        (folder / "partition_manifest.json").write_text('{"c0":[0],"c1":[1]}')
        curve = [
            {"round": r, "client_losses": {"c0": 1 / (r + 1), "c1": 2 / (r + 1)}, "mean_client_loss": 1.5 / (r + 1)}
            for r in range(2)
        ]
        (folder / "training_curve.json").write_text(json.dumps(curve))
        torch.save({"state": {"w": torch.tensor([2.0])}}, folder / "checkpoints/server_round_final.pt")
    return tmp_path


def check(out, **kwargs):
    return comparison.compare_outputs(
        out / "reference", out / "native", checkpoint_every=0, expected_rounds=2, num_clients=2, **kwargs
    )


def test_final_only_still_checks_every_loss_and_writes_plotting_curves(outputs):
    report = check(outputs)
    assert report[0]["max_weight_error"] is None
    assert report[1]["checkpoint"] == "server_round_final.pt" and report[1]["max_weight_error"] == 0
    comparison.write_curves(outputs)
    with (outputs / "training_curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert [(r["round"], r["reference_loss"]) for r in rows if r["client"] == "mean"] == [("1", "1.5"), ("2", "0.75")]


@pytest.mark.parametrize("damage", ["loss", "client", "round", "nan"])
def test_final_only_does_not_mask_an_intermediate_round_defect(outputs, damage):
    path = outputs / "native/training_curve.json"
    curve = json.loads(path.read_text())
    if damage == "loss":
        curve[0]["client_losses"]["c0"] += 0.1
    elif damage == "client":
        del curve[0]["client_losses"]["c1"]
    elif damage == "round":
        curve.pop(0)
    else:
        curve[0]["client_losses"]["c0"] = float("nan")
    path.write_text(json.dumps(curve))
    with pytest.raises(AssertionError):
        check(outputs)


def test_missing_weight_checkpoint_cannot_report_success(outputs):
    (outputs / "native/checkpoints/server_round_final.pt").unlink()
    with pytest.raises(FileNotFoundError):
        check(outputs)


def test_per_round_mode_requires_intermediate_weights(outputs):
    with pytest.raises(FileNotFoundError):
        comparison.compare_outputs(outputs / "reference", outputs / "native", checkpoint_every=1)


def test_nested_explicit_repos_infers_the_count_from_repo_lists():
    cfg = comparison.prepare_config(
        {"data": {"partitioning": {"scheme": "explicit_repos"}, "client_repo_ids": [["r/a"], ["r/b"]]}, "fl": {}}
    )
    assert cfg["data"]["num_clients"] == cfg["fl"]["min_clients"] == 2
    assert cfg["fl"]["nvflare"]["threads"] == 2
