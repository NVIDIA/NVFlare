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

import evaluate


def test_test_caches_are_opened_only_after_validation_selection(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    cache_dir.mkdir()
    metadata = {
        "input_dim": 4,
        "sites": {
            site: {"train": {"paired_examples": 1 if site == "site-1" else 0}}
            for site in ("site-1", "site-2", "site-3")
        },
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata))
    selection_complete = False
    opened = []

    def fake_load(_cache_dir, site, split):
        nonlocal selection_complete
        if split == "test":
            assert selection_complete
        opened.append((site, split))
        return {"site": site, "split": split}

    def fake_select(_statistics, aggregate_loss_tolerance=0.0):
        nonlocal selection_complete
        selection_complete = True
        return 0.0, [{"alpha": 0.0, "feasible": True}]

    summary = {
        "missing_before": {"auroc": 0.5},
        "missing_after": {"auroc": 0.5},
        "missing_delta_auroc": 0.0,
        "aggregate_before": {"auroc": 0.5},
        "aggregate_after": {"auroc": 0.5},
        "aggregate_delta_auroc": 0.0,
    }
    monkeypatch.setattr(evaluate, "_load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluate, "load_cache_split", fake_load)
    monkeypatch.setattr(evaluate, "prepare_site", lambda site, payload, model: payload)
    monkeypatch.setattr(evaluate, "validation_sufficient_statistics", lambda sites, grid: [{"alpha": 0.0}])
    monkeypatch.setattr(evaluate, "select_alpha_from_statistics", fake_select)
    monkeypatch.setattr(evaluate, "evaluate_sites", lambda sites, alpha: (summary, {}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--cache-dir",
            str(cache_dir),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-dir",
            str(output_dir),
        ],
    )

    evaluate.main()

    assert opened[:3] == [("site-1", "val"), ("site-2", "val"), ("site-3", "val")]
    assert opened[3:] == [("site-1", "test"), ("site-2", "test"), ("site-3", "test")]
