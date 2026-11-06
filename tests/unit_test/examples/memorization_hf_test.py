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
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "research/memorization-reduction/huggingface"


def load(name):
    spec = importlib.util.spec_from_file_location(f"trap_hf_{name}", EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cohort_isolation_and_final_membership():
    data = load("hf_model").make_cohorts()

    def ids(rows):
        return {r["id"] for r in rows}

    train = ids(data["site-1"]["train"] + data["site-2"]["train"])
    local = ids(data["site-1"]["members"] + data["site-2"]["members"])
    final = ids(data["final"]["members"])
    assert len(train) == 32 and len(local) == len(final) == 16
    assert not local & final and local | final == train
    other_groups = [ids(data[key]) for key in ("reference", "validation")]
    other_groups += [ids(data[key]["nonmembers"]) for key in ("site-1", "site-2", "final")]
    for index, group in enumerate(other_groups):
        assert not group & train
        assert all(not group & earlier for earlier in other_groups[:index])


def test_completion_mask_and_no_silent_truncation():
    subject = load("hf_model")
    tokenizer = SimpleNamespace(eos_token_id=0, encode=lambda text, **kwargs: [ord(c) for c in text])
    value = subject.batch(tokenizer, [{"prompt": "abc", "answer": "de"}, {"prompt": "f", "answer": "g"}], "cpu")
    assert value["labels"].tolist() == [[-100, -100, -100, 100, 101], [-100, 103, -100, -100, -100]]
    assert value["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
    with pytest.raises(ValueError, match="overlong"):
        subject.batch(tokenizer, [{"prompt": "x" * 128, "answer": "y"}], "cpu")


def test_coverage_detects_missing_failed_duplicate_and_partial_audits():
    check = load("check_coverage").coverage
    report = {
        "site": "site-1",
        "round": 0,
        "stage": "client_task_result",
        "status": "completed",
        "model_fingerprint": "candidate",
        "baseline_fingerprint": "baseline",
        "sample_counts": {name: {"members": 8, "nonmembers": 8} for name in ("likelihood", "tra")},
    }
    expected = {("site-1", 0)}
    assert check([report], expected)["complete"]
    assert not check([], expected)["complete"]
    assert not check([report, report], expected)["complete"]
    assert not check([report], expected | {("site-2", 0)})["complete"]
    assert not check([{**report, "sample_counts": {}}], expected)["complete"]
    for status in ("failed", "unsupported", "skipped"):
        assert not check([{**report, "status": status}], expected)["complete"]
