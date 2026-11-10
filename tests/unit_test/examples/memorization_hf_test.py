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
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.app_common.filters.memorization_audit_filter import MemorizationAuditFilter

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
    # Cohort labels must not be exposed in the text scored by the membership attacks.
    rows = data["reference"] + data["validation"]
    rows += [row for site in ("site-1", "site-2") for row in data[site]["train"] + data[site]["nonmembers"]]
    rows += data["final"]["nonmembers"]
    assert len({row["prompt"] for row in rows}) == len(rows)
    assert all(re.fullmatch(r"Fictional dossier [0-9a-f]{8}\. Access code:", row["prompt"]) for row in rows)
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


def test_coverage_reports_sdk_failure_with_unknown_round(tmp_path, monkeypatch, capsys):
    engine = Mock()
    engine.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    context = FLContextManager(engine=engine, identity_name="site-1", job_id="test-job").new_context()
    result = Shareable()
    result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    MemorizationAuditFilter(load("hf_model").DossierAuditor(str(tmp_path))).process(result, context)
    report = json.loads(next(tmp_path.rglob("memorization_audits/*.json")).read_text())
    assert report["round"] is None
    assert (report["status"], report["reason"]) == ("skipped", "task_result_not_ok")

    monkeypatch.setattr(sys, "argv", ["check_coverage.py", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        load("check_coverage").main()

    assert exc.value.code == 1
    summary = json.loads(capsys.readouterr().out)
    assert not summary["complete"]
    assert summary["reports"] == 1 and summary["expected"] == 4
    unknown = [problem for problem in summary["problems"] if problem["round"] is None]
    assert unknown == [
        {
            "site": "site-1",
            "round": None,
            "reason": "unknown site or round",
            "status": report["status"],
            "report_reason": report["reason"],
        }
    ]
    missing = [problem for problem in summary["problems"] if problem["round"] is not None]
    assert {(problem["site"], problem["round"]) for problem in missing} == {
        (site, round_) for site in ("site-1", "site-2") for round_ in (0, 1)
    }
    assert all(problem["reason"] == "unexpected, missing or duplicate report" for problem in missing)


def test_offline_global_report_has_valid_workspace(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hf_model", load("hf_model"))
    subject = load("evaluate")
    context = subject.FLContextManager(
        engine=subject.EvaluationWorkspace(str(tmp_path)), identity_name="global-evaluation", job_id="final-evaluation"
    ).new_context()
    path = subject.MemorizationAuditReporter().report(
        {"status": "completed", "round": 1, "model_fingerprint": "checkpoint"}, context, stage="global_model"
    )
    assert path is not None and path.is_relative_to(tmp_path)
    report = json.loads(path.read_text())
    assert (report["stage"], report["job_id"], report["site"]) == (
        "global_model",
        "final-evaluation",
        "global-evaluation",
    )
