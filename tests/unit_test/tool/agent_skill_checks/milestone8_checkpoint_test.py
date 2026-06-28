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

import sys
from pathlib import Path

CHECKS_PARENT = Path(__file__).resolve().parents[4] / "dev_tools" / "agent" / "skills"
sys.path.insert(0, str(CHECKS_PARENT))

from checks import milestone8_checkpoint as checkpoint  # noqa: E402


def _valid_run(agent, skill):
    return {
        "agent": agent,
        "skill": skill,
        "correctness": "validated",
        "runtime_seconds": 12.5,
        "dependency_behavior": "used installed dependencies",
        "generated_structure": "generated job directory",
        "token_usage": {"input_tokens": 100, "output_tokens": 25},
        "metric_evidence": "pytest passed",
        "artifact_location": "outputs/run.json",
    }


def _valid_payload():
    return {
        "schema_version": "1",
        "runs": [
            _valid_run(agent, skill) for agent in checkpoint.BENCHMARK_AGENTS for skill in checkpoint.CONVERSION_SKILLS
        ],
    }


def test_validate_benchmark_evidence_rejects_placeholder_values():
    payload = _valid_payload()
    payload["runs"][0].update(
        {
            "correctness": " ",
            "runtime_seconds": None,
            "token_usage": {},
            "metric_evidence": "",
        }
    )

    findings = checkpoint._validate_benchmark_evidence_payload(payload)

    assert "runs[0] field correctness must be a non-empty string" in findings
    assert "runs[0] field runtime_seconds must be a non-negative number" in findings
    assert "runs[0] field token_usage must contain at least one positive token count" in findings
    assert "runs[0] field metric_evidence must be a non-empty string" in findings


def test_validate_benchmark_evidence_accepts_complete_records():
    assert checkpoint._validate_benchmark_evidence_payload(_valid_payload()) == []


def test_check_skill_lints_passes_only_skills_root(monkeypatch, tmp_path):
    calls = []

    def fake_run_v1_lints(skills_root):
        calls.append(skills_root)
        return {"status": "ok", "summary": {"skill_count": 2}}

    monkeypatch.setattr(checkpoint, "run_v1_lints", fake_run_v1_lints)

    result = checkpoint._check_skill_lints(tmp_path)

    assert result["status"] == "ok"
    assert result["data"]["summary"] == {"skill_count": 2}
    assert calls == [tmp_path / "skills"]


def test_check_packaging_fails_when_release_install_plan_reports_errors(monkeypatch, tmp_path):
    skill_names = list(checkpoint.CONVERSION_SKILLS)
    release_manifest = {"skills": [{"name": name} for name in skill_names]}

    def fake_copy_released_skills_to_bundle(skills_root, bundle_root, **kwargs):
        if kwargs.get("include_analysis_files", True):
            bundle_root.joinpath(checkpoint.LIGHTNING_SKILL, "evals").mkdir(parents=True)
            bundle_root.joinpath(checkpoint.LIGHTNING_SKILL, "evals", "evals.json").write_text("{}", encoding="utf-8")
        else:
            bundle_root.joinpath(checkpoint.LIGHTNING_SKILL).mkdir(parents=True)
        return release_manifest

    monkeypatch.setattr(checkpoint, "copy_released_skills_to_bundle", fake_copy_released_skills_to_bundle)
    monkeypatch.setattr(
        checkpoint,
        "install_skills",
        lambda **kwargs: {"applied": False, "errors": [{"message": "copy failed"}]},
    )

    result = checkpoint._check_packaging(tmp_path)

    assert result["status"] == "failed"
    assert result["message"] == "release skill bundle did not install successfully"
    assert result["data"]["install_plan"]["errors"]


def test_check_packaging_fails_when_release_install_is_not_listable(monkeypatch, tmp_path):
    skill_names = list(checkpoint.CONVERSION_SKILLS)
    release_manifest = {"skills": [{"name": name} for name in skill_names]}

    def fake_copy_released_skills_to_bundle(skills_root, bundle_root, **kwargs):
        if kwargs.get("include_analysis_files", True):
            bundle_root.joinpath(checkpoint.LIGHTNING_SKILL, "evals").mkdir(parents=True)
            bundle_root.joinpath(checkpoint.LIGHTNING_SKILL, "evals", "evals.json").write_text("{}", encoding="utf-8")
        else:
            bundle_root.joinpath(checkpoint.LIGHTNING_SKILL).mkdir(parents=True)
        return release_manifest

    monkeypatch.setattr(checkpoint, "copy_released_skills_to_bundle", fake_copy_released_skills_to_bundle)
    monkeypatch.setattr(checkpoint, "install_skills", lambda **kwargs: {"applied": True, "errors": []})
    monkeypatch.setattr(
        checkpoint,
        "list_skills",
        lambda **kwargs: {"installed": [{"name": checkpoint.PYTORCH_SKILL}], "errors": []},
    )

    result = checkpoint._check_packaging(tmp_path)

    assert result["status"] == "failed"
    assert result["message"] == "conversion skills are not listable from the installed release bundle"
    assert result["data"]["missing"] == [checkpoint.LIGHTNING_SKILL]
