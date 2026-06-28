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

"""Milestone 8 checkpoint for the two base conversion skills.

This is a private development check, not an ``nvflare agent`` command. It
automates the deterministic parts of the Stage 5 checkpoint and validates
separately supplied benchmark evidence for Codex/Claude conversion runs.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Optional

try:
    from .lints import run_v1_lints
except ImportError:
    from lints import run_v1_lints

from nvflare.tool.agent.inspector import inspect_path
from nvflare.tool.agent.skill_manager import SkillSource, install_skills, list_skills
from nvflare.tool.agent.skill_manifest import build_skill_manifest, copy_released_skills_to_bundle

CHECKPOINT_SCHEMA_VERSION = "1"
CHECKPOINT_STAGE = "milestone8_two_conversion_skill_checkpoint"
PYTORCH_SKILL = "nvflare-convert-pytorch"
LIGHTNING_SKILL = "nvflare-convert-lightning"
CONVERSION_SKILLS = (PYTORCH_SKILL, LIGHTNING_SKILL)
BENCHMARK_AGENTS = ("codex", "claude")
TOKEN_COUNT_KEY_TERMS = ("token", "input", "output", "prompt", "completion", "total", "cached", "reasoning")
REQUIRED_BENCHMARK_FIELDS = (
    "agent",
    "skill",
    "correctness",
    "runtime_seconds",
    "dependency_behavior",
    "generated_structure",
    "token_usage",
    "metric_evidence",
    "artifact_location",
)
DESCRIPTIVE_BENCHMARK_FIELDS = (
    "agent",
    "skill",
    "correctness",
    "dependency_behavior",
    "generated_structure",
    "metric_evidence",
    "artifact_location",
)


def run_milestone8_checkpoint(
    repo_root: Path | str = ".", *, benchmark_evidence: Optional[Path | str] = None
) -> dict[str, Any]:
    """Run deterministic Stage 5 checks and validate optional benchmark evidence."""
    root = Path(repo_root).resolve()
    checks = [
        _check_skill_lints(root),
        _check_inspect_routing(),
        _check_install_list(root),
        _check_packaging(root),
        _check_benchmark_evidence(Path(benchmark_evidence) if benchmark_evidence else None),
    ]
    status = _overall_status(checks)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": CHECKPOINT_STAGE,
        "status": status,
        "passed": status == "ok",
        "summary": {
            "ok_count": sum(1 for check in checks if check["status"] == "ok"),
            "manual_required_count": sum(1 for check in checks if check["status"] == "manual_required"),
            "failed_count": sum(1 for check in checks if check["status"] == "failed"),
        },
        "checks": checks,
    }


def _check_skill_lints(repo_root: Path) -> dict[str, Any]:
    skills_root = repo_root / "skills"
    docs_root = repo_root / "docs" / "design"
    try:
        result = run_v1_lints(skills_root, docs_root=docs_root)
    except Exception as e:
        return _failed("skill_admission_lints", f"skill lint/admission checks raised {type(e).__name__}: {e}")
    if result.get("status") != "ok":
        return _failed(
            "skill_admission_lints",
            "skill lint/admission checks reported findings",
            {"summary": result.get("summary", {}), "findings": result.get("findings", [])},
        )
    return _ok("skill_admission_lints", "skill lint/admission checks passed", {"summary": result.get("summary", {})})


def _check_inspect_routing() -> dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = root / "plain_pytorch.py"
            plain.write_text(
                "import torch\n\nclass Net(torch.nn.Module):\n    pass\n\ndef train():\n    return Net()\n",
                encoding="utf-8",
            )

            lightning = root / "lightning_train.py"
            lightning.write_text(
                "import pytorch_lightning as pl\n\nclass Net(pl.LightningModule):\n    pass\n",
                encoding="utf-8",
            )

            exported = root / "exported"
            exported.joinpath("app_server").mkdir(parents=True)
            exported.joinpath("app_server", "config_fed_server.json").write_text("{}\n", encoding="utf-8")
            exported.joinpath("client.py").write_text(
                "import pytorch_lightning as pl\n\nclass Net(pl.LightningModule):\n    pass\n",
                encoding="utf-8",
            )

            plain_result = inspect_path(plain)
            lightning_result = inspect_path(lightning)
            exported_result = inspect_path(exported)
            _assert_recommended_skill(plain_result, PYTORCH_SKILL)
            _assert_recommended_skill(lightning_result, LIGHTNING_SKILL)
            _assert_recommended_skill(exported_result, "nvflare-job-lifecycle")
    except Exception as e:
        return _failed("inspect_routing", f"inspect routing checkpoint failed: {e}")
    return _ok(
        "inspect_routing",
        "agent inspect routes PyTorch, Lightning, and exported jobs as expected",
        {
            "pytorch_skill": PYTORCH_SKILL,
            "lightning_skill": LIGHTNING_SKILL,
            "exported_job_skill": "nvflare-job-lifecycle",
        },
    )


def _check_install_list(repo_root: Path) -> dict[str, Any]:
    source = _editable_skill_source(repo_root)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "codex-skills"
            install_plan = install_skills(agent="codex", target_dir=target, source=source)
            listed = list_skills(agent="codex", target_dir=target, source=source)
            installed = {skill["name"] for skill in listed["installed"]}
            missing = [skill for skill in CONVERSION_SKILLS if skill not in installed]
            if missing:
                return _failed(
                    "install_list",
                    "conversion skills were not installed/listed",
                    {"missing": missing, "installed": sorted(installed), "install_plan": install_plan},
                )
    except Exception as e:
        return _failed("install_list", f"conversion skill install/list checkpoint raised {type(e).__name__}: {e}")
    return _ok("install_list", "PyTorch and Lightning skills install and list successfully")


def _check_packaging(repo_root: Path) -> dict[str, Any]:
    skills_root = repo_root / "skills"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            dev_bundle = tmp_root / "dev-bundle"
            release_bundle = tmp_root / "release-bundle"
            dev_manifest = copy_released_skills_to_bundle(skills_root, dev_bundle, nvflare_version="2.8.0")
            release_manifest = copy_released_skills_to_bundle(
                skills_root,
                release_bundle,
                nvflare_version="2.8.0",
                include_analysis_files=False,
            )
            if not dev_bundle.joinpath(LIGHTNING_SKILL, "evals", "evals.json").is_file():
                return _failed("packaging", "dev skill bundle is missing Lightning eval metadata")
            if release_bundle.joinpath(LIGHTNING_SKILL, "evals").exists():
                return _failed("packaging", "release skill bundle still contains Lightning eval metadata")

            release_source = SkillSource(source_type="wheel", root=release_bundle, manifest=release_manifest)
            release_target = tmp_root / "release-target"
            install_plan = install_skills(agent="codex", target_dir=release_target, source=release_source)
            if _install_plan_failed(install_plan):
                return _failed(
                    "packaging",
                    "release skill bundle did not install successfully",
                    {"install_plan": install_plan},
                )
            listed = list_skills(agent="codex", target_dir=release_target, source=release_source)

            dev_names = {skill["name"] for skill in dev_manifest.get("skills", [])}
            release_names = {skill["name"] for skill in release_manifest.get("skills", [])}
            missing = [skill for skill in CONVERSION_SKILLS if skill not in dev_names or skill not in release_names]
            if missing:
                return _failed(
                    "packaging",
                    "conversion skills are missing from dev or release manifests",
                    {"missing": missing},
                )

            installed = {skill["name"] for skill in listed.get("installed", [])}
            release_missing = [skill for skill in CONVERSION_SKILLS if skill not in installed]
            if release_missing or listed.get("errors"):
                return _failed(
                    "packaging",
                    "conversion skills are not listable from the installed release bundle",
                    {
                        "missing": release_missing,
                        "installed": sorted(installed),
                        "list_errors": listed.get("errors", []),
                        "install_plan": install_plan,
                    },
                )
    except Exception as e:
        return _failed("packaging", f"packaging checkpoint raised {type(e).__name__}: {e}")
    return _ok(
        "packaging",
        "dev bundle includes analysis metadata and release bundle installs without it",
        {"skills": list(CONVERSION_SKILLS)},
    )


def _check_benchmark_evidence(evidence_path: Optional[Path]) -> dict[str, Any]:
    if evidence_path is None:
        return _manual_required(
            "benchmark_evidence",
            "Codex/Claude PyTorch and Lightning benchmark evidence was not supplied",
            {"required_pairs": _required_benchmark_pair_records()},
        )
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception as e:
        return _failed("benchmark_evidence", f"failed to read benchmark evidence: {type(e).__name__}: {e}")
    findings = _validate_benchmark_evidence_payload(payload)
    if findings:
        return _failed("benchmark_evidence", "benchmark evidence is incomplete", {"findings": findings})
    return _ok("benchmark_evidence", "benchmark evidence covers Codex/Claude PyTorch and Lightning conversion runs")


def _validate_benchmark_evidence_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["benchmark evidence must be a JSON object"]
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return ["benchmark evidence must use schema_version '1'"]
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return ["benchmark evidence must contain a runs list"]

    findings = []
    indexed_runs = {}
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            findings.append(f"runs[{index}] must be an object")
            continue

        missing_fields = [field for field in REQUIRED_BENCHMARK_FIELDS if field not in run]
        if missing_fields:
            findings.append(f"runs[{index}] missing required fields: {', '.join(missing_fields)}")

        findings.extend(_validate_benchmark_run_values(index, run))

        agent = run.get("agent")
        skill = run.get("skill")
        if isinstance(agent, str) and agent and isinstance(skill, str) and skill:
            indexed_runs[(agent, skill)] = run

    for agent, skill in _required_benchmark_pairs():
        if (agent, skill) not in indexed_runs:
            findings.append(f"missing benchmark run for agent={agent}, skill={skill}")
    return findings


def _validate_benchmark_run_values(index: int, run: dict[str, Any]) -> list[str]:
    findings = []
    for field in DESCRIPTIVE_BENCHMARK_FIELDS:
        if field in run and not _is_non_empty_string(run[field]):
            findings.append(f"runs[{index}] field {field} must be a non-empty string")

    if "runtime_seconds" in run and not _is_non_negative_number(run["runtime_seconds"]):
        findings.append(f"runs[{index}] field runtime_seconds must be a non-negative number")

    if "token_usage" in run and not _is_token_usage_record(run["token_usage"]):
        findings.append(f"runs[{index}] field token_usage must contain at least one positive token count")
    return findings


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_non_negative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _is_token_usage_record(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return any(_is_positive_number(token_count) for token_count in _iter_token_counts(value))


def _iter_token_counts(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_token_count_key(key) and _is_non_negative_number(child):
                yield child
            else:
                yield from _iter_token_counts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_token_counts(child)


def _is_token_count_key(value: Any) -> bool:
    return isinstance(value, str) and any(term in value.lower() for term in TOKEN_COUNT_KEY_TERMS)


def _is_positive_number(value: Any) -> bool:
    return _is_non_negative_number(value) and value > 0


def _install_plan_failed(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return True
    return not plan.get("applied") or bool(plan.get("errors"))


def _required_benchmark_pairs() -> list[tuple[str, str]]:
    return [(agent, skill) for agent in BENCHMARK_AGENTS for skill in CONVERSION_SKILLS]


def _required_benchmark_pair_records() -> list[dict[str, str]]:
    return [{"agent": agent, "skill": skill} for agent, skill in _required_benchmark_pairs()]


def _editable_skill_source(repo_root: Path) -> SkillSource:
    skills_root = repo_root / "skills"
    return SkillSource(
        source_type="editable",
        root=skills_root,
        manifest=build_skill_manifest(skills_root, source_type="editable", nvflare_version="2.8.0"),
    )


def _assert_recommended_skill(result: dict[str, Any], expected_skill: str) -> None:
    recommended = result.get("skill_selection", {}).get("recommended_skills", [])
    if expected_skill not in recommended:
        raise AssertionError(f"expected {expected_skill}, got {recommended}")


def _overall_status(checks: list[dict[str, Any]]) -> str:
    if any(check["status"] == "failed" for check in checks):
        return "failed"
    if any(check["status"] == "manual_required" for check in checks):
        return "incomplete"
    return "ok"


def _ok(check_id: str, message: str, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _check(check_id, "ok", message, data)


def _manual_required(check_id: str, message: str, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _check(check_id, "manual_required", message, data)


def _failed(check_id: str, message: str, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _check(check_id, "failed", message, data)


def _check(check_id: str, status: str, message: str, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    result = {"id": check_id, "status": status, "message": message}
    if data:
        result["data"] = data
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dev_tools.agent.skills.checks.milestone8_checkpoint",
        description="Run the private Milestone 8 Stage 5 two-conversion-skill checkpoint.",
    )
    parser.add_argument("--repo-root", default=".", help="repository root")
    parser.add_argument("--benchmark-evidence", help="JSON file with Codex/Claude conversion benchmark evidence")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="output format")
    args = parser.parse_args(argv)

    result = run_milestone8_checkpoint(args.repo_root, benchmark_evidence=args.benchmark_evidence)
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_text_result(result)
    if result["status"] == "ok":
        return 0
    if result["status"] == "incomplete":
        return 2
    return 1


def _print_text_result(result: dict[str, Any]) -> None:
    print(f"milestone 8 checkpoint: {result['status']}")
    for check in result["checks"]:
        print(f"{check['status'].upper()} {check['id']}: {check['message']}")


if __name__ == "__main__":
    raise SystemExit(main())
