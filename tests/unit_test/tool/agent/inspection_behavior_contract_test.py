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
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from tests.unit_test.tool.agent.inspection_behavior_normalizer import (
    CORPUS_ROOT,
    ENVIRONMENT_NORMALIZATIONS,
    EXPECTED_FILE,
    inspect_corpus,
    load_case_definitions,
    normalizer_sha256,
)

NORMALIZER_SCRIPT = Path(__file__).with_name("inspection_behavior_normalizer.py")


def _framework_evidence(result: dict, framework: str) -> list[dict]:
    for item in result["frameworks"]:
        if item["name"] == framework:
            return item["evidence"]
    return []


def _assert_semantic_expectation(case_id: str, result: dict, expected: dict) -> None:
    selection = result["skill_selection"]
    ownership = result["framework_ownership"]
    assert result["classification_incomplete"] is False, case_id
    assert selection["detected_framework"] == expected["detected_framework"], case_id
    assert result["conversion_state"] == expected["conversion_state"], case_id
    assert selection["recommended_skills"] == expected["recommended_skills"], case_id
    assert ownership["state"] == expected["ownership_state"], case_id
    if "owners" in expected:
        assert ownership["owners"] == expected["owners"], case_id
    if "candidates" in expected:
        assert ownership["candidates"] == expected["candidates"], case_id
    for framework, forbidden_kinds in expected.get("forbidden_evidence_kinds", {}).items():
        evidence_kinds = {item["kind"] for item in _framework_evidence(result, framework)}
        assert evidence_kinds.isdisjoint(forbidden_kinds), case_id


def test_inspection_behavior_corpus_matches_declared_semantics():
    actual = inspect_corpus()
    definitions = load_case_definitions()

    assert list(actual["cases"]) == [case["id"] for case in definitions]
    for case in definitions:
        _assert_semantic_expectation(case["id"], actual["cases"][case["id"]], case["expected"])


def test_normalizer_freezes_only_declared_environment_fields():
    assert ENVIRONMENT_NORMALIZATIONS == (
        "path",
        "nvflare_version",
        "installed_skills",
    )


def test_inspection_behavior_matches_immutable_golden():
    assert EXPECTED_FILE.is_file(), "expected.json is required and must come from the agreed reference parent commit"
    expected = json.loads(EXPECTED_FILE.read_text(encoding="utf-8"))
    assert expected["normalizer_sha256"] == normalizer_sha256()
    assert expected["environment_normalizations"] == list(ENVIRONMENT_NORMALIZATIONS)
    assert inspect_corpus(CORPUS_ROOT) == expected


def test_normalizer_cli_ignores_nvflare_editable_install_finder(tmp_path):
    reference_root = tmp_path / "reference"
    wrong_root = tmp_path / "wrong"
    corpus_root = tmp_path / "corpus"
    bootstrap_root = tmp_path / "bootstrap"
    output_file = tmp_path / "actual.json"

    for root, marker in ((reference_root, "reference"), (wrong_root, "wrong")):
        package_root = root / "nvflare"
        inspector_root = package_root / "tool" / "agent"
        inspector_root.mkdir(parents=True)
        package_root.joinpath("__init__.py").write_text("", encoding="utf-8")
        package_root.joinpath("tool", "__init__.py").write_text("", encoding="utf-8")
        inspector_root.joinpath("__init__.py").write_text("", encoding="utf-8")
        inspector_root.joinpath("inspector.py").write_text(
            textwrap.dedent(
                f"""
                MARKER = {marker!r}

                def inspect_path(path):
                    return {{
                        "path": str(path),
                        "nvflare_version": "test",
                        "installed_skills": [],
                        "marker": MARKER,
                    }}
                """
            ),
            encoding="utf-8",
        )

    corpus_root.mkdir()
    corpus_root.joinpath("cases.json").write_text(
        '{"cases": [{"id": "source", "path": "."}]}\n',
        encoding="utf-8",
    )
    bootstrap_root.mkdir()
    bootstrap_root.joinpath("sitecustomize.py").write_text(
        textwrap.dedent(
            f"""
            import importlib.util
            import sys

            class _EditableFinder:
                @classmethod
                def find_spec(cls, fullname, path=None, target=None):
                    if fullname == "nvflare":
                        return importlib.util.spec_from_file_location(
                            fullname,
                            {str(wrong_root / "nvflare" / "__init__.py")!r},
                            submodule_search_locations=[{str(wrong_root / "nvflare")!r}],
                        )
                    return None

            _EditableFinder.__module__ = "__editable___nvflare_test_finder"
            sys.meta_path.insert(0, _EditableFinder)
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(bootstrap_root)
    subprocess.run(
        [
            sys.executable,
            str(NORMALIZER_SCRIPT),
            "--source-root",
            str(reference_root),
            "--corpus",
            str(corpus_root),
            "--output",
            str(output_file),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    output = json.loads(output_file.read_text(encoding="utf-8"))
    assert output["cases"]["source"]["marker"] == "reference"
