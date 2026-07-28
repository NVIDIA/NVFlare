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
from pathlib import Path

import pytest

from nvflare.tool.agent.inspection import python_scanner
from nvflare.tool.agent.inspector import inspect_path


def test_inspect_directory_reports_inspected_target_path(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root)

    assert data["path"] == str(root.resolve(strict=False))
    assert data["path"] != "."


def test_inspect_file_reports_inspected_target_path(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import torch\n", encoding="utf-8")

    data = inspect_path(script)

    assert data["path"] == str(script.resolve(strict=False))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_inspect_symlink_reports_link_path_without_resolving_target(tmp_path):
    target_dir = tmp_path / "outside"
    target_dir.mkdir()
    (target_dir / "train.py").write_text("import tensorflow\n", encoding="utf-8")
    link_dir = tmp_path / "linked-repo"
    link_dir.symlink_to(target_dir, target_is_directory=True)

    data = inspect_path(link_dir)

    assert data["path"] == os.path.abspath(os.path.normpath(str(link_dir)))
    assert data["path"] != str(target_dir.resolve(strict=False))
    assert data["scan"]["files_skipped"][0]["code"] == "SYMLINK_SKIPPED"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_inspect_symlinked_file_does_not_classify_target(tmp_path):
    target_file = tmp_path / "outside.py"
    target_file.write_text("import torch\n", encoding="utf-8")
    link_file = tmp_path / "linked-train.py"
    link_file.symlink_to(target_file)

    data = inspect_path(link_file)

    assert data["target_type"] == "unknown_target"
    assert data["frameworks"] == []
    assert data["scan"]["files_skipped"] == [
        {
            "code": "SYMLINK_SKIPPED",
            "path": link_file.name,
            "target": "<REDACTED_PATH>",
            "message": "symlink was not followed during static inspection",
        }
    ]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_inspect_dangling_symlink_is_reported_as_skipped(tmp_path):
    link_file = tmp_path / "dangling-train.py"
    link_file.symlink_to(tmp_path / "missing.py")

    data = inspect_path(link_file)

    assert data["target_type"] == "unknown_target"
    assert data["scan"]["files_skipped"][0]["code"] == "SYMLINK_SKIPPED"
    assert data["scan"]["files_skipped"][0]["path"] == link_file.name


def test_inspect_redacts_secret_literals_and_absolute_paths_by_default(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "API_TOKEN = 'super-secret-value'\n" "DATA_ROOT = '/Users/alice/private/data'\n" "import tensorflow as tf\n",
        encoding="utf-8",
    )

    data = inspect_path(script)
    dumped = json.dumps(data)

    assert "super-secret-value" not in dumped
    assert "/Users/alice/private/data" not in dumped
    assert "<REDACTED>" in dumped
    assert "<REDACTED_PATH>" in dumped
    assert data["frameworks"][0]["name"] == "tensorflow"
    assert {finding["code"] for finding in data["findings"]} == {"SECRET_LITERAL_REDACTED"}
    assert data["patterns"]["absolute_data_paths"][0]["code"] == "ABSOLUTE_DATA_PATH"


def test_inspect_redaction_can_be_disabled_for_local_debugging(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("PASSWORD = 'local-debug-secret'\nDATA_ROOT = '/opt/data'\n", encoding="utf-8")

    data = inspect_path(script, redact=False)
    dumped = json.dumps(data)

    assert "local-debug-secret" in dumped
    assert "/opt/data" in dumped


def test_inspect_skips_symlink_without_scanning_target(tmp_path):
    target = tmp_path / "outside.py"
    target.write_text("import tensorflow\n", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "linked.py").symlink_to(target)
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root)

    assert [framework["name"] for framework in data["frameworks"]] == ["pytorch"]
    assert data["scan"]["files_skipped"][0]["code"] == "SYMLINK_SKIPPED"


def test_inspect_bom_prefixed_source_still_detects_framework(tmp_path):
    # A leading UTF-8 BOM (Windows/Notepad-authored source) must not blind the
    # inspector: it should still parse and detect the framework, not degrade to a
    # parse error with no evidence.
    script = tmp_path / "train.py"
    script.write_text(
        "﻿import torch\n"
        "\n"
        "\n"
        "class Net(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    Net()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert "nvflare-convert-pytorch" in data["skill_selection"]["recommended_skills"]
    assert not any(finding["code"] == "PYTHON_PARSE_ERROR" for finding in data["findings"])


def test_inspect_skips_deep_ast_and_continues_classifying_other_files(tmp_path, monkeypatch):
    (tmp_path / "generated.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "train.py").write_text(
        "import torch\n\n" "class Net(torch.nn.Module):\n" "    pass\n",
        encoding="utf-8",
    )
    original_visit = python_scanner._PythonInspector.visit

    def raise_for_generated_file(visitor, tree):
        if visitor.rel_path == "generated.py":
            raise RecursionError("AST depth exceeded")
        return original_visit(visitor, tree)

    monkeypatch.setattr(python_scanner._PythonInspector, "visit", raise_for_generated_file)

    data = inspect_path(tmp_path)

    assert data["classification_incomplete"] is True
    assert data["skill_selection"]["detected_framework"] == "pytorch"
    assert "nvflare-convert-pytorch" in data["skill_selection"]["recommended_skills"]
    findings = [finding for finding in data["findings"] if finding["code"] == "PYTHON_AST_DEPTH_LIMIT"]
    assert findings == [
        {
            "code": "PYTHON_AST_DEPTH_LIMIT",
            "severity": "warning",
            "file": "generated.py",
            "line": None,
            "message": "Python file exceeds the safe static-inspection AST depth.",
        }
    ]


def test_inspect_name_only_job_py_without_flare_evidence_is_not_flare_job(tmp_path):
    # A plain training repo that happens to have a launcher named job.py (a common
    # SLURM filename) and no nvflare imports must route to conversion, not be
    # misclassified as an existing FLARE job.
    (tmp_path / "job.py").write_text(
        "import torch\n\n\nclass Net(torch.nn.Module):\n    pass\n\n\ndef main():\n    return Net()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["conversion_state"] == "not_converted"
    assert data["target_type"] == "training_repository"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_inspect_simenv_call_without_flare_evidence_is_not_flare_job(tmp_path):
    # SimEnv is a natural class name in RL/robotics code; a call to a local SimEnv
    # with no nvflare imports must not be classified as a FLARE job.
    (tmp_path / "train.py").write_text(
        "class SimEnv:\n    pass\n\n\ndef main():\n    env = SimEnv()\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["job"]["sim_env_used"] is True
    assert data["conversion_state"] != "flare_job"
    assert data["target_type"] != "flare_job_source"


def test_inspect_export_command_requires_flare_evidence(tmp_path):
    # `.export` calls over-match (torch.onnx.export); without nvflare evidence the
    # inspector must not ship a `job.py --export` command that would fail argparse.
    (tmp_path / "job.py").write_text(
        "import torch\n\n\ndef main():\n    torch.onnx.export(None, (), 'm.onnx')\n",
        encoding="utf-8",
    )

    data = inspect_path(tmp_path)

    assert data["job"]["export_support"] is True
    assert "python job.py --export --export-dir <job-dir>" not in data["recommended_next_commands"]


def test_inspect_does_not_treat_builtin_compile_as_torch_compile(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n" "\n" "def build_code():\n" "    return compile('x = 1', '<inline>', 'exec')\n",
        encoding="utf-8",
    )

    data = inspect_path(script)

    assert data["frameworks"][0]["name"] == "pytorch"
    assert not any(item["kind"] == "torch_compile" for item in data["patterns"]["dynamic"])


def test_inspect_stops_and_caps_skips_after_file_limit(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(20):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    assert data["classification_incomplete"] is True
    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_considered"] == 20
    assert data["scan"]["files_scanned"] == 3
    assert data["scan"]["files_skipped_count"] == 17
    assert data["scan"]["files_skipped_count_approximate"] is False
    assert data["scan"]["files_skipped_truncated"] is True
    assert data["scan"]["files_skipped_evidence_truncated"] is True
    assert len(data["scan"]["files_skipped"]) == 12
    assert data["scan"]["files_skipped"][0] == {
        "code": "FILE_LIMIT_REACHED",
        "path": "train_03.py",
        "message": "file scan limit reached",
    }
    assert data["scan"]["files_skipped"][-1]["path"] == "train_14.py"
    assert data["skill_selection"]["recommended_skills"] == ["nvflare-orient"]


def test_inspect_exact_file_limit_without_unvisited_files_is_complete(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(3):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    assert data["classification_incomplete"] is False
    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_considered"] == 3
    assert data["scan"]["files_scanned"] == 3
    assert data["scan"]["files_skipped_count"] == 0
    assert data["scan"]["files_skipped_count_approximate"] is False
    assert data["scan"]["files_skipped_truncated"] is False
    assert data["scan"]["files_skipped_evidence_truncated"] is False


def test_inspect_file_limit_accounting_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr("nvflare.tool.agent.inspection.scanner.MAX_FILE_LIMIT_ACCOUNTED_SKIPS", 3)
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(10):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=1)

    assert data["classification_incomplete"] is True
    assert data["scan"]["entries_visited"] == 1
    assert data["scan"]["files_considered"] == 4
    assert data["scan"]["files_scanned"] == 1
    assert data["scan"]["files_skipped_count"] == 3
    assert data["scan"]["files_skipped_count_approximate"] is True
    assert data["scan"]["files_skipped_truncated"] is True
    assert data["scan"]["files_skipped"] == [
        {"code": "FILE_LIMIT_REACHED", "path": "train_01.py", "message": "file scan limit reached"},
        {"code": "FILE_LIMIT_REACHED", "path": "train_02.py", "message": "file scan limit reached"},
        {"code": "FILE_LIMIT_REACHED", "path": "train_03.py", "message": "file scan limit reached"},
    ]


def test_inspect_file_limit_unreadable_directory_accounting_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr("nvflare.tool.agent.inspection.scanner.MAX_FILE_LIMIT_ACCOUNTED_SKIPS", 3)
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(5):
        (root / f"a_{index:02d}").mkdir()
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    original_iterdir = Path.iterdir

    def fake_iterdir(path):
        if path.name.startswith("a_"):
            raise OSError("blocked")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    data = inspect_path(root, max_files=1)

    assert data["classification_incomplete"] is True
    assert data["scan"]["entries_visited"] == 1
    assert data["scan"]["files_considered"] == 1
    assert data["scan"]["files_scanned"] == 1
    assert data["scan"]["files_skipped_count"] == 3
    assert data["scan"]["files_skipped_count_approximate"] is True
    assert data["scan"]["files_skipped_truncated"] is True
    assert data["scan"]["files_skipped"] == [
        {
            "code": "DIRECTORY_NOT_SCANNED_FILE_LIMIT",
            "path": "a_00",
            "message": "directory not scanned because file scan limit was reached",
        },
        {
            "code": "UNREADABLE_DIRECTORY",
            "path": "a_00",
            "message": "could not read directory",
            "error_type": "OSError",
        },
        {
            "code": "DIRECTORY_NOT_SCANNED_FILE_LIMIT",
            "path": "a_01",
            "message": "directory not scanned because file scan limit was reached",
        },
    ]


def test_inspect_file_limit_records_unvisited_stack_directories(tmp_path):
    root = tmp_path / "repo"
    nested = root / "a_nested"
    nested.mkdir(parents=True)
    (nested / "train_nested.py").write_text("import torch\n", encoding="utf-8")
    for index in range(5):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    skipped = {(entry["code"], entry["path"]) for entry in data["scan"]["files_skipped"]}
    assert ("FILE_LIMIT_REACHED", "train_03.py") in skipped
    assert ("FILE_LIMIT_REACHED", "train_04.py") in skipped
    assert ("DIRECTORY_NOT_SCANNED_FILE_LIMIT", "a_nested") in skipped
    assert ("FILE_LIMIT_REACHED", "a_nested/train_nested.py") in skipped
    assert data["classification_incomplete"] is True
    assert data["scan"]["files_considered"] == 6
    assert data["scan"]["files_skipped_count"] == 4


def test_inspect_file_limit_records_pending_directories_when_last_child_reaches_limit(tmp_path):
    root = tmp_path / "repo"
    nested = root / "a_nested"
    nested.mkdir(parents=True)
    (nested / "train_nested.py").write_text("import torch\n", encoding="utf-8")
    for index in range(3):
        (root / f"train_{index:02d}.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    skipped = {(entry["code"], entry["path"]) for entry in data["scan"]["files_skipped"]}
    assert ("DIRECTORY_NOT_SCANNED_FILE_LIMIT", "a_nested") in skipped
    assert ("FILE_LIMIT_REACHED", "a_nested/train_nested.py") in skipped
    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_considered"] == 4
    assert data["scan"]["files_scanned"] == 3
    assert data["scan"]["files_skipped_count"] == 2


def test_inspect_file_limit_counts_non_python_entries(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(5):
        (root / f"metadata_{index:02d}.json").write_text("{}\n", encoding="utf-8")
    (root / "train.py").write_text("import torch\n", encoding="utf-8")

    data = inspect_path(root, max_files=3)

    assert data["classification_incomplete"] is True
    assert data["scan"]["entries_visited"] == 3
    assert data["scan"]["files_considered"] == 6
    assert data["scan"]["files_scanned"] == 0
    assert data["scan"]["files_skipped"] == [
        {"code": "FILE_LIMIT_REACHED", "path": "metadata_03.json", "message": "file scan limit reached"},
        {"code": "FILE_LIMIT_REACHED", "path": "metadata_04.json", "message": "file scan limit reached"},
        {"code": "FILE_LIMIT_REACHED", "path": "train.py", "message": "file scan limit reached"},
    ]


def test_inspect_benign_directory_skip_does_not_self_recommend_orient(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "train.py").write_text("import torch\n", encoding="utf-8")
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    data = inspect_path(root)

    assert data["classification_incomplete"] is False
    assert data["scan"]["files_skipped_count"] == 1
    assert data["scan"]["files_skipped"] == [
        {"code": "DIRECTORY_SKIPPED", "path": ".git", "message": "directory skipped"}
    ]
    assert data["skill_selection"]["recommended_skills"] == []
