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

"""CoCo external-runtime contracts; run serially outside unit workers."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from tests.coco_test_utils.process import run_coco_command

ROOT = Path(__file__).resolve().parents[4]

SOURCE = ROOT / "examples/devops/coco"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo package validation requires Python 3.11+")
def test_working_tree_validation_checks_new_guides_and_pending_deletions(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "examples/devops/coco"
    shutil.copytree(SOURCE, source)
    # The source-checkout guide links to the canonical Trustee documentation.
    reference = Path("nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md")
    (repo / reference).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE.parents[2] / reference, repo / reference)
    run_coco_command(["git", "init", "-q", str(repo)], check=True)
    obsolete = source / "obsolete.md"
    obsolete.write_text("Retired guide\n")
    run_coco_command(["git", "-C", str(repo), "add", "."], check=True)
    obsolete.unlink()
    guide = source / "NEW-GUIDE.md"
    guide.write_text("[Missing guide](missing-guide.md)\n")
    command = [sys.executable, str(source / "validate-package.py"), "--working-tree"]
    result = run_coco_command(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "NEW-GUIDE.md: missing-guide.md" in result.stderr
    guide.write_text("[Start](README.md)\n")
    result = run_coco_command(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo package validation requires Python 3.11+")
def test_working_tree_validation_rejects_untracked_symlinks(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "examples/devops/coco"
    shutil.copytree(SOURCE, source)
    run_coco_command(["git", "init", "-q", str(repo)], check=True)
    (source / "linked.md").symlink_to("README.md")
    result = run_coco_command(
        [sys.executable, str(source / "validate-package.py"), "--working-tree"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Source must contain regular files: linked.md" in result.stderr


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo package validation requires Python 3.11+")
@pytest.mark.parametrize("case", ["missing-git", "archive", "ignored-copy"])
def test_working_tree_validation_reports_unsupported_sources(tmp_path, case):
    repo = tmp_path / "repo"
    source = repo / "copy"
    source.mkdir(parents=True)
    shutil.copyfile(SOURCE / "validate-package.py", source / "validate-package.py")
    (source / "README.md").write_text("Package source\n")
    env = os.environ.copy()
    if case == "missing-git":
        env["PATH"] = str(tmp_path / "no-tools")
        expected = "Git is required for package validation"
    elif case == "archive":
        expected = "requires an NVFlare Git source checkout"
    else:
        run_coco_command(["git", "init", "-q", str(repo)], check=True)
        (repo / ".gitignore").write_text("copy/\n")
        expected = "Package sources are excluded from Git"
    result = run_coco_command(
        [sys.executable, str(source / "validate-package.py")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert expected in result.stderr
    assert "Traceback" not in result.stderr
    assert "Validated" not in result.stdout
