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

"""Validate source-file boundaries without modifying the user's index."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
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
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    obsolete = source / "obsolete.md"
    obsolete.write_text("Retired guide\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    obsolete.unlink()
    guide = source / "NEW-GUIDE.md"
    guide.write_text("[Missing guide](missing-guide.md)\n")
    command = [sys.executable, str(source / "validate-package.py"), "--working-tree"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "NEW-GUIDE.md: missing-guide.md" in result.stderr
    guide.write_text("[Start](README.md)\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo package validation requires Python 3.11+")
def test_working_tree_validation_rejects_untracked_symlinks(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "examples/devops/coco"
    shutil.copytree(SOURCE, source)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (source / "linked.md").symlink_to("README.md")
    result = subprocess.run(
        [sys.executable, str(source / "validate-package.py"), "--working-tree"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Source must contain regular files: linked.md" in result.stderr
