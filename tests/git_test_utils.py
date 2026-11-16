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

import shutil
import subprocess
from pathlib import Path


def _repo_root():
    return str(Path(__file__).resolve().parents[1])


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _versioned_repository(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "NVFlare Test")
    _git(repository, "config", "user.email", "nvflare-test@example.com")
    for relative in ("versioneer.py", "setup.cfg", "nvflare/_version.py"):
        target = repository / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copy(Path(_repo_root()) / relative, target)
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "test source")
    _git(repository, "tag", "2.10.0dev0")
    return repository, _git(repository, "rev-parse", "HEAD")


def require_signing_in_user_config(monkeypatch, tmp_path):
    """Exercise temporary Git operations with signing required by the invoking user."""
    config = tmp_path / "git-user-config"
    config.write_text(
        "[commit]\n    gpgsign = true\n[tag]\n    gpgsign = true\n[gpg]\n    program = /nonexistent-nvflare-test-signing-program\n"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
