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

from nvflare.tool.install_skills import available_skills, install_skills


def test_available_skills_includes_autofl():
    assert {"name": "nvflare-autofl", "package": "nvflare.agent.skills.autofl"} in available_skills()


def test_install_skills_without_target_is_safe(monkeypatch):
    monkeypatch.delenv("NVFLARE_SKILLS_DIR", raising=False)

    result = install_skills()

    assert result["installed"] == []
    assert result["errors"] == []
    assert result["skipped"] == [{"name": "nvflare-autofl", "reason": "no target_dir or NVFLARE_SKILLS_DIR configured"}]


def test_install_skills_copies_bundled_autofl_skill(tmp_path):
    result = install_skills(target_dir=str(tmp_path))

    skill_path = tmp_path / "nvflare-autofl" / "SKILL.md"
    assert result["errors"] == []
    assert result["installed"] == [{"name": "nvflare-autofl", "path": str(tmp_path / "nvflare-autofl")}]
    assert skill_path.exists()
    assert "NVFlare Auto-FL" in skill_path.read_text(encoding="utf-8")

    second = install_skills(target_dir=str(tmp_path))
    assert second["installed"] == []
    assert second["skipped"] == [
        {"name": "nvflare-autofl", "path": str(tmp_path / "nvflare-autofl"), "reason": "already current"}
    ]


def test_install_skills_dry_run_does_not_write(tmp_path):
    result = install_skills(target_dir=str(tmp_path), dry_run=True)

    assert result["installed"] == [{"name": "nvflare-autofl", "path": str(tmp_path / "nvflare-autofl")}]
    assert not (tmp_path / "nvflare-autofl").exists()
