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

from nvflare.tool.agent.inspection import skill_discovery


def _write_skill(root, directory, *, name, description):
    skill_dir = root / directory
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        f'---\nname: "{name}"\ndescription: "{description}"\n---\n',
        encoding="utf-8",
    )
    return skill_dir


def test_discover_installed_skills_preserves_scope_order_and_deduplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(skill_discovery, "_GLOBAL_SKILL_DIRS", ())
    _write_skill(tmp_path, ".claude/skills/alpha", name="alpha", description="First")
    _write_skill(tmp_path, ".agents/skills/alpha-copy", name="alpha", description="Duplicate")
    _write_skill(tmp_path, ".agents/skills/beta", name="beta", description="Second")
    target = tmp_path / "train.py"
    target.write_text("import torch\n", encoding="utf-8")

    skills = skill_discovery.discover_installed_skills(target)

    assert [(item["name"], item["scope"]) for item in skills] == [
        ("alpha", "project"),
        ("beta", "project"),
    ]
    assert skills[0]["description"] == "First"


def test_discover_installed_skills_ignores_invalid_and_symlinked_skill_files(monkeypatch, tmp_path):
    monkeypatch.setattr(skill_discovery, "_GLOBAL_SKILL_DIRS", ())
    invalid = tmp_path / ".agents/skills/invalid"
    invalid.mkdir(parents=True)
    invalid.joinpath("SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")
    valid = _write_skill(tmp_path, ".agents/skills/valid", name="valid", description="Valid")
    linked = tmp_path / ".agents/skills/linked"
    linked.mkdir()
    linked.joinpath("SKILL.md").symlink_to(valid / "SKILL.md")

    skills = skill_discovery.discover_installed_skills(tmp_path)

    assert [item["name"] for item in skills] == ["valid"]
