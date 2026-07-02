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

from nvflare.tool.agent.doctor import doctor_environment, format_doctor_human


def test_doctor_reports_conversion_scoped_readiness(monkeypatch, tmp_path):
    # Doctor is conversion-scoped: nvflare import, the agent command surface,
    # optional ML framework availability, and the installed skill bundle.
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    data = doctor_environment()

    assert data["nvflare"]["import_ok"] is True
    assert {"nvflare", "commands", "optional_dependencies", "skills", "findings", "status"} <= set(data)
    # Running doctor does not create local CLI config.
    assert not home.joinpath(".nvflare", "config.conf").exists()


def test_doctor_excludes_deployment_and_poc_checks(monkeypatch, tmp_path):
    # Deployment/POC readiness (startup kits, POC workspace, live-server status)
    # is intentionally out of the conversion-only scope and not reported.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    data = doctor_environment()

    assert "startup_kits" not in data
    assert "poc" not in data
    assert "online" not in data


def test_doctor_command_registry_matches_agent_command_surface(monkeypatch, tmp_path):
    from nvflare.tool.agent.command_registry import agent_commands

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    data = doctor_environment()

    commands = data["commands"]["commands"]
    command_names = {item["command"] for item in commands}
    assert commands == agent_commands()
    assert "nvflare agent skills install" in command_names
    assert "nvflare agent skills list" in command_names
    assert "nvflare agent skills performance" not in command_names
    assert "nvflare agent skills benchmark" not in command_names


def test_doctor_reports_optional_dependency_and_skill_status(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    data = doctor_environment()

    dep_names = {dep["name"] for dep in data["optional_dependencies"]}
    assert {"torch", "tensorflow", "sklearn"} <= dep_names
    assert "status" in data["skills"]


def test_doctor_human_summary_omits_deployment_sections(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    text = format_doctor_human(doctor_environment())

    assert "NVFLARE Agent Doctor" in text
    assert "startup kits:" not in text
    assert "poc:" not in text
    assert "online:" not in text
