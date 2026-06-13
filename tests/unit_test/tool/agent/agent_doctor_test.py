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

import builtins
from types import SimpleNamespace
from unittest.mock import patch

from nvflare.tool.agent.doctor import doctor_environment


def _configure_active_startup_kit(tmp_path, monkeypatch):
    home = tmp_path / "home"
    admin_dir = tmp_path / "active-admin"
    startup_dir = admin_dir / "startup"
    transfer_dir = admin_dir / "transfer"
    startup_dir.mkdir(parents=True)
    transfer_dir.mkdir()
    (startup_dir / "fed_admin.json").write_text(
        '{"admin": {"username": "admin@nvidia.com", "download_dir": "transfer"}}',
        encoding="utf-8",
    )
    (startup_dir / "client.crt").write_text("cert", encoding="utf-8")
    (startup_dir / "rootCA.pem").write_text("root", encoding="utf-8")

    config_dir = home / ".nvflare"
    config_dir.mkdir(parents=True)
    (config_dir / "config.conf").write_text(
        f"""
        version = 2
        startup_kits {{
          active = "admin@nvidia.com"
          entries {{
            "admin@nvidia.com" = "{admin_dir}"
          }}
        }}
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
    return admin_dir


def test_doctor_local_readiness_does_not_create_config(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    data = doctor_environment()

    assert data["nvflare"]["import_ok"] is True
    assert data["online"] == {"enabled": False, "status": "not_requested"}
    assert data["startup_kits"]["active_id"] is None
    assert any(finding["code"] == "STARTUP_KIT_NOT_CONFIGURED" for finding in data["findings"])
    assert not home.joinpath(".nvflare", "config.conf").exists()


def test_doctor_command_registry_matches_agent_command_surface(monkeypatch, tmp_path):
    from nvflare.tool.agent.command_registry import agent_commands

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    data = doctor_environment()

    commands = data["commands"]["commands"]
    command_names = {item["command"] for item in commands}
    assert commands == agent_commands()
    assert "nvflare agent skills install" in command_names
    assert "nvflare agent skills list" in command_names
    assert "nvflare agent skills performance" not in command_names
    assert "nvflare agent skills benchmark" not in command_names
    assert "nvflare agent skills evaluate" not in command_names


def test_doctor_poc_config_tolerates_missing_pyhocon(monkeypatch, tmp_path):
    from nvflare.tool.poc import poc_commands

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
    monkeypatch.setattr(poc_commands, "DEFAULT_WORKSPACE", str(tmp_path / "missing-poc"))
    original_import = builtins.__import__

    def import_without_pyhocon(name, *args, **kwargs):
        if name == "pyhocon":
            raise ImportError("pyhocon unavailable")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_pyhocon):
        data = doctor_environment()

    assert data["poc"]["status"] == "missing"
    assert any(finding["code"] == "POC_WORKSPACE_MISSING" for finding in data["findings"])


def test_doctor_online_uses_read_only_status_check(monkeypatch, tmp_path):
    _configure_active_startup_kit(tmp_path, monkeypatch)

    class FakeSession:
        def __init__(self):
            self.closed = False

        def check_status(self, target_type, client_names):
            assert target_type == "all"
            assert client_names is None
            return {
                "server_status": "running",
                "clients": [{"client_name": "site-1"}],
                "client_status": [],
                "jobs": [],
            }

        def close(self):
            self.closed = True

    fake_session = FakeSession()
    with patch("nvflare.tool.cli_session.new_cli_session_for_args", return_value=fake_session):
        data = doctor_environment(online=True, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["online"]["status"] == "ok"
    assert data["online"]["server_status"] == "running"
    assert data["online"]["clients"] == [{"client_name": "site-1"}]
    assert fake_session.closed is True


def test_doctor_online_reports_timeout_separately(monkeypatch, tmp_path):
    _configure_active_startup_kit(tmp_path, monkeypatch)

    class FakeSession:
        def check_status(self, target_type, client_names):
            raise TimeoutError("status timed out")

        def close(self):
            return None

    with patch("nvflare.tool.cli_session.new_cli_session_for_args", return_value=FakeSession()):
        data = doctor_environment(online=True, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["online"]["status"] == "timeout"
    assert data["online"]["findings"][0]["code"] == "ONLINE_CHECK_TIMEOUT"


def test_doctor_online_generic_error_includes_exception_type(monkeypatch, tmp_path):
    _configure_active_startup_kit(tmp_path, monkeypatch)

    class FakeSession:
        def check_status(self, target_type, client_names):
            raise RuntimeError("unexpected")

        def close(self):
            return None

    with patch("nvflare.tool.cli_session.new_cli_session_for_args", return_value=FakeSession()):
        data = doctor_environment(online=True, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["online"]["status"] == "error"
    assert data["online"]["findings"][0]["code"] == "ONLINE_CHECK_FAILED_RUNTIMEERROR"


def test_doctor_online_preflight_error_is_structured(monkeypatch, tmp_path):
    _configure_active_startup_kit(tmp_path, monkeypatch)

    with patch("nvflare.tool.agent.doctor._online_read_only_preflight", side_effect=RuntimeError("preflight failed")):
        data = doctor_environment(online=True, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["online"]["status"] == "error"
    assert data["online"]["findings"][0]["code"] == "ONLINE_PREFLIGHT_FAILED_RUNTIMEERROR"


def test_doctor_online_skips_when_session_would_create_download_dir(monkeypatch, tmp_path):
    home = tmp_path / "home"
    admin_dir = tmp_path / "active-admin"
    startup_dir = admin_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_admin.json").write_text(
        '{"admin": {"username": "admin@nvidia.com", "download_dir": "missing-transfer"}}',
        encoding="utf-8",
    )
    (startup_dir / "client.crt").write_text("cert", encoding="utf-8")
    (startup_dir / "rootCA.pem").write_text("root", encoding="utf-8")
    config_dir = home / ".nvflare"
    config_dir.mkdir(parents=True)
    (config_dir / "config.conf").write_text(
        f"""
        version = 2
        startup_kits {{
          active = "admin@nvidia.com"
          entries {{
            "admin@nvidia.com" = "{admin_dir}"
          }}
        }}
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    with patch("nvflare.tool.cli_session.new_cli_session_for_args") as new_session:
        data = doctor_environment(online=True, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["online"]["status"] == "skipped"
    assert data["online"]["findings"][0]["code"] == "ONLINE_CHECK_WOULD_CREATE_DOWNLOAD_DIR"
    new_session.assert_not_called()


def test_doctor_online_skips_invalid_download_dir_type(monkeypatch, tmp_path):
    home = tmp_path / "home"
    admin_dir = tmp_path / "active-admin"
    startup_dir = admin_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_admin.json").write_text(
        '{"admin": {"username": "admin@nvidia.com", "download_dir": ["transfer"]}}',
        encoding="utf-8",
    )
    (startup_dir / "client.crt").write_text("cert", encoding="utf-8")
    (startup_dir / "rootCA.pem").write_text("root", encoding="utf-8")
    config_dir = home / ".nvflare"
    config_dir.mkdir(parents=True)
    (config_dir / "config.conf").write_text(
        f"""
        version = 2
        startup_kits {{
          active = "admin@nvidia.com"
          entries {{
            "admin@nvidia.com" = "{admin_dir}"
          }}
        }}
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    with patch("nvflare.tool.cli_session.new_cli_session_for_args") as new_session:
        data = doctor_environment(online=True, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["online"]["status"] == "skipped"
    assert data["online"]["findings"][0]["code"] == "ONLINE_CHECK_DOWNLOAD_DIR_INVALID"
    new_session.assert_not_called()
