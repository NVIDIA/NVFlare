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
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nvflare.tool.agent.doctor import (
    MAX_ONLINE_STATUS_ITEMS,
    _inspect_doctor_startup_kit,
    _load_status_admin_config,
    _new_doctor_status_session,
    _read_bounded_regular_file,
    doctor_environment,
    format_doctor_human,
)


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


def test_doctor_preserves_schema_v1_and_reports_conversion_readiness(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    data = doctor_environment(online=False, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["schema_version"] == "1"
    assert data["nvflare"]["import_ok"] is True
    assert {
        "nvflare",
        "commands",
        "startup_kits",
        "optional_dependencies",
        "skills",
        "poc",
        "online",
        "findings",
        "status",
    } <= set(data)
    assert data["online"] == {"enabled": False, "status": "not_requested"}
    assert data["startup_kits"]["active_id"] is None
    assert any(finding["code"] == "STARTUP_KIT_NOT_CONFIGURED" for finding in data["findings"])
    assert data["conversion_status"] == "ok"
    assert data["deployment_status"] == "attention"
    assert data["status_scope"] == "conversion"
    assert data["status"] == "ok"
    # Running doctor does not create local CLI config.
    assert not home.joinpath(".nvflare", "config.conf").exists()


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
    with patch("nvflare.tool.agent.doctor._new_doctor_status_session", return_value=fake_session):
        data = doctor_environment(online=True, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["online"]["status"] == "ok"
    assert data["online"]["server_status"] == "running"
    assert data["online"]["clients"] == [{"client_name": "site-1"}]
    assert fake_session.closed is True


def test_doctor_status_config_rejects_symlink_without_following_it(tmp_path):
    kit = tmp_path / "admin"
    startup = kit / "startup"
    startup.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{"admin": {"username": "attacker"}}', encoding="utf-8")
    (startup / "fed_admin.json").symlink_to(outside)

    with pytest.raises(ValueError, match="regular file"):
        _load_status_admin_config(kit)


def test_doctor_bounded_reader_rejects_hard_link(tmp_path):
    source = tmp_path / "source.json"
    linked = tmp_path / "linked.json"
    source.write_text('{"admin": {}}', encoding="utf-8")
    os.link(source, linked)

    with pytest.raises(ValueError, match="hard-linked"):
        _read_bounded_regular_file(linked, 1024)


@pytest.mark.skipif(
    os.open not in getattr(os, "supports_dir_fd", set())
    or not getattr(os, "O_NOFOLLOW", 0)
    or not getattr(os, "O_DIRECTORY", 0),
    reason="anchored no-follow directory descriptors are not supported",
)
def test_doctor_bounded_reader_anchors_parent_during_directory_swap(tmp_path):
    kit = tmp_path / "kit"
    startup = kit / "startup"
    startup.mkdir(parents=True)
    target = startup / "fed_admin.json"
    target.write_text('{"admin": {"username": "original"}}', encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.joinpath("fed_admin.json").write_text(
        '{"admin": {"username": "outside"}}',
        encoding="utf-8",
    )
    moved = kit / "startup-original"
    real_open = os.open
    raced = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal raced
        if path == "fed_admin.json" and dir_fd is not None and not raced:
            raced = True
            startup.rename(moved)
            startup.symlink_to(outside, target_is_directory=True)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    with patch("nvflare.tool.agent.doctor.os.open", side_effect=racing_open) as mocked_open:
        with patch(
            "nvflare.tool.agent.doctor.os.supports_dir_fd",
            {*os.supports_dir_fd, mocked_open},
        ):
            with pytest.raises(ValueError, match="changed while being read"):
                _read_bounded_regular_file(target, 1024)

    assert raced is True


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are not supported on this platform")
def test_doctor_bounded_reader_rejects_fifo_without_blocking(tmp_path):
    fifo = tmp_path / "fed_admin.json"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="regular file"):
        _read_bounded_regular_file(fifo, 1024)


def test_doctor_status_session_ignores_custom_handlers_and_file_transfer(tmp_path):
    from nvflare.fuel.flare_api.flare_api import Session

    kit = tmp_path / "admin"
    startup = kit / "startup"
    custom = kit / "local" / "custom"
    startup.mkdir(parents=True)
    custom.mkdir(parents=True)
    sentinel = tmp_path / "handler-executed"
    custom.joinpath("evil_handler.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    for name in ("client.key", "client.crt", "rootCA.pem"):
        startup.joinpath(name).write_text("fixture", encoding="utf-8")
    startup.joinpath("fed_admin.json").write_text(
        json.dumps(
            {
                "admin": {
                    "username": "admin@nvidia.com",
                    "client_key": "client.key",
                    "client_cert": "client.crt",
                    "ca_cert": "rootCA.pem",
                    "download_dir": "new-transfer",
                },
                "handlers": [{"path": "evil_handler.EvilHandler"}],
            }
        ),
        encoding="utf-8",
    )

    copied_credentials = []

    class FakeAdminAPI:
        closed = False

        def __init__(self, **kwargs):
            assert kwargs["cmd_modules"] == []
            assert kwargs["event_handlers"] == []
            copied_credentials.extend(kwargs["admin_config"][key] for key in ("client_key", "client_cert", "ca_cert"))

        def logout(self):
            self.closed = True

    original_sys_path = list(sys.path)
    with (
        patch("nvflare.fuel.hci.client.api.AdminAPI", FakeAdminAPI),
        patch.object(Session, "try_connect", return_value=None),
    ):
        session = _new_doctor_status_session({"path": str(kit)}, timeout=0.1)
        session.close()

    assert not sentinel.exists()
    assert sys.path == original_sys_path
    assert not (kit / "new-transfer").exists()
    assert all(not Path(path).exists() for path in copied_credentials)


def test_doctor_status_session_falls_back_to_certificate_identity(tmp_path):
    from nvflare.fuel.flare_api.flare_api import Session

    kit = tmp_path / "admin"
    startup = kit / "startup"
    startup.mkdir(parents=True)
    for name in ("client.key", "client.crt", "rootCA.pem"):
        startup.joinpath(name).write_text("fixture", encoding="utf-8")
    startup.joinpath("fed_admin.json").write_text(
        json.dumps(
            {
                "admin": {
                    "client_key": "client.key",
                    "client_cert": "client.crt",
                    "ca_cert": "rootCA.pem",
                    "uid_source": "cert",
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeAdminAPI:
        def __init__(self, **kwargs):
            assert kwargs["user_name"] == "cert-admin"

        def logout(self):
            return None

    with (
        patch("nvflare.tool.agent.doctor._certificate_identity", return_value="cert-admin"),
        patch("nvflare.fuel.hci.client.api.AdminAPI", FakeAdminAPI),
        patch.object(Session, "try_connect", return_value=None),
    ):
        session = _new_doctor_status_session({"path": str(kit)}, timeout=0.1)
        session.close()


def test_doctor_startup_summary_preserves_certificate_rca_metadata(tmp_path):
    from cryptography.hazmat.primitives import serialization

    from nvflare.lighter.utils import Identity, generate_cert, generate_keys

    kit = tmp_path / "admin"
    startup = kit / "startup"
    startup.mkdir(parents=True)
    root_key, root_public = generate_keys()
    expired = generate_cert(
        subject=Identity("admin@nvidia.com", "nvidia", "project_admin"),
        issuer=Identity("example-project", "nvidia"),
        signing_pri_key=root_key,
        subject_pub_key=root_public,
        not_valid_before=datetime.now(timezone.utc) - timedelta(days=10),
        not_valid_after=datetime.now(timezone.utc) - timedelta(days=1),
    )
    startup.joinpath("client.crt").write_bytes(expired.public_bytes(serialization.Encoding.PEM))
    startup.joinpath("rootCA.pem").write_text("fixture", encoding="utf-8")
    startup.joinpath("fed_admin.json").write_text(
        json.dumps({"admin": {"username": "admin@nvidia.com", "client_cert": "client.crt"}}),
        encoding="utf-8",
    )

    status, normalized, metadata = _inspect_doctor_startup_kit(str(kit))

    assert status == "invalid"
    assert normalized == str(kit)
    assert metadata["identity"] == "admin@nvidia.com"
    assert metadata["cert_role"] == "project_admin"
    assert metadata["role"] == "project_admin"
    assert metadata["org"] == "nvidia"
    assert metadata["project"] == "example-project"
    assert metadata["certificate"]["status"] == "expired"
    assert any(finding["code"] == "STARTUP_KIT_CERT_EXPIRED" for finding in metadata["findings"])


def test_doctor_startup_summary_distinguishes_missing_registration(tmp_path):
    missing = tmp_path / "missing-admin"

    status, normalized, metadata = _inspect_doctor_startup_kit(str(missing))

    assert status == "missing"
    assert normalized is None
    assert metadata["findings"][0]["code"] == "STARTUP_KIT_PATH_MISSING"


def test_doctor_offline_rejects_hocon_include_without_reading_target(monkeypatch, tmp_path):
    home = tmp_path / "home"
    config_dir = home / ".nvflare"
    config_dir.mkdir(parents=True)
    outside = tmp_path / "outside.conf"
    outside.write_text('poc.workspace = "/secret/outside"\n', encoding="utf-8")
    config_dir.joinpath("config.conf").write_text(
        f'include url("{outside.as_uri()}")\nversion = 2\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    with patch("pyhocon.ConfigFactory.parse_URL", side_effect=AssertionError("include followed")) as parse_url:
        data = doctor_environment()

    parse_url.assert_not_called()
    assert data["poc"]["workspace"] != "/secret/outside"
    assert any(finding["code"] == "STARTUP_KIT_CONFIG_INVALID" for finding in data["findings"])


def test_doctor_direct_kit_rejects_oversized_config_before_general_loader(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    kit = tmp_path / "admin"
    startup = kit / "startup"
    startup.mkdir(parents=True)
    for name in ("client.crt", "rootCA.pem"):
        startup.joinpath(name).write_text("fixture", encoding="utf-8")
    startup.joinpath("fed_admin.json").write_text(
        json.dumps({"admin": {"username": "admin", "padding": "x" * (1024 * 1024)}}),
        encoding="utf-8",
    )

    with patch(
        "nvflare.fuel.utils.config_factory.ConfigFactory.load_config",
        side_effect=AssertionError("unbounded loader used"),
    ) as general_loader:
        data = doctor_environment(
            online=True,
            args=SimpleNamespace(kit_id=None, startup_kit=str(kit)),
        )

    general_loader.assert_not_called()
    assert data["online"]["status"] == "error"
    assert "exceeds" in data["online"]["findings"][0]["message"]


def test_doctor_online_bounds_remote_status(monkeypatch, tmp_path):
    _configure_active_startup_kit(tmp_path, monkeypatch)

    class FakeSession:
        def check_status(self, _target_type, _targets):
            return {
                "server_status": "running",
                "clients": [],
                "client_status": [],
                "jobs": [{"job_id": str(index)} for index in range(MAX_ONLINE_STATUS_ITEMS + 5)],
            }

        def close(self):
            return None

    with patch("nvflare.tool.agent.doctor._new_doctor_status_session", return_value=FakeSession()):
        data = doctor_environment(online=True, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["online"]["status"] == "partial"
    assert len(data["online"]["jobs"]) == MAX_ONLINE_STATUS_ITEMS
    assert data["online"]["status_truncated"] is True
    assert any(finding["code"] == "ONLINE_STATUS_TRUNCATED" for finding in data["online"]["findings"])


def test_doctor_online_rejects_type_confused_remote_status(monkeypatch, tmp_path):
    _configure_active_startup_kit(tmp_path, monkeypatch)

    class FakeSession:
        def check_status(self, _target_type, _targets):
            return {"server_status": "running", "clients": "not-a-list"}

        def close(self):
            return None

    with patch("nvflare.tool.agent.doctor._new_doctor_status_session", return_value=FakeSession()):
        data = doctor_environment(online=True, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["online"]["status"] == "error"
    assert "clients must be a list" in data["online"]["findings"][0]["message"]
    assert data["conversion_status"] == "ok"
    assert data["deployment_status"] == "attention"
    assert data["status_scope"] == "conversion_and_online"
    assert data["status"] == "attention"


def test_doctor_online_close_failure_does_not_override_result(monkeypatch, tmp_path):
    _configure_active_startup_kit(tmp_path, monkeypatch)

    class FakeSession:
        def check_status(self, _target_type, _targets):
            return {"server_status": "running", "clients": [], "client_status": [], "jobs": []}

        def close(self):
            raise RuntimeError("close failed")

    with patch("nvflare.tool.agent.doctor._new_doctor_status_session", return_value=FakeSession()):
        data = doctor_environment(online=True, args=SimpleNamespace(kit_id=None, startup_kit=None))

    assert data["online"]["status"] == "partial"
    assert any(finding["code"] == "ONLINE_SESSION_CLEANUP_FAILED" for finding in data["online"]["findings"])


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
    assert {"torch", "lightning", "pytorch_lightning", "tensorflow", "sklearn"} <= dep_names
    assert "status" in data["skills"]


def test_doctor_human_summary_includes_compatible_deployment_sections(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    text = format_doctor_human(doctor_environment())

    assert "NVFLARE Agent Doctor" in text
    assert "startup kits:" in text
    assert "optional dependencies:" in text
    assert "poc:" in text
    assert "online: not_requested" in text


def test_doctor_human_counts_ok_startup_kits_as_valid():
    text = format_doctor_human(
        {
            "status": "ok",
            "nvflare": {"version": "2.8.0", "import_ok": True},
            "commands": {"status": "ok", "commands": []},
            "startup_kits": {"active_id": "admin", "entries": [{"status": "ok"}]},
            "skills": {},
            "optional_dependencies": [],
            "poc": {"status": "missing", "workspace": "/missing"},
            "online": {"enabled": False, "status": "not_requested"},
            "findings": [],
        }
    )

    assert "startup kits: 1/1 valid" in text


def test_doctor_does_not_report_regular_file_as_poc_workspace(monkeypatch, tmp_path):
    workspace_file = tmp_path / "not-a-workspace"
    workspace_file.write_text("file\n", encoding="utf-8")
    monkeypatch.setenv("NVFLARE_POC_WORKSPACE", str(workspace_file))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    data = doctor_environment()

    assert data["poc"]["status"] == "missing"
