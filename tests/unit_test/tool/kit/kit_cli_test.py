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

import argparse
import datetime
import json
import sys
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pyhocon import ConfigFactory as CF


def _make_admin_startup_kit(parent: Path, name: str = "admin@nvidia.com") -> Path:
    kit_dir = parent / name
    startup_dir = kit_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_admin.json").write_text('{"admin": {"username": "%s"}}\n' % name)
    (startup_dir / "client.crt").write_text("dummy client cert\n")
    (startup_dir / "client.key").write_text("dummy client key\n")
    (startup_dir / "rootCA.pem").write_text("dummy root ca\n")
    return kit_dir


def _make_invalid_startup_kit(parent: Path, name: str = "invalid@nvidia.com") -> Path:
    kit_dir = parent / name
    (kit_dir / "startup").mkdir(parents=True)
    (kit_dir / "startup" / "fed_admin.json").write_text('{"admin": {"username": "%s"}}\n' % name)
    return kit_dir


def _make_site_startup_kit(parent: Path, name: str = "site-1") -> Path:
    kit_dir = parent / name
    startup_dir = kit_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_client.json").write_text("{}\n")
    return kit_dir


def _write_client_cert(
    kit_dir: Path,
    *,
    common_name: str = "lead@nvidia.com",
    org: str = "NVIDIA",
    role: str = "lead",
    issuer_project: str = "CancerProject",
    expires_delta: datetime.timedelta = datetime.timedelta(days=90),
):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    not_valid_after = now + expires_delta
    not_valid_before = min(now - datetime.timedelta(minutes=1), not_valid_after - datetime.timedelta(days=1))
    cert = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name(
                [
                    x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
                    x509.NameAttribute(NameOID.UNSTRUCTURED_NAME, role),
                ]
            )
        )
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_project)]))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_valid_before.replace(tzinfo=None))
        .not_valid_after(not_valid_after.replace(tzinfo=None))
        .sign(key, hashes.SHA256())
    )
    (kit_dir / "startup" / "client.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _run_kit_command(argv, monkeypatch):
    from nvflare.cli import def_config_parser, handle_config_cmd

    root = argparse.ArgumentParser(prog="nvflare")
    subparsers = root.add_subparsers(dest="sub_command")
    def_config_parser(subparsers)
    argv = [str(arg) for arg in argv]
    monkeypatch.setattr(sys, "argv", ["nvflare", "config", *argv])
    args = root.parse_args(["config", *argv])
    handle_config_cmd(args)
    return args


def _read_config(home: Path):
    return CF.parse_file(str(home / ".nvflare" / "config.conf"))


def _entry(config, kit_id: str):
    try:
        return Path(config.get(f'startup_kits.entries."{kit_id}"')).resolve()
    except Exception:
        return None


def _write_config(home: Path, text: str) -> Path:
    config_dir = home / ".nvflare"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.conf"
    config_path.write_text(text.strip() + "\n")
    return config_path


@pytest.fixture(autouse=True)
def _isolated_cli(monkeypatch, tmp_path):
    from nvflare.tool import cli_output

    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
    monkeypatch.setattr(cli_output, "_output_format", "txt")
    return home


class TestKitCli:
    def test_parser_accepts_all_config_startup_kit_subcommands_and_schema(self):
        from nvflare.cli import def_config_parser

        root = argparse.ArgumentParser(prog="nvflare")
        subparsers = root.add_subparsers(dest="sub_command")
        def_config_parser(subparsers)

        assert root.parse_args(["config", "add", "admin", "/tmp/startup", "--force"]).force is True
        assert root.parse_args(["config", "use", "admin"]).config_sub_cmd == "use"
        assert root.parse_args(["config", "inspect"]).config_sub_cmd == "inspect"
        assert root.parse_args(["config", "list"]).config_sub_cmd == "list"
        assert root.parse_args(["config", "remove", "admin"]).config_sub_cmd == "remove"
        assert root.parse_args(["config", "inspect", "--schema"]).schema is True
        with pytest.raises(SystemExit):
            root.parse_args(["config", "show"])

    @pytest.mark.parametrize(
        ("cmd_name", "mutating", "idempotent"),
        [
            ("inspect", False, True),
            ("list", False, True),
            ("use", True, True),
        ],
    )
    def test_agent_facing_config_schema_includes_command_contract_metadata(
        self, cmd_name, mutating, idempotent, capsys
    ):
        from unittest.mock import MagicMock, patch

        from nvflare.cli import def_config_parser
        from nvflare.tool.kit import kit_cli as kit_cli_mod

        root = argparse.ArgumentParser(prog="nvflare")
        subparsers = root.add_subparsers(dest="sub_command")
        def_config_parser(subparsers)

        with patch("sys.argv", ["nvflare", "config", cmd_name, "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                getattr(kit_cli_mod, f"cmd_kit_{cmd_name}")(MagicMock())

        assert exc_info.value.code == 0
        schema = json.loads(capsys.readouterr().out)
        assert schema["output_modes"] == ["json"]
        assert schema["streaming"] is False
        assert schema["mutating"] is mutating
        assert schema["idempotent"] is idempotent
        assert schema["retry_token"] == {"supported": False}

    def test_root_config_command_prints_current_config_without_usage_error(self, monkeypatch, capsys):
        _run_kit_command([], monkeypatch)

        out = capsys.readouterr().out
        assert "config_file:" in out
        assert "Invalid arguments" not in out

    def test_add_use_show_list_remove_flow_and_add_never_activates(self, tmp_path, monkeypatch, capsys):
        home = Path.home()
        kit_dir = _make_admin_startup_kit(tmp_path, "admin@nvidia.com")
        _write_client_cert(kit_dir, common_name="admin@nvidia.com", role="project_admin")

        _run_kit_command(["add", "admin@nvidia.com", kit_dir], monkeypatch)
        out = capsys.readouterr().out
        assert "registered startup kit: admin@nvidia.com" in out or "registered_startup_kit: admin@nvidia.com" in out
        assert "next_step: nvflare config use admin@nvidia.com" in out

        config = _read_config(home)
        assert _entry(config, "admin@nvidia.com") == kit_dir.resolve()
        assert config.get("startup_kits.active", None) is None

        _run_kit_command(["inspect"], monkeypatch)
        out = capsys.readouterr().out
        assert "No active startup kit" in out
        assert "nvflare config use <id>" in out

        _run_kit_command(["use", "admin@nvidia.com"], monkeypatch)
        out = capsys.readouterr().out
        assert "active startup kit: admin@nvidia.com" in out or "active_startup_kit: admin@nvidia.com" in out
        assert str(kit_dir) in out
        assert _read_config(home).get("startup_kits.active") == "admin@nvidia.com"

        _run_kit_command(["inspect"], monkeypatch)
        out = capsys.readouterr().out
        assert "active  id" in out
        assert "status" in out
        assert "identity" in out
        assert "cert_role" in out
        assert "path" in out
        assert "*       admin@nvidia.com" in out
        assert "ok" in out
        assert "admin@nvidia.com" in out
        assert "project_admin" in out
        assert str(kit_dir) in out
        assert "config_file:" in out
        assert "active:" not in out
        assert "cert_role:" not in out

        _run_kit_command(["list"], monkeypatch)
        out = capsys.readouterr().out
        assert "*" in out
        assert "admin@nvidia.com" in out
        assert "ok" in out

        _run_kit_command(["remove", "admin@nvidia.com"], monkeypatch)
        out = capsys.readouterr().out
        assert "removed startup kit: admin@nvidia.com" in out or "removed_startup_kit: admin@nvidia.com" in out
        assert "no active startup kit" in out.lower()
        assert "nvflare config use <id>" in out

        config = _read_config(home)
        assert config.get("startup_kits.active", None) is None
        assert _entry(config, "admin@nvidia.com") is None
        assert kit_dir.exists()

    def test_duplicate_id_fails_without_force_and_force_replaces_registration(self, tmp_path, monkeypatch):
        home = Path.home()
        first_kit = _make_admin_startup_kit(tmp_path / "first", "admin@nvidia.com")
        replacement_kit = _make_admin_startup_kit(tmp_path / "replacement", "admin@nvidia.com")

        _run_kit_command(["add", "project_admin", first_kit], monkeypatch)

        with pytest.raises(SystemExit) as exc_info:
            _run_kit_command(["add", "project_admin", replacement_kit], monkeypatch)
        assert exc_info.value.code == 4
        assert _entry(_read_config(home), "project_admin") == first_kit.resolve()

        _run_kit_command(["add", "project_admin", replacement_kit, "--force"], monkeypatch)

        assert _entry(_read_config(home), "project_admin") == replacement_kit.resolve()
        assert first_kit.exists()

    def test_email_id_is_persisted_with_hocon_quoting(self, tmp_path, monkeypatch):
        home = Path.home()
        kit_dir = _make_admin_startup_kit(tmp_path, "lead@nvidia.com")

        _run_kit_command(["add", "lead@nvidia.com", kit_dir], monkeypatch)

        config_path = home / ".nvflare" / "config.conf"
        assert '"lead@nvidia.com"' in config_path.read_text()
        config = CF.parse_file(str(config_path))
        assert Path(config.get('startup_kits.entries."lead@nvidia.com"')).resolve() == kit_dir.resolve()

    @pytest.mark.parametrize("bad_path_kind", ["missing", "invalid"])
    def test_add_rejects_missing_or_invalid_startup_kit_paths(self, tmp_path, monkeypatch, bad_path_kind):
        if bad_path_kind == "missing":
            kit_path = tmp_path / "missing"
        else:
            kit_path = _make_invalid_startup_kit(tmp_path, "bad_admin@nvidia.com")

        with pytest.raises(SystemExit) as exc_info:
            _run_kit_command(["add", "bad_admin", kit_path], monkeypatch)

        assert exc_info.value.code == 4
        assert not (Path.home() / ".nvflare" / "config.conf").exists()

    def test_list_marks_stale_entries_without_failing(self, tmp_path, monkeypatch, capsys):
        home = Path.home()
        valid_kit = _make_admin_startup_kit(tmp_path, "admin@nvidia.com")
        site_kit = _make_site_startup_kit(tmp_path, "site-1")
        missing_kit = tmp_path / "missing_admin"
        invalid_kit = _make_invalid_startup_kit(tmp_path, "invalid_admin@nvidia.com")
        _write_config(
            home,
            f"""
            version = 2
            startup_kits {{
              active = "missing_admin"
              entries {{
                project_admin = "{valid_kit}"
                "site-1" = "{site_kit}"
                missing_admin = "{missing_kit}"
                invalid_admin = "{invalid_kit}"
              }}
            }}
            """,
        )

        _run_kit_command(["list"], monkeypatch)
        out = capsys.readouterr().out

        assert "project_admin" in out
        assert "ok" in out
        assert "site-1" in out
        assert "missing_admin" in out
        assert "missing" in out
        assert "invalid_admin" in out
        assert "invalid" in out
        assert any(line.startswith("*") and "missing_admin" in line for line in out.splitlines())

    def test_inspect_and_list_json_include_identity_certificate_and_findings(
        self, tmp_path, monkeypatch, capsys, _isolated_cli
    ):
        from nvflare.tool import cli_output

        home = _isolated_cli
        valid_kit = _make_admin_startup_kit(tmp_path, "lead@nvidia.com")
        _write_client_cert(
            valid_kit,
            common_name="lead@nvidia.com",
            org="NVIDIA",
            role="lead",
            issuer_project="CancerProject",
            expires_delta=datetime.timedelta(days=90),
        )
        missing_kit = tmp_path / "missing_admin"
        _write_config(
            home,
            f"""
            version = 2
            startup_kits {{
              active = "project_admin"
              entries {{
                project_admin = "{valid_kit}"
                missing_admin = "{missing_kit}"
              }}
            }}
            """,
        )
        monkeypatch.setattr(cli_output, "_output_format", "json")

        _run_kit_command(["inspect"], monkeypatch)
        inspect_payload = json.loads(capsys.readouterr().out)
        inspect_data = inspect_payload["data"]
        assert inspect_data["status"] == "ok"
        assert inspect_data["identity"] == "lead@nvidia.com"
        assert inspect_data["cert_role"] == "lead"
        assert inspect_data["role"] == "lead"
        assert inspect_data["org"] == "NVIDIA"
        assert inspect_data["project"] == "CancerProject"
        assert inspect_data["certificate"]["status"] == "ok"
        assert inspect_data["findings"] == []

        _run_kit_command(["list"], monkeypatch)
        list_payload = json.loads(capsys.readouterr().out)
        rows = {row["id"]: row for row in list_payload["data"]}
        assert rows["project_admin"]["org"] == "NVIDIA"
        assert rows["project_admin"]["project"] == "CancerProject"
        assert rows["project_admin"]["certificate"]["status"] == "ok"
        assert rows["missing_admin"]["status"] == "missing"
        assert any(f["code"] == "STARTUP_KIT_PATH_MISSING" for f in rows["missing_admin"]["findings"])

    def test_use_json_warns_that_active_kit_is_global_state(self, tmp_path, monkeypatch, capsys):
        from nvflare.tool import cli_output

        kit_dir = _make_admin_startup_kit(tmp_path, "admin@nvidia.com")
        _write_client_cert(kit_dir, common_name="admin@nvidia.com", role="project_admin")
        _run_kit_command(["add", "project_admin", kit_dir], monkeypatch)
        capsys.readouterr()
        monkeypatch.setattr(cli_output, "_output_format", "json")

        _run_kit_command(["use", "project_admin"], monkeypatch)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert any(f["code"] == "CONFIG_USE_MUTATES_GLOBAL_STATE" for f in payload["data"]["findings"])

    def test_inspect_json_does_not_print_env_warning(self, tmp_path, monkeypatch, capsys):
        from nvflare.tool import cli_output

        kit_dir = _make_admin_startup_kit(tmp_path, "admin@nvidia.com")
        _write_config(
            Path.home(),
            f"""
            version = 2
            startup_kits {{
              active = "project_admin"
              entries {{
                project_admin = "{kit_dir}"
              }}
            }}
            """,
        )
        monkeypatch.setenv("NVFLARE_STARTUP_KIT_DIR", str(kit_dir))
        monkeypatch.setattr(cli_output, "_output_format", "json")

        _run_kit_command(["inspect"], monkeypatch)

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["status"] == "ok"
        assert captured.err == ""

    def test_use_unknown_or_stale_registration_does_not_change_active(self, tmp_path, monkeypatch):
        home = Path.home()
        valid_kit = _make_admin_startup_kit(tmp_path, "admin@nvidia.com")
        invalid_kit = _make_invalid_startup_kit(tmp_path, "invalid_admin@nvidia.com")
        _write_config(
            home,
            f"""
            version = 2
            startup_kits {{
              active = "project_admin"
              entries {{
                project_admin = "{valid_kit}"
                invalid_admin = "{invalid_kit}"
              }}
            }}
            """,
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_kit_command(["use", "missing_admin"], monkeypatch)
        assert exc_info.value.code == 4
        assert _read_config(home).get("startup_kits.active") == "project_admin"

        with pytest.raises(SystemExit) as exc_info:
            _run_kit_command(["use", "invalid_admin"], monkeypatch)
        assert exc_info.value.code == 4
        assert _read_config(home).get("startup_kits.active") == "project_admin"
