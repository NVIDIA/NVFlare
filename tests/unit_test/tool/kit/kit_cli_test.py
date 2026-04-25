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
import sys
from pathlib import Path

import pytest
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


def _run_kit_command(argv, monkeypatch):
    from nvflare.cli import def_config_parser, handle_config_cmd

    root = argparse.ArgumentParser(prog="nvflare")
    subparsers = root.add_subparsers(dest="sub_command")
    def_config_parser(subparsers)
    argv = [str(arg) for arg in argv]
    monkeypatch.setattr(sys, "argv", ["nvflare", "config", "kit", *argv])
    args = root.parse_args(["config", "kit", *argv])
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
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
    monkeypatch.setattr(cli_output, "_output_format", "txt")
    return home


class TestKitCli:
    def test_parser_accepts_all_kit_subcommands_and_schema(self):
        from nvflare.cli import def_config_parser

        root = argparse.ArgumentParser(prog="nvflare")
        subparsers = root.add_subparsers(dest="sub_command")
        def_config_parser(subparsers)

        assert root.parse_args(["config", "kit", "add", "admin", "/tmp/startup", "--force"]).force is True
        assert root.parse_args(["config", "kit", "use", "admin"]).kit_sub_cmd == "use"
        assert root.parse_args(["config", "kit", "show"]).kit_sub_cmd == "show"
        assert root.parse_args(["config", "kit", "list"]).kit_sub_cmd == "list"
        assert root.parse_args(["config", "kit", "remove", "admin"]).kit_sub_cmd == "remove"
        assert root.parse_args(["config", "kit", "show", "--schema"]).schema is True

    def test_root_kit_command_prints_help_without_usage_error(self, monkeypatch, capsys):
        _run_kit_command([], monkeypatch)

        out = capsys.readouterr().out
        assert "usage: nvflare config kit" in out
        assert "kit subcommands" in out
        assert "Invalid arguments" not in out

    def test_add_use_show_list_remove_flow_and_add_never_activates(self, tmp_path, monkeypatch, capsys):
        home = Path.home()
        kit_dir = _make_admin_startup_kit(tmp_path, "admin@nvidia.com")

        _run_kit_command(["add", "admin@nvidia.com", kit_dir], monkeypatch)
        out = capsys.readouterr().out
        assert "registered startup kit: admin@nvidia.com" in out or "registered_startup_kit: admin@nvidia.com" in out
        assert "next_step: nvflare config kit use admin@nvidia.com" in out

        config = _read_config(home)
        assert _entry(config, "admin@nvidia.com") == kit_dir.resolve()
        assert config.get("startup_kits.active", None) is None

        _run_kit_command(["show"], monkeypatch)
        out = capsys.readouterr().out
        assert "No active startup kit" in out
        assert "nvflare config kit use <id>" in out

        _run_kit_command(["use", "admin@nvidia.com"], monkeypatch)
        out = capsys.readouterr().out
        assert "active startup kit: admin@nvidia.com" in out or "active_startup_kit: admin@nvidia.com" in out
        assert str(kit_dir) in out
        assert _read_config(home).get("startup_kits.active") == "admin@nvidia.com"

        _run_kit_command(["list"], monkeypatch)
        out = capsys.readouterr().out
        assert "*" in out
        assert "admin@nvidia.com" in out
        assert "ok" in out

        _run_kit_command(["remove", "admin@nvidia.com"], monkeypatch)
        out = capsys.readouterr().out
        assert "removed startup kit: admin@nvidia.com" in out or "removed_startup_kit: admin@nvidia.com" in out
        assert "no active startup kit" in out.lower()
        assert "nvflare config kit use <id>" in out

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
