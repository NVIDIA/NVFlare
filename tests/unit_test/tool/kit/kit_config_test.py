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


def _make_poc_admin_startup_kit(parent: Path, name: str = "admin@nvidia.com") -> Path:
    kit_dir = parent / name
    startup_dir = kit_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_admin.json").write_text('{"admin": {"username": "", "uid_source": "cert"}}\n')
    (startup_dir / "client.crt").write_text("dummy client cert\n")
    (startup_dir / "client.key").write_text("dummy client key\n")
    (startup_dir / "rootCA.pem").write_text("dummy root ca\n")
    return kit_dir


def _make_site_startup_kit(parent: Path, name: str = "site-1") -> Path:
    kit_dir = parent / name
    startup_dir = kit_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_client.json").write_text("{}\n")
    return kit_dir


def _make_invalid_startup_kit(parent: Path, name: str = "invalid@nvidia.com") -> Path:
    kit_dir = parent / name
    (kit_dir / "startup").mkdir(parents=True)
    (kit_dir / "startup" / "fed_admin.json").write_text('{"admin": {"username": "%s"}}\n' % name)
    return kit_dir


def _entry_path(config, kit_id: str):
    try:
        value = config.get(f'startup_kits.entries."{kit_id}"')
    except Exception:
        return None
    return Path(value).resolve()


def _write_cli_config(home: Path, text: str) -> Path:
    config_dir = home / ".nvflare"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.conf"
    config_path.write_text(text.strip() + "\n")
    return config_path


def _assert_error_contains(exc_info, *parts: str):
    message = str(exc_info.value)
    hint = getattr(exc_info.value, "hint", None) or ""
    message = f"{message}\n{hint}"
    for part in parts:
        assert part in message


class TestStartupKitRegistryConfig:
    """Tests for the proposed nvflare.tool.kit.kit_config public helper surface."""

    def test_add_email_id_is_quoted_when_saved_and_never_activates(self, tmp_path, monkeypatch):
        from nvflare.tool.kit import kit_config

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        kit_dir = _make_admin_startup_kit(tmp_path, "lead@nvidia.com")

        config = CF.parse_string("version = 2")
        updated = kit_config.add_startup_kit_entry(config, "lead@nvidia.com", str(kit_dir))
        assert _entry_path(updated, "lead@nvidia.com") == kit_dir.resolve()
        assert updated.get("startup_kits.active", None) is None

        kit_config.save_cli_config(updated)

        config_path = home / ".nvflare" / "config.conf"
        persisted = config_path.read_text()
        assert '"lead@nvidia.com"' in persisted

        loaded = CF.parse_file(str(config_path))
        assert Path(loaded.get('startup_kits.entries."lead@nvidia.com"')).resolve() == kit_dir.resolve()
        assert loaded.get("startup_kits.active", None) is None

    def test_duplicate_id_requires_force_and_force_replaces_only_registration(self, tmp_path):
        from nvflare.tool.kit import kit_config

        old_kit = _make_admin_startup_kit(tmp_path / "old", "admin@nvidia.com")
        new_kit = _make_admin_startup_kit(tmp_path / "new", "admin@nvidia.com")
        config = CF.parse_string(
            """
            version = 2
            startup_kits {
              active = "active_admin"
            }
            """
        )

        config = kit_config.add_startup_kit_entry(config, "active_admin", str(old_kit))
        with pytest.raises(Exception) as exc_info:
            kit_config.add_startup_kit_entry(config, "active_admin", str(new_kit))
        _assert_error_contains(exc_info, "active_admin")

        updated = kit_config.add_startup_kit_entry(config, "active_admin", str(new_kit), force=True)
        assert _entry_path(updated, "active_admin") == new_kit.resolve()
        assert updated.get("startup_kits.active") == "active_admin"
        assert old_kit.exists()

    def test_set_active_validates_registered_path(self, tmp_path):
        from nvflare.tool.kit import kit_config

        kit_dir = _make_admin_startup_kit(tmp_path, "admin@nvidia.com")
        config = CF.parse_string("version = 2")
        config = kit_config.add_startup_kit_entry(config, "project_admin", str(kit_dir))

        updated = kit_config.set_active_startup_kit(config, "project_admin")

        assert updated.get("startup_kits.active") == "project_admin"

    def test_set_active_rejects_unknown_id_without_mutating_active(self, tmp_path):
        from nvflare.tool.kit import kit_config

        kit_dir = _make_admin_startup_kit(tmp_path, "admin@nvidia.com")
        config = CF.parse_string(
            """
            version = 2
            startup_kits {
              active = "project_admin"
            }
            """
        )
        config = kit_config.add_startup_kit_entry(config, "project_admin", str(kit_dir))

        with pytest.raises(Exception) as exc_info:
            kit_config.set_active_startup_kit(config, "missing_admin")

        _assert_error_contains(exc_info, "missing_admin")
        assert config.get("startup_kits.active") == "project_admin"

    def test_site_startup_kit_can_be_registered_but_not_activated(self, tmp_path):
        from nvflare.tool.kit import kit_config

        site_kit = _make_site_startup_kit(tmp_path, "site-1")
        config = CF.parse_string("version = 2")

        updated = kit_config.add_startup_kit_entry(config, "site-1", str(site_kit))

        assert _entry_path(updated, "site-1") == site_kit.resolve()
        status, normalized_path, metadata = kit_config.get_startup_kit_status(str(site_kit))
        assert status == "ok"
        assert Path(normalized_path) == site_kit.resolve()
        assert metadata["identity"] == "site-1"
        assert metadata["kind"] == "site"

        with pytest.raises(Exception) as exc_info:
            kit_config.set_active_startup_kit(updated, "site-1")

        _assert_error_contains(exc_info, "admin use")
        assert updated.get("startup_kits.active", None) is None


class TestStartupKitResolution:
    def test_env_var_resolution_prefers_valid_env_path(self, tmp_path, monkeypatch):
        from nvflare.tool.kit.kit_config import resolve_startup_kit_dir

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        kit_dir = _make_admin_startup_kit(tmp_path, "env_admin@nvidia.com")
        monkeypatch.setenv("NVFLARE_STARTUP_KIT_DIR", str(kit_dir))

        assert Path(resolve_startup_kit_dir()).resolve() == kit_dir.resolve()

    def test_env_var_resolution_reports_missing_path(self, tmp_path, monkeypatch):
        from nvflare.tool.kit.kit_config import resolve_startup_kit_dir

        missing_path = tmp_path / "missing"
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("NVFLARE_STARTUP_KIT_DIR", str(missing_path))

        with pytest.raises(Exception) as exc_info:
            resolve_startup_kit_dir()

        _assert_error_contains(exc_info, "NVFLARE_STARTUP_KIT_DIR", str(missing_path))
        assert "missing" in str(exc_info.value).lower() or "does not exist" in str(exc_info.value).lower()

    def test_env_var_resolution_reports_invalid_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.kit.kit_config import resolve_startup_kit_dir

        invalid_kit = _make_invalid_startup_kit(tmp_path, "env_invalid@nvidia.com")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("NVFLARE_STARTUP_KIT_DIR", str(invalid_kit))

        with pytest.raises(Exception) as exc_info:
            resolve_startup_kit_dir()

        _assert_error_contains(exc_info, "NVFLARE_STARTUP_KIT_DIR", str(invalid_kit))
        assert "valid startup kit" in str(exc_info.value).lower()

    def test_active_missing_reports_kit_use_hint(self, tmp_path, monkeypatch):
        from nvflare.tool.kit.kit_config import resolve_startup_kit_dir

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
        _write_cli_config(
            home,
            """
            version = 2
            startup_kits {
              entries {}
            }
            """,
        )

        with pytest.raises(Exception) as exc_info:
            resolve_startup_kit_dir()

        _assert_error_contains(exc_info, "active startup kit", "nvflare config kit use")

    def test_active_id_absent_reports_kit_list_and_use_hint(self, tmp_path, monkeypatch):
        from nvflare.tool.kit.kit_config import resolve_startup_kit_dir

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
        _write_cli_config(
            home,
            """
            version = 2
            startup_kits {
              active = "cancer_lead"
              entries {}
            }
            """,
        )

        with pytest.raises(Exception) as exc_info:
            resolve_startup_kit_dir()

        _assert_error_contains(exc_info, "cancer_lead", "nvflare config kit list", "nvflare config kit use")

    def test_active_path_missing_reports_registered_id_and_path(self, tmp_path, monkeypatch):
        from nvflare.tool.kit.kit_config import resolve_startup_kit_dir

        missing_path = tmp_path / "missing_admin"
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
        _write_cli_config(
            home,
            f"""
            version = 2
            startup_kits {{
              active = "cancer_lead"
              entries {{
                cancer_lead = "{missing_path}"
              }}
            }}
            """,
        )

        with pytest.raises(Exception) as exc_info:
            resolve_startup_kit_dir()

        _assert_error_contains(exc_info, "cancer_lead", str(missing_path), "nvflare config kit use")
        assert "missing" in str(exc_info.value).lower() or "does not exist" in str(exc_info.value).lower()

    def test_active_path_invalid_reports_registered_id_and_path(self, tmp_path, monkeypatch):
        from nvflare.tool.kit.kit_config import resolve_startup_kit_dir

        invalid_kit = _make_invalid_startup_kit(tmp_path, "bad_admin@nvidia.com")
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
        _write_cli_config(
            home,
            f"""
            version = 2
            startup_kits {{
              active = "bad_admin"
              entries {{
                bad_admin = "{invalid_kit}"
              }}
            }}
            """,
        )

        with pytest.raises(Exception) as exc_info:
            resolve_startup_kit_dir()

        _assert_error_contains(exc_info, "bad_admin", str(invalid_kit))
        assert "valid startup kit" in str(exc_info.value).lower()

    def test_resolve_admin_user_falls_back_to_cert_identity_for_poc_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.kit.kit_config import resolve_admin_user_and_dir_from_startup_kit

        kit_dir = _make_poc_admin_startup_kit(tmp_path, "admin@nvidia.com")

        def fake_metadata(path):
            assert Path(path).resolve() == kit_dir.resolve()
            return {"identity": "admin@nvidia.com", "cert_role": "project_admin"}

        monkeypatch.setattr("nvflare.tool.kit.kit_config.inspect_startup_kit_metadata", fake_metadata)

        username, resolved_dir = resolve_admin_user_and_dir_from_startup_kit(str(kit_dir))

        assert username == "admin@nvidia.com"
        assert Path(resolved_dir).resolve() == kit_dir.resolve()

    def test_resolution_ignores_legacy_target_startup_kit_keys(self, tmp_path, monkeypatch):
        from nvflare.tool.kit.kit_config import resolve_startup_kit_dir

        legacy_kit = _make_admin_startup_kit(tmp_path, "legacy_admin@nvidia.com")
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
        _write_cli_config(
            home,
            f"""
            version = 2
            poc {{
              startup_kit = "{legacy_kit}"
            }}
            prod {{
              startup_kit = "{legacy_kit}"
            }}
            """,
        )

        with pytest.raises(Exception) as exc_info:
            resolve_startup_kit_dir()

        _assert_error_contains(exc_info, "active startup kit", "nvflare config kit use")
