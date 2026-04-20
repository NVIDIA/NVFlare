# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from unittest.mock import patch

import pytest
from pyhocon import ConfigFactory as CF

from nvflare.utils.cli_utils import (
    append_if_not_in_list,
    backup_hidden_config_file,
    create_startup_kit_config,
    ensure_hidden_config_migrated,
    get_hidden_nvflare_config_path,
    get_hidden_nvflare_dir,
    get_startup_kit_dir_for_target,
    load_hidden_config,
    migrate_config_to_v2,
    persist_hidden_config_migration,
    save_config,
)


class TestCLIUtils:
    def test_get_hidden_nvflare_dir(self):
        hidden_dir = get_hidden_nvflare_dir()
        assert str(hidden_dir) == str(Path.home() / ".nvflare")

    def test_get_hidden_nvflare_config_path(self):
        assert get_hidden_nvflare_config_path(str(get_hidden_nvflare_dir())) == str(
            Path.home() / ".nvflare/config.conf"
        )

    def test_create_startup_kit_config(self):
        with patch("nvflare.utils.cli_utils.check_startup_dir", side_effect=None) as mock:
            mock.return_value = ""
            with patch("os.path.isdir", side_effect=None) as mock1:
                mock1.return_value = True
                prev_conf = CF.parse_string(
                    """
                        version = 2
                        poc {
                            workspace = "/tmp/nvflare/poc"
                        }
                    """
                )
                config = create_startup_kit_config(
                    nvflare_config=prev_conf,
                    target="prod",
                    startup_kit_dir="/tmp/nvflare/poc/example_project/prod_00",
                )

                assert "/tmp/nvflare/poc" == config.get("poc.workspace")
                assert "/tmp/nvflare/poc/example_project/prod_00" == config.get("prod.startup_kit")

                config = create_startup_kit_config(nvflare_config=prev_conf, target="prod", startup_kit_dir="")

                assert config.get("prod.startup_kit", None) is None

    def test_migrate_config_to_v2_copies_legacy_startup_and_workspace(self):
        legacy = CF.parse_string(
            """
                startup_kit {
                    path = "/tmp/nvflare/legacy/prod_00"
                }
                poc_workspace {
                    path = "/tmp/nvflare/poc"
                }
            """
        )

        config = migrate_config_to_v2(legacy)

        assert 2 == config.get("version")
        assert "/tmp/nvflare/legacy/prod_00" == config.get("poc.startup_kit")
        assert "/tmp/nvflare/poc" == config.get("poc.workspace")
        config_dict = config.as_plain_ordered_dict()
        assert "version" in config_dict
        assert "config_version" not in config_dict
        assert "startup_kit" not in config_dict
        assert "poc_workspace" not in config_dict
        assert "prod" not in config_dict

    def test_get_startup_kit_dir_for_target_prefers_env_override(self, monkeypatch):
        monkeypatch.setenv("NVFLARE_STARTUP_KIT_DIR", "/tmp/override")
        with patch("nvflare.utils.cli_utils.check_startup_dir"):
            assert "/tmp/override" == get_startup_kit_dir_for_target(target="prod")

    def test_get_startup_kit_dir_for_target_reads_targeted_v2_config(self):
        config = CF.parse_string(
            """
                version = 2
                poc {
                    startup_kit = "/tmp/nvflare/poc/prod_00"
                }
                prod {
                    startup_kit = "/tmp/nvflare/prod/admin@nvidia.com"
                }
            """
        )
        with patch("nvflare.utils.cli_utils.load_hidden_config", return_value=config):
            with patch("nvflare.utils.cli_utils.check_startup_dir"):
                assert "/tmp/nvflare/poc/prod_00" == get_startup_kit_dir_for_target(target="poc")
                assert "/tmp/nvflare/prod/admin@nvidia.com" == get_startup_kit_dir_for_target(target="prod")

    def test_load_hidden_config_migrates_in_memory_without_persisting(self, tmp_path, monkeypatch):
        hidden_dir = tmp_path / ".nvflare"
        hidden_dir.mkdir()
        config_path = hidden_dir / "config.conf"
        legacy_text = """
startup_kit {
  path = "/tmp/nvflare/legacy/prod_00"
}
poc_workspace {
  path = "/tmp/nvflare/poc"
}
""".strip()
        config_path.write_text(legacy_text)

        monkeypatch.setattr("nvflare.utils.cli_utils.get_or_create_hidden_nvflare_dir", lambda: hidden_dir)

        migrated = load_hidden_config()

        assert migrated.get_int("version") == 2
        persisted = config_path.read_text()
        assert persisted.strip() == legacy_text
        assert not (hidden_dir / "config.conf.bak").exists()

    def test_persist_hidden_config_migration_writes_backup_and_migrated_config(self, tmp_path):
        hidden_dir = tmp_path / ".nvflare"
        hidden_dir.mkdir()
        config_path = hidden_dir / "config.conf"
        legacy_text = """
startup_kit {
  path = "/tmp/nvflare/legacy/prod_00"
}
poc_workspace {
  path = "/tmp/nvflare/poc"
}
""".strip()
        config_path.write_text(legacy_text)

        migrated = migrate_config_to_v2(CF.parse_string(legacy_text))
        persist_hidden_config_migration(str(config_path), migrated)

        persisted = config_path.read_text()
        assert "version = 2" in persisted
        assert "config_version" not in persisted
        assert "startup_kit {" not in persisted
        backup_path = hidden_dir / "config.conf.bak"
        assert backup_path.exists()
        assert backup_path.read_text().strip() == legacy_text

    def test_ensure_hidden_config_migrated_persists_legacy_file_once(self, tmp_path, monkeypatch):
        hidden_dir = tmp_path / ".nvflare"
        hidden_dir.mkdir()
        config_path = hidden_dir / "config.conf"
        legacy_text = """
startup_kit {
  path = "/tmp/nvflare/legacy/prod_00"
}
poc_workspace {
  path = "/tmp/nvflare/poc"
}
""".strip()
        config_path.write_text(legacy_text)

        monkeypatch.setattr("nvflare.utils.cli_utils.get_or_create_hidden_nvflare_dir", lambda: hidden_dir)

        ensure_hidden_config_migrated()

        persisted = config_path.read_text()
        assert "version = 2" in persisted
        assert "startup_kit {" not in persisted
        assert (hidden_dir / "config.conf.bak").read_text().strip() == legacy_text

    def test_backup_hidden_config_file_returns_none_when_source_missing(self, tmp_path):
        hidden_dir = tmp_path / ".nvflare"
        hidden_dir.mkdir()
        config_path = hidden_dir / "config.conf"

        backup_path = backup_hidden_config_file(str(config_path))

        assert backup_path is None
        assert not (hidden_dir / "config.conf.bak").exists()

    def test_save_config_replaces_file_atomically_without_temp_leak(self, tmp_path):
        config_path = tmp_path / "config.conf"
        config_path.write_text("version = 1\n")
        config = CF.parse_string(
            """
                version = 2
                poc {
                    startup_kit = "/tmp/nvflare/poc/prod_00"
                }
            """
        )

        save_config(config, str(config_path))

        persisted = config_path.read_text()
        assert "version = 2" in persisted
        assert 'startup_kit = "/tmp/nvflare/poc/prod_00"' in persisted
        assert list(tmp_path.glob("tmp*")) == []

    @pytest.mark.parametrize(
        "inputs, result", [(([], "a"), ["a"]), ((["a"], "a"), ["a"]), ((["a", "b"], "b"), ["a", "b"])]
    )
    def test_append_if_not_in_list(self, inputs, result):
        arr, item = inputs
        assert result == append_if_not_in_list(arr, item)
