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

import pytest
from pyhocon import ConfigFactory as CF

from nvflare.utils.cli_utils import (
    append_if_not_in_list,
    backup_hidden_config_file,
    find_startup_kit_config_keys,
    get_hidden_nvflare_config_path,
    get_hidden_nvflare_dir,
    load_hidden_config,
    remove_startup_kit_config_keys,
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

    def test_remove_startup_kit_config_keys_preserves_registry(self):
        config = CF.parse_string(
            """
                version = 2
                startup_kits {
                    active = "admin"
                    entries {
                        admin = "/tmp/admin"
                    }
                }
                poc {
                    startup_kit = "/tmp/old-poc"
                    workspace = "/tmp/nvflare/poc"
                }
                prod {
                    startup_kit = "/tmp/old-prod"
                }
            """
        )

        updated = remove_startup_kit_config_keys(config)

        assert updated.get("startup_kits.active") == "admin"
        assert updated.get("startup_kits.entries.admin") == "/tmp/admin"
        assert updated.get("poc.workspace") == "/tmp/nvflare/poc"
        assert updated.get("poc.startup_kit", None) is None
        assert updated.get("prod.startup_kit", None) is None

    def test_find_startup_kit_config_keys_reports_removed_keys(self):
        config = CF.parse_string(
            """
                startup_kits {
                    active = "admin"
                }
                poc {
                    startup_kit = "/tmp/old-poc"
                    workspace = "/tmp/nvflare/poc"
                }
                prod {
                    startup_kit = "/tmp/old-prod"
                }
            """
        )

        assert find_startup_kit_config_keys(config) == ["poc.startup_kit", "prod.startup_kit"]

    def test_load_hidden_config_reads_without_migration_or_persistence(self, tmp_path, monkeypatch):
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

        loaded = load_hidden_config()

        assert loaded.get("startup_kit.path") == "/tmp/nvflare/legacy/prod_00"
        assert loaded.get("poc_workspace.path") == "/tmp/nvflare/poc"
        persisted = config_path.read_text()
        assert persisted.strip() == legacy_text
        assert not (hidden_dir / "config.conf.bak").exists()

    def test_backup_hidden_config_file_returns_none_when_source_missing(self, tmp_path):
        hidden_dir = tmp_path / ".nvflare"
        hidden_dir.mkdir()
        config_path = hidden_dir / "config.conf"

        backup_path = backup_hidden_config_file(str(config_path))

        assert backup_path is None
        assert not (hidden_dir / "config.conf.bak").exists()

    def test_backup_hidden_config_file_uses_non_overwriting_suffixes(self, tmp_path):
        config_path = tmp_path / "config.conf"
        config_path.write_text("current\n")
        backup_path = tmp_path / "config.conf.bak"
        backup1_path = tmp_path / "config.conf.bak1"
        backup_path.write_text("older\n")
        backup1_path.write_text("older1\n")

        new_backup_path = backup_hidden_config_file(str(config_path))

        assert new_backup_path == str(tmp_path / "config.conf.bak2")
        assert Path(new_backup_path).read_text() == "current\n"
        assert backup_path.read_text() == "older\n"
        assert backup1_path.read_text() == "older1\n"

    def test_save_config_replaces_file_atomically_without_temp_leak(self, tmp_path):
        config_path = tmp_path / "config.conf"
        config_path.write_text("version = 1\n")
        config = CF.parse_string(
            """
                version = 2
                poc {
                    workspace = "/tmp/nvflare/poc"
                }
            """
        )

        save_config(config, str(config_path))

        persisted = config_path.read_text()
        assert "version = 2" in persisted
        assert 'workspace = "/tmp/nvflare/poc"' in persisted
        assert list(tmp_path.glob("tmp*")) == []

    @pytest.mark.parametrize(
        "inputs, result", [(([], "a"), ["a"]), ((["a"], "a"), ["a"]), ((["a", "b"], "b"), ["a", "b"])]
    )
    def test_append_if_not_in_list(self, inputs, result):
        arr, item = inputs
        assert result == append_if_not_in_list(arr, item)
