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
import json
from pathlib import Path

import pytest

from nvflare.utils.cli_utils import (
    append_if_not_in_list,
    get_hidden_nvflare_config_path,
    get_hidden_nvflare_dir,
    load_hidden_config,
    save_config,
)


class TestCLIUtils:
    def test_get_hidden_nvflare_dir(self):
        hidden_dir = get_hidden_nvflare_dir()
        assert str(hidden_dir) == str(Path.home() / ".nvflare")

    def test_get_hidden_nvflare_config_path(self):
        assert get_hidden_nvflare_config_path(str(get_hidden_nvflare_dir())) == str(
            Path.home() / ".nvflare/config.json"
        )

    def test_load_hidden_config_reads_without_persistence(self, tmp_path, monkeypatch):
        hidden_dir = tmp_path / ".nvflare"
        hidden_dir.mkdir()
        config_path = hidden_dir / "config.json"
        config_text = json.dumps({"poc": {"workspace": "/tmp/nvflare/poc"}})
        config_path.write_text(config_text)

        monkeypatch.setattr("nvflare.utils.cli_utils.get_or_create_hidden_nvflare_dir", lambda: hidden_dir)

        loaded = load_hidden_config()

        assert loaded == {"poc": {"workspace": "/tmp/nvflare/poc"}}
        assert config_path.read_text() == config_text

    def test_load_hidden_config_warns_about_legacy_hocon_file(self, tmp_path, monkeypatch, capsys):
        hidden_dir = tmp_path / ".nvflare"
        hidden_dir.mkdir()
        legacy_path = hidden_dir / "config.conf"
        legacy_path.write_text('poc { workspace = "/tmp/nvflare/poc" }')

        monkeypatch.setattr("nvflare.utils.cli_utils.get_or_create_hidden_nvflare_dir", lambda: hidden_dir)

        assert load_hidden_config() is None
        assert f"{legacy_path} is no longer read" in capsys.readouterr().err
        assert legacy_path.exists()
        assert not (hidden_dir / "config.json").exists()

    @pytest.mark.parametrize("config_text", ["[1, 2]", '"text"', "{not json"])
    def test_load_hidden_config_rejects_non_object_or_malformed_json(self, tmp_path, monkeypatch, config_text):
        hidden_dir = tmp_path / ".nvflare"
        hidden_dir.mkdir()
        (hidden_dir / "config.json").write_text(config_text)

        monkeypatch.setattr("nvflare.utils.cli_utils.get_or_create_hidden_nvflare_dir", lambda: hidden_dir)

        with pytest.raises(ValueError, match="cannot parse .*config.json"):
            load_hidden_config()

    def test_save_config_replaces_file_atomically_without_temp_leak(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text('{"version": 1}\n')
        config = {"version": 2, "poc": {"workspace": "/tmp/nvflare/poc"}}

        save_config(config, str(config_path))

        assert json.loads(config_path.read_text()) == config
        assert list(tmp_path.glob("tmp*")) == []

    def test_save_config_uses_final_filename_extension(self, tmp_path):
        config_path = tmp_path / "client.backup.json"

        save_config({"version": 2}, str(config_path))

        assert json.loads(config_path.read_text())["version"] == 2

    def test_save_config_supports_compound_default_extension(self, tmp_path):
        config_path = tmp_path / "client.json.default"

        save_config({"version": 2}, str(config_path))

        assert json.loads(config_path.read_text())["version"] == 2

    @pytest.mark.parametrize(
        "name",
        [
            "client.json.backup",
            "config_fed_client.conf",
            "config_fed_client.yml",
            "config_fed_client.yaml",
            "config_fed_client.conf.default",
            "config_fed_client.yml.default",
            "config_fed_client.yaml.default",
        ],
    )
    def test_save_config_rejects_non_json_extension(self, tmp_path, name):
        config_path = tmp_path / name

        with pytest.raises(ValueError, match="invalid configuration file path"):
            save_config({"version": 2}, str(config_path))

        assert not config_path.exists()

    @pytest.mark.parametrize(
        "inputs, result", [(([], "a"), ["a"]), ((["a"], "a"), ["a"]), ((["a", "b"], "b"), ["a", "b"])]
    )
    def test_append_if_not_in_list(self, inputs, result):
        arr, item = inputs
        assert result == append_if_not_in_list(arr, item)
