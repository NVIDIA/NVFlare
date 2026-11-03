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

import pytest

from nvflare.fuel.utils.config import CONFIG_EXTENSIONS, find_config_file, load_config


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return str(path)


def test_only_json_extensions_in_priority_order():
    assert CONFIG_EXTENSIONS == (".json", ".json.default")


def test_json_preferred_over_default_in_same_dir(tmp_path):
    _write(tmp_path / "x.json.default", {"v": "default"})
    expected = _write(tmp_path / "x.json", {"v": "json"})

    assert find_config_file("x.json.default", [str(tmp_path)]) == expected
    assert load_config("x.json", [str(tmp_path)]) == {"v": "json"}


def test_nested_json_preferred_over_top_level_default(tmp_path):
    _write(tmp_path / "x.json.default", "{not json")
    expected = _write(tmp_path / "nested" / "x.json", {"v": 1})

    assert find_config_file("x.json", [str(tmp_path)]) == expected


def test_default_used_when_no_json(tmp_path):
    expected = _write(tmp_path / "x.json.default", {"v": "default"})

    assert find_config_file("x.json", [str(tmp_path)]) == expected
    assert load_config("x.json", [str(tmp_path)]) == {"v": "default"}


def test_search_dirs_searched_in_order(tmp_path):
    _write(tmp_path / "a" / "x.json.default", {"v": "a"})
    _write(tmp_path / "b" / "x.json", {"v": "b"})

    assert load_config("x.json", [str(tmp_path / "a"), str(tmp_path / "b")]) == {"v": "a"}


def test_defaults_to_parent_dir_of_file_path(tmp_path):
    expected = _write(tmp_path / "x.json.default", {"v": 1})

    assert find_config_file(str(tmp_path / "x.json")) == expected
    assert find_config_file(str(tmp_path / "x")) == expected


def test_hocon_and_yaml_files_are_ignored(tmp_path):
    _write(tmp_path / "x.conf", "v = 1")
    _write(tmp_path / "x.yml", "v: 1")

    assert find_config_file("x.json", [str(tmp_path)]) is None
    assert load_config("x.json", [str(tmp_path)]) is None


def test_malformed_config_raises(tmp_path):
    _write(tmp_path / "x.json", "{not json")

    with pytest.raises(json.JSONDecodeError):
        load_config("x.json", [str(tmp_path)])
