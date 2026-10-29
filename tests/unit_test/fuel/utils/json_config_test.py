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

import pytest

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.json_config_loader import JsonConfigLoader


@pytest.mark.parametrize(
    "text, expected",
    [
        ('// note\n{"x": "${PRIVATE_VALUE}"} /* note */', {"x": "${PRIVATE_VALUE}"}),
        ('{"x": [1,],} // final comment', {"x": [1]}),
        ("[1, /* trailing */ 2, // trailing\r\n]", [1, 2]),
        ('[",]", ",}", "/* literal */",]', [",]", ",}", "/* literal */"]),
        (
            '{"url": "https://example.test/*literal*/", "quote": "\\"//literal"}',
            {"url": "https://example.test/*literal*/", "quote": '"//literal'},
        ),
    ],
)
def test_comments_do_not_change_string_values(text, expected):
    assert JsonConfigLoader().load_config_from_str(text).to_dict() == expected


@pytest.mark.parametrize(
    "text",
    [
        "{/* unfinished",
        'include "private.conf"',
        '{"x": "unfinished}',
        "[1// c\n2]",
        "[1/* c */2]",
        "[,]",
        "{,}",
        "[1,,]",
    ],
)
def test_invalid_json_is_rejected(text):
    with pytest.raises(ValueError):
        JsonConfigLoader().load_config_from_str(text)


def test_comments_preserve_error_locations():
    text = '{\n/* first line\nsecond line */\n"x": invalid\n}'
    with pytest.raises(json.JSONDecodeError) as error:
        JsonConfigLoader().load_config_from_str(text)
    assert error.value.pos == text.index("invalid")
    assert error.value.lineno == 4
    assert error.value.colno == 6


@pytest.mark.parametrize("text", ['{"x": 1, "x": 2}', '{"outer": {"x": {}, "x": {"y": 1}}}'])
def test_duplicate_keys_are_rejected(text):
    with pytest.raises(ValueError, match="Duplicate configuration key: x"):
        JsonConfigLoader().load_config_from_str(text)


class TestJsonConfig:
    def return_dict(self, file_name):
        if file_name == "test.json":
            return {
                "a": {
                    "a1": 1,
                    "a2": 2,
                },
                "b": 1,
                "c": "hi",
                "d": [1, 2],
            }
        else:  # default
            return {
                "a": {
                    "a1": 2,
                    "a2": 4,
                },
                "b": 2,
                "c": "hello",
                "d": [2, 4],
            }

    def test_json_loader(self):
        loader = JsonConfigLoader()
        assert loader.get_format() == ConfigFormat.JSON
        loader._from_file = self.return_dict
        dicts = {
            "a": {
                "a1": 200,
            },
            "c": "hello",
            "d": [200, 400, 500],
            "e1": "Yes",
            "e2": "True",
            "e3": "NO",
        }

        config = loader.load_config("test.json")
        assert config is not None
        conf = config.get_native_conf()
        assert conf["a"]["a1"] == 1
        assert conf.get("b") == 1
        assert conf.get("c") == "hi"
        assert conf.get("d") == [1, 2]
        assert conf.get("e4", None) is None
        assert config.get_format() == ConfigFormat.JSON

        assert conf.get("d") == [1, 2]
        assert conf.get("a") == {"a1": 1, "a2": 2}

        config = loader.load_config_from_dict(dicts)
        assert config.get_format() == ConfigFormat.JSON
        conf = config.get_native_conf()

        assert config is not None
        assert conf == dicts
        assert config.get_format() == ConfigFormat.JSON

        config = loader.load_config_from_str(json.dumps(dicts))
        assert config is not None
        assert config.get_native_conf() == dicts
        assert config.get_format() == ConfigFormat.JSON

    def test_load_json_cofig_from_dict(self):
        loader = JsonConfigLoader()
        assert loader.get_format() == ConfigFormat.JSON
        dicts = {
            "a": {
                "a1": 200,
            },
            "c": "hello",
            "d": [200, 400, 500],
            "e1": "Yes",
            "e2": "True",
            "e3": "NO",
        }

        config = loader.load_config_from_dict(dicts)
        assert config.get_format() == ConfigFormat.JSON
        conf = config.get_native_conf()

        assert config is not None
        assert conf == dicts
        assert config.get_format() == ConfigFormat.JSON

    def test_load_json_config_from_str(self):
        loader = JsonConfigLoader()
        assert loader.get_format() == ConfigFormat.JSON
        dicts = {
            "a": {
                "a1": 200,
            },
            "c": "hello",
            "d": [200, 400, 500],
            "e1": "Yes",
            "e2": "True",
            "e3": "NO",
        }
        config = loader.load_config_from_str(json.dumps(dicts))
        assert config is not None
        assert config.get_native_conf() == dicts
        assert config.get_format() == ConfigFormat.JSON
