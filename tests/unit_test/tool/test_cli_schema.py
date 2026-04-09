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

from nvflare.tool.cli_schema import SCHEMA_VERSION, parser_to_schema


def _make_parser(**kwargs) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(**kwargs)


class TestParserToSchema:
    def test_schema_version_present(self):
        parser = _make_parser(description="Test parser")
        schema = parser_to_schema(parser, "test cmd")
        assert schema["schema_version"] == SCHEMA_VERSION

    def test_command_field(self):
        parser = _make_parser()
        schema = parser_to_schema(parser, "nvflare cert init")
        assert schema["command"] == "nvflare cert init"

    def test_description_field(self):
        parser = _make_parser(description="My description")
        schema = parser_to_schema(parser, "cmd")
        assert schema["description"] == "My description"

    def test_empty_description(self):
        parser = _make_parser()
        schema = parser_to_schema(parser, "cmd")
        assert schema["description"] == ""

    def test_examples_default_empty(self):
        parser = _make_parser()
        schema = parser_to_schema(parser, "cmd")
        assert schema["examples"] == []

    def test_examples_passed_through(self):
        parser = _make_parser()
        schema = parser_to_schema(parser, "cmd", examples=["nvflare cert init -n myproj -o /tmp/ca"])
        assert len(schema["examples"]) == 1
        assert "nvflare cert init" in schema["examples"][0]

    def test_help_action_excluded(self):
        parser = _make_parser()
        schema = parser_to_schema(parser, "cmd")
        names = [a["name"] for a in schema["args"]]
        assert "--help" not in names
        assert "-h" not in names

    def test_store_true_inferred_as_boolean(self):
        parser = _make_parser()
        parser.add_argument("--force", action="store_true", default=False, help="Force overwrite.")
        schema = parser_to_schema(parser, "cmd")
        force_arg = next(a for a in schema["args"] if a["name"] == "--force")
        assert force_arg["type"] == "boolean"

    def test_store_false_inferred_as_boolean(self):
        parser = _make_parser()
        parser.add_argument("--no-verify", action="store_false", dest="verify", help="Disable verify.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if "--no-verify" in a["name"])
        assert arg["type"] == "boolean"

    def test_int_type_inferred_as_integer(self):
        parser = _make_parser()
        parser.add_argument("--count", type=int, default=5, help="Count.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--count")
        assert arg["type"] == "integer"

    def test_float_type_inferred_as_number(self):
        parser = _make_parser()
        parser.add_argument("--ratio", type=float, default=0.5, help="Ratio.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--ratio")
        assert arg["type"] == "number"

    def test_output_dir_inferred_as_path(self):
        parser = _make_parser()
        parser.add_argument("-o", "--output-dir", dest="output_dir", help="Output directory.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--output-dir")
        assert arg["type"] == "path"

    def test_path_keyword_in_dest_inferred_as_path(self):
        parser = _make_parser()
        parser.add_argument("--ca-dir", dest="ca_dir", help="CA directory.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--ca-dir")
        assert arg["type"] == "path"

    def test_generic_string_arg(self):
        parser = _make_parser()
        parser.add_argument("--name", help="Name.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--name")
        assert arg["type"] == "string"

    def test_required_flag_for_required_arg(self):
        parser = _make_parser()
        parser.add_argument("--name", required=True, help="Name.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--name")
        assert arg["required"] is True

    def test_required_flag_false_for_optional_arg(self):
        parser = _make_parser()
        parser.add_argument("--org", required=False, default=None, help="Org.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--org")
        assert arg["required"] is False

    def test_positional_arg_required_true(self):
        parser = _make_parser()
        parser.add_argument("project_name", help="Project name.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "project_name")
        assert arg["required"] is True

    def test_choices_included(self):
        parser = _make_parser()
        parser.add_argument("--type", choices=["client", "server", "org_admin"], help="Type.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--type")
        assert "choices" in arg
        assert set(arg["choices"]) == {"client", "server", "org_admin"}

    def test_no_choices_key_when_no_choices(self):
        parser = _make_parser()
        parser.add_argument("--name", help="Name.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--name")
        assert "choices" not in arg

    def test_default_included(self):
        parser = _make_parser()
        parser.add_argument("--retries", type=int, default=3, help="Retries.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--retries")
        assert arg["default"] == 3

    def test_default_none_when_not_set(self):
        parser = _make_parser()
        parser.add_argument("--name", help="Name.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--name")
        assert arg["default"] is None

    def test_aliases_included_for_short_and_long(self):
        parser = _make_parser()
        parser.add_argument("-n", "--name", help="Name.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--name")
        assert "aliases" in arg
        assert "-n" in arg["aliases"]

    def test_no_aliases_for_long_only_arg(self):
        parser = _make_parser()
        parser.add_argument("--org", help="Org.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--org")
        assert "aliases" not in arg

    def test_help_text_in_description_field(self):
        parser = _make_parser()
        parser.add_argument("--name", help="The participant name.")
        schema = parser_to_schema(parser, "cmd")
        arg = next(a for a in schema["args"] if a["name"] == "--name")
        assert arg["description"] == "The participant name."

    def test_args_list_is_list(self):
        parser = _make_parser()
        parser.add_argument("--foo", help="Foo.")
        schema = parser_to_schema(parser, "cmd")
        assert isinstance(schema["args"], list)

    def test_multiple_args_all_present(self):
        parser = _make_parser()
        parser.add_argument("-n", "--name", required=True, help="Name.")
        parser.add_argument("-o", "--output-dir", required=True, dest="output_dir", help="Output dir.")
        parser.add_argument("--force", action="store_true", default=False, help="Force.")
        schema = parser_to_schema(parser, "cmd")
        names = [a["name"] for a in schema["args"]]
        assert "--name" in names
        assert "--output-dir" in names
        assert "--force" in names
