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

from argparse import ArgumentParser, Namespace

import pytest


def test_recipe_missing_subcommand_prints_help_then_error(capsys):
    from nvflare.tool.recipe.recipe_cli import def_recipe_parser, handle_recipe_cmd

    parser = ArgumentParser(prog="nvflare")
    subparsers = parser.add_subparsers(dest="sub_command")
    def_recipe_parser(subparsers)

    with pytest.raises(SystemExit) as exc_info:
        handle_recipe_cmd(Namespace(recipe_sub_cmd=None))
    assert exc_info.value.code == 4

    captured = capsys.readouterr()
    assert "usage: nvflare recipe" in captured.err
    assert "\n\nInvalid arguments. — recipe subcommand required\n" in captured.err
    assert "Hint: Run with -h for usage." in captured.err
    assert "Code: INVALID_ARGS (exit 4)" in captured.err
