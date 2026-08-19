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
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

from nvflare import cli

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEPLOYMENT_TUTORIAL = _REPO_ROOT / (
    "examples/tutorials/self-paced-training/part-2_federated_learning_system/"
    "chapter-3_federated_computing_platform/03.2_deployment_simulation/simulate_real_world_deployment.ipynb"
)
_SETUP_TUTORIAL = _REPO_ROOT / "examples/tutorials/setup_poc.ipynb"


_SHELL_OPERATORS = {"|", "||", "&&", ";", "&"}


def _nvflare_commands(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell_index, cell in enumerate(notebook["cells"]):
        in_fence = False
        for line_number, line in enumerate("".join(cell.get("source", [])).splitlines(), start=1):
            stripped = line.strip()
            if cell["cell_type"] == "markdown":
                if stripped.startswith("```"):
                    in_fence = not in_fence
                    continue
                if not in_fence:
                    continue
            elif cell["cell_type"] != "code" or not stripped.startswith("!"):
                continue

            command = stripped.removeprefix("!").strip()
            if "nvflare" not in command:
                continue
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            try:
                tokens = list(lexer)
            except ValueError as e:
                raise AssertionError(f"{path}: cell {cell_index}, line {line_number}: {e}") from e

            for index, token in enumerate(tokens):
                if token != "nvflare":
                    continue
                if index > 0 and tokens[index - 1] not in _SHELL_OPERATORS:
                    continue
                end = next((i for i in range(index + 1, len(tokens)) if tokens[i] in _SHELL_OPERATORS), len(tokens))
                location = f"{path}: cell {cell_index}, line {line_number}"
                yield location, tokens[index:end]


def test_tutorial_nvflare_commands_parse_with_real_cli():
    for path in (_DEPLOYMENT_TUTORIAL, _SETUP_TUTORIAL):
        commands = list(_nvflare_commands(path))
        assert commands, path
        for location, command in commands:
            is_usage_example = any("[" in token or token in {"-h", "--help"} for token in command)
            if is_usage_example:
                continue
            with patch.object(sys, "argv", command):
                try:
                    cli.parse_args("nvflare")
                except SystemExit as e:
                    raise AssertionError(f"{location}: {' '.join(command)} exited with code {e.code}") from e
