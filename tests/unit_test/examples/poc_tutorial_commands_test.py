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
import json
import re
import shlex
from pathlib import Path

from nvflare.cli import def_config_parser
from nvflare.tool.poc.poc_commands import def_poc_parser
from nvflare.tool.system.system_cli import def_system_cli_parser

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEPLOYMENT_TUTORIAL = _REPO_ROOT / (
    "examples/tutorials/self-paced-training/part-2_federated_learning_system/"
    "chapter-3_federated_computing_platform/03.2_deployment_simulation/simulate_real_world_deployment.ipynb"
)
_SETUP_TUTORIAL = _REPO_ROOT / "examples/tutorials/setup_poc.ipynb"


def _notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _nvflare_commands(path: Path):
    for line in _notebook_text(path).splitlines():
        command = line.strip().removeprefix("!").strip()
        if command.startswith("nvflare "):
            yield shlex.split(command)


def _tutorial_parser():
    parser = argparse.ArgumentParser(prog="nvflare")
    subcommands = parser.add_subparsers(dest="subcommand")
    def_poc_parser(subcommands)
    def_config_parser(subcommands)
    system_parser = subcommands.add_parser("system")
    def_system_cli_parser(system_parser)
    return parser


def test_poc_tutorial_commands_parse():
    parser = _tutorial_parser()

    for path in (_DEPLOYMENT_TUTORIAL, _SETUP_TUTORIAL):
        for command in _nvflare_commands(path):
            is_usage_example = any("[" in token or token in {"-h", "--help"} for token in command)
            if is_usage_example:
                continue
            parser.parse_args(command[1:])


def test_tutorials_do_not_pipe_into_nvflare_commands():
    for path in (_DEPLOYMENT_TUTORIAL, _SETUP_TUTORIAL):
        assert re.search(r"\|\s*nvflare\b", _notebook_text(path)) is None


def test_deployment_tutorial_commands_use_current_docker_participants():
    commands = list(_nvflare_commands(_DEPLOYMENT_TUTORIAL))

    assert ["nvflare", "poc", "start", "-ex", "site-2"] in commands
    for command in commands:
        assert not {"site-3", "site-4", "site-5"}.intersection(command), command


def test_poc_tutorial_commands_do_not_repeat_scalar_service_options():
    for path in (_DEPLOYMENT_TUTORIAL, _SETUP_TUTORIAL):
        for command in _nvflare_commands(path):
            assert command.count("-p") + command.count("--service") <= 1, command
            assert command.count("-ex") + command.count("--exclude") <= 1, command
