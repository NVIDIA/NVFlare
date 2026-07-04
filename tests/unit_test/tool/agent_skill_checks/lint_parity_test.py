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

"""Parity tests for the hand-mirrored tables in the skill lint engine.

These tables (`V1_LINT_IDS`, the `LINT_SKILL_*` constants, and the
`_KNOWN_AGENT_*` command/flag tables in the command-drift lint) are maintained
by hand and can silently drift from each other and from the real ``nvflare
agent`` CLI. Nothing referenced them before, so a mismatch could ship
unnoticed. These tests pin the internal parity and the command-name parity
against the actual argparse parser.

Exact per-command flag parity against the full CLI is intentionally out of scope
here: several flags (e.g. ``--format``) are added by the global CLI wrapper
rather than ``def_agent_cli_parser``, so asserting exact flag sets would couple
this test to global-arg wiring. Command names and table self-consistency are the
drift-prone parts that matter for the drift lint.
"""

import argparse
import sys
from pathlib import Path

CHECKS_PARENT = Path(__file__).resolve().parents[4] / "dev_tools" / "agent" / "skills"
sys.path.insert(0, str(CHECKS_PARENT))

from checks import lints as lints_module  # noqa: E402


def test_v1_lint_ids_match_lint_skill_constants():
    # Every LINT_SKILL_* constant must be registered in V1_LINT_IDS and vice
    # versa, so adding a lint (or its constant) can't silently skip registration.
    constant_values = {
        value for name, value in vars(lints_module).items() if name.startswith("LINT_SKILL_") and isinstance(value, str)
    }
    assert constant_values == set(lints_module.V1_LINT_IDS)
    # V1_LINT_IDS has no duplicates.
    assert len(lints_module.V1_LINT_IDS) == len(set(lints_module.V1_LINT_IDS))


def _agent_parser_tree():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="sub_cmd")
    from nvflare.tool.agent.agent_cli import def_agent_cli_parser

    def_agent_cli_parser(sub)
    return sub.choices["agent"]


def _subparser_choices(parser):
    if parser._subparsers is None:
        return {}
    for action in parser._subparsers._group_actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def test_known_agent_command_tables_match_real_cli():
    # Command names in the drift-lint tables must match the actual agent CLI, so
    # the lint neither rejects a real command nor waves through a removed one.
    agent = _agent_parser_tree()
    agent_subs = _subparser_choices(agent)
    assert set(agent_subs) == lints_module._KNOWN_AGENT_COMMANDS

    skills_subs = _subparser_choices(agent_subs["skills"])
    assert set(skills_subs) == lints_module._KNOWN_AGENT_SKILLS_COMMANDS

    assert "agent" in lints_module._KNOWN_NVFLARE_ROOT_COMMANDS


def test_known_agent_flag_table_keys_are_valid_command_paths():
    # Every _KNOWN_AGENT_FLAGS key must be a real command path built from the
    # command tables ("agent", "agent <cmd>", or "agent skills <cmd>"), and each
    # command must at least allow --schema (the universal agent flag).
    valid_keys = {"agent"}
    for command in lints_module._KNOWN_AGENT_COMMANDS:
        valid_keys.add(f"agent {command}")
    for command in lints_module._KNOWN_AGENT_SKILLS_COMMANDS:
        valid_keys.add(f"agent skills {command}")

    for key, flags in lints_module._KNOWN_AGENT_FLAGS.items():
        assert key in valid_keys, f"_KNOWN_AGENT_FLAGS key '{key}' is not a known agent command path"
        assert "--schema" in flags, f"command '{key}' should allow --schema"


def test_known_agent_flags_match_real_cli_options():
    agent = _agent_parser_tree()
    agent_subs = _subparser_choices(agent)
    skills_subs = _subparser_choices(agent_subs["skills"])
    parsers = {
        "agent": agent,
        **{f"agent {name}": parser for name, parser in agent_subs.items()},
        **{f"agent skills {name}": parser for name, parser in skills_subs.items()},
    }

    for command_path, expected in lints_module._KNOWN_AGENT_FLAGS.items():
        parser = parsers[command_path]
        actual = {
            option
            for action in parser._actions
            if not isinstance(action, (argparse._HelpAction, argparse._SubParsersAction))
            for option in action.option_strings
        }
        # --format is a top-level option normalized so it is accepted after any
        # subcommand; it is intentionally not repeated on each nested parser.
        actual.add("--format")
        assert expected == actual, f"flag registry drift for '{command_path}'"


def test_runtime_boundary_excluded_dirs_match_packaging_exclusions():
    # The runtime-boundary lint's excluded dirs and the packaging exclusion set
    # are hand-mirrored: both name the directories that are stripped from a
    # shipped skill (evals/, __pycache__). Keep them in sync so the lint never
    # scans content packaging removes, and packaging never ships content the lint
    # skips. Packaging also lists byte-code file globs (*.pyc/*.pyo), which are
    # not directory names and are excluded from this comparison.
    from nvflare.tool.agent.skill_manifest import SKILL_PACKAGING_EXCLUDE_NAMES

    packaging_dir_names = {name for name in SKILL_PACKAGING_EXCLUDE_NAMES if not name.startswith("*")}
    assert lints_module._RUNTIME_BOUNDARY_EXCLUDED_DIRS == packaging_dir_names
