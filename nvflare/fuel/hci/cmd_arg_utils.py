# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import io
import os
import re
import shlex
from typing import List

from nvflare.apis.utils.format_check import type_pattern_mapping


def split_to_args(line: str) -> List[str]:
    if '"' in line:
        return shlex.split(line)
    else:
        line = re.sub(" +", " ", line)
        return line.split(" ")


def join_args(segs: List[str]) -> str:
    result = ""
    sep = ""
    for a in segs:
        parts = a.split()
        if len(parts) < 2:
            p = parts[0]
        else:
            p = '"' + a + '"'
        result = result + sep + p
        sep = " "

    return result


class ArgValidator(argparse.ArgumentParser):
    def __init__(self, name):
        """Validator for admin shell commands that uses argparse to check arguments and get usage through print_help.

        Args:
            name: name of the program to pass to ArgumentParser
        """
        argparse.ArgumentParser.__init__(self, prog=name, add_help=False)
        self.err = ""

    def error(self, message):
        self.err = message

    def validate(self, args):
        try:
            result = self.parse_args(args)
            return self.err, result
        except Exception:
            return 'argument error; try "? cmdName to show supported usage for a command"', None

    def get_usage(self) -> str:
        buffer = io.StringIO()
        self.print_help(buffer)
        usage_output = buffer.getvalue().split("\n", 1)[1]
        buffer.close()
        return usage_output


def process_targets_into_str(targets: List[str]) -> str:
    if not isinstance(targets, list):
        raise SyntaxError("targets is not a list.")
    if not all(isinstance(t, str) for t in targets):
        raise SyntaxError("all targets in the list of targets must be strings.")
    for t in targets:
        try:
            validate_required_target_string(t)
        except SyntaxError:
            raise SyntaxError(f"invalid target {t}")
    return " ".join(targets)


def validate_required_target_string(target: str) -> str:
    """Returns the target string if it exists and is valid."""
    if not target:
        raise SyntaxError("target is required but not specified.")
    if not isinstance(target, str):
        raise SyntaxError("target is not str.")
    if not re.match("^[A-Za-z0-9._-]*$", target):
        raise SyntaxError("target must be a string of only valid characters and no spaces.")
    return target


def validate_options_string(options: str) -> str:
    """Returns the options string if it is valid."""
    if not isinstance(options, str):
        raise SyntaxError("options is not str.")
    if not re.match("^[A-Za-z0-9- ]*$", options):
        raise SyntaxError("options must be a string of only valid characters.")
    return options


def validate_path_string(path: str) -> str:
    """Returns the path string if it is valid."""
    if not isinstance(path, str):
        raise SyntaxError("path is not str.")
    if not re.match("^[A-Za-z0-9-._/]*$", path):
        raise SyntaxError("unsupported characters in path {}".format(path))
    if os.path.isabs(path):
        raise SyntaxError("absolute path is not allowed")
    paths = path.split(os.path.sep)
    for p in paths:
        if p == "..":
            raise SyntaxError(".. in path name is not allowed")
    return path


def validate_file_string(file: str) -> str:
    """Returns the file string if it is valid."""
    validate_path_string(file)
    basename, file_extension = os.path.splitext(file)
    if file_extension not in [".txt", ".log", ".json", ".csv", ".sh", ".config", ".py"]:
        raise SyntaxError(
            "this command cannot be applied to file {}. Only files with the following extensions are "
            "permitted: .txt, .log, .json, .csv, .sh, .config, .py".format(file)
        )
    return file


def validate_sp_string(sp_string) -> str:
    if re.match(
        type_pattern_mapping.get("sp_end_point"),
        sp_string,
    ):
        return sp_string
    else:
        raise SyntaxError("sp_string must be of the format example.com:8002:8003")
