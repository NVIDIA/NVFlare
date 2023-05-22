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
import re
import shlex
from typing import List


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
