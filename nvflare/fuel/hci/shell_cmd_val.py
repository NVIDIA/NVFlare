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

from typing import List

from nvflare.fuel.hci.cmd_arg_utils import ArgValidator


class ShellCommandValidator(object):
    def __init__(self, arg_validator: ArgValidator):
        """Base class for validators to be called by command executors for shell commands.

        Args:
            arg_validator: instance of ArgValidator
        """
        self.arg_validator = arg_validator

    def validate(self, args: List[str]):
        self.arg_validator.err = ""
        return self.arg_validator.validate(args)

    def get_usage(self):
        return self.arg_validator.get_usage()


class TailValidator(ShellCommandValidator):
    def __init__(self):
        """Validator for the tail command."""
        val = ArgValidator("tail")
        val.add_argument("-c", type=int, help="output the last C bytes")
        val.add_argument("-n", type=int, help="output the last N lines")
        val.add_argument("files", metavar="file", type=str, nargs="+")
        ShellCommandValidator.__init__(self, val)


class HeadValidator(ShellCommandValidator):
    def __init__(self):
        """Validator for the head command."""
        val = ArgValidator("head")
        val.add_argument("-c", type=int, help="print the first C bytes of each file")
        val.add_argument("-n", type=int, help="print the first N lines instead of the first 10")
        val.add_argument("files", metavar="file", type=str, nargs="+")
        ShellCommandValidator.__init__(self, val)


class GrepValidator(ShellCommandValidator):
    def __init__(self):
        """Validator for the grep command."""
        val = ArgValidator("grep")
        val.add_argument("-n", action="store_true", help="print line number with output lines")
        val.add_argument("-i", action="store_true", help="ignore case distinctions")
        val.add_argument("-b", action="store_true", help="print the byte offset with output lines")
        val.add_argument("pattern", metavar="pattern", type=str)
        val.add_argument("files", metavar="file", type=str, nargs="+")
        ShellCommandValidator.__init__(self, val)


class CatValidator(ShellCommandValidator):
    def __init__(self):
        """Validator for the cat command."""
        val = ArgValidator("cat")
        val.add_argument("-n", action="store_true", help="number all output lines")
        val.add_argument("-b", action="store_true", help="number nonempty output lines, overrides -n")
        val.add_argument("-s", action="store_true", help="suppress repeated empty output lines")
        val.add_argument("-T", action="store_true", help="display TAB characters as ^I")
        val.add_argument("files", metavar="file", type=str, nargs="+")
        ShellCommandValidator.__init__(self, val)


class LsValidator(ShellCommandValidator):
    def __init__(self):
        """Validator for the ls command."""
        val = ArgValidator("ls")
        val.add_argument("-a", action="store_true")
        val.add_argument("-l", action="store_true", help="use a long listing format")
        val.add_argument("-t", action="store_true", help="sort by modification time, newest first")
        val.add_argument("-S", action="store_true", help="sort by file size, largest first")
        val.add_argument("-R", action="store_true", help="list subdirectories recursively")
        val.add_argument("-u", action="store_true", help="with -l: show access time, otherwise: sort by access time")
        val.add_argument("files", metavar="file", type=str, nargs="?")
        ShellCommandValidator.__init__(self, val)
