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

from nvflare.fuel.hci.cmd_arg_utils import parse_command_line, split_to_args


class TestCmdArgUtils:
    def test_split_to_args_strips_single_quoted_submit_job_path(self):
        args = split_to_args("submit_job '/tmp/nvflare/job_configs/cifar10_splitnn'")
        assert args == ["submit_job", "/tmp/nvflare/job_configs/cifar10_splitnn"]

    def test_split_to_args_falls_back_for_unmatched_single_quote(self):
        args = split_to_args("submit_job /tmp/nvflare/o'connor_job")
        assert args == ["submit_job", "/tmp/nvflare/o'connor_job"]

    def test_parse_command_line_supports_single_quoted_path_with_props(self):
        line, args, props = parse_command_line("submit_job '/tmp/nvflare/my job' #format=csv")
        assert line == "submit_job '/tmp/nvflare/my job'"
        assert args == ["submit_job", "/tmp/nvflare/my job"]
        assert props == "format=csv"

    def test_parse_command_line_keeps_unquoted_props_behavior(self):
        line, args, props = parse_command_line("list_job #format=csv")
        assert line == "list_job"
        assert args == ["list_job"]
        assert props == "format=csv"
