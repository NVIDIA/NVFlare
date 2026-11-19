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

import pytest

from nvflare.fuel.hci.cmd_arg_utils import (
    parse_command_line,
    split_to_args,
    validate_file_string,
    validate_text_file_name,
)


class TestCmdArgUtils:
    def test_split_to_args_strips_single_quoted_submit_job_path(self):
        args = split_to_args("submit_job '/tmp/nvflare/job_configs/cifar10_splitnn'")
        assert args == ["submit_job", "/tmp/nvflare/job_configs/cifar10_splitnn"]

    def test_split_to_args_preserves_backslashes_for_unquoted_input(self):
        args = split_to_args(r"submit_job C:\tmp\job")
        assert args == ["submit_job", r"C:\tmp\job"]

    def test_split_to_args_falls_back_for_unmatched_single_quote(self):
        args = split_to_args("submit_job /tmp/nvflare/o'connor_job")
        assert args == ["submit_job", "/tmp/nvflare/o'connor_job"]

    def test_parse_command_line_supports_single_quoted_hash_path(self):
        line, args, props = parse_command_line("submit_job '/tmp/#1_job'")
        assert line == "submit_job '/tmp/#1_job'"
        assert args == ["submit_job", "/tmp/#1_job"]
        assert props is None

    def test_parse_command_line_keeps_double_quoted_behavior(self):
        line, args, props = parse_command_line('submit_job "/tmp/my job"')
        assert line == 'submit_job "/tmp/my job"'
        assert args == ["submit_job", "/tmp/my job"]
        assert props is None

    def test_parse_command_line_keeps_unquoted_props_behavior(self):
        line, args, props = parse_command_line("list_job #test_prop=value")
        assert line == "list_job"
        assert args == ["list_job"]
        assert props == "test_prop=value"

    @pytest.mark.parametrize(
        "file_name",
        [
            "audit.jsonl",
            "job-1/audit.jsonl",
            "audit.jsonl.1",
            "log.txt",
            "server.log",
            "config.json",
            "metrics.csv",
            "startup.sh",
            "site.config",
            "job.py",
        ],
    )
    def test_text_file_names_include_jsonl(self, file_name):
        assert validate_text_file_name(file_name) == ""
        assert validate_file_string(file_name) == file_name

    @pytest.mark.parametrize("file_name", ["audit.bin", "audit.jsonl.exe"])
    def test_non_text_file_names_remain_rejected(self, file_name):
        error = validate_text_file_name(file_name)
        assert "Only files with the following extensions" in error
        assert ".jsonl," in error
        with pytest.raises(SyntaxError):
            validate_file_string(file_name)

    @pytest.mark.parametrize("file_name", ["/audit.jsonl", "../audit.jsonl", "job/../audit.jsonl", "audit.jsonl;pwd"])
    def test_jsonl_extension_does_not_bypass_path_checks(self, file_name):
        with pytest.raises(SyntaxError):
            validate_file_string(file_name)
