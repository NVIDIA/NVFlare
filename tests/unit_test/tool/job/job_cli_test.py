# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import sys

import pytest

from nvflare.tool.job import job_cli
from nvflare.tool.job.job_cli import convert_args_list_to_dict


class TestJobCLI:
    @pytest.mark.parametrize("inputs, result", [(["a=1", "b=2", "c = 3"], dict(a="1", b="2", c="3"))])
    def test_convert_args_list_to_dict(self, inputs, result):
        r = convert_args_list_to_dict(inputs)
        assert r == result

    @pytest.mark.parametrize(
        "directory, path, expected",
        [("/home/user/project", "/home/user/project/subdir", True), (".", ".", True), ("./code", ".", False)],
    )
    def test_is_sub_dir(self, path, directory, expected):
        print(f"{input=}, {directory=}, {expected=}")
        assert expected == job_cli.is_subdir(path, directory)

    def test_log_config_parser_accepts_alias_and_canonical_name(self):
        parser = argparse.ArgumentParser(prog="nvflare")
        subparsers = parser.add_subparsers(dest="command")
        job_cli.def_job_cli_parser(subparsers)

        args = parser.parse_args(["job", "log-config", "job-1", "DEBUG"])
        assert args.job_sub_cmd == "log-config"

        args = parser.parse_args(["job", "log", "job-1", "DEBUG"])
        assert args.job_sub_cmd == "log"

    @pytest.mark.parametrize(
        ("subcommand", "args_before_study", "args_after_study"),
        [
            ("meta", ["job-1"], []),
            ("abort", ["job-1"], ["--force"]),
            ("clone", ["job-1"], []),
            ("download", ["job-1"], []),
            ("delete", ["job-1"], ["--force"]),
            ("stats", ["job-1"], []),
            ("logs", ["job-1"], []),
            ("monitor", ["job-1"], []),
            ("wait", ["job-1"], []),
            ("log-config", ["job-1", "DEBUG"], []),
            ("log", ["job-1", "DEBUG"], []),
        ],
    )
    def test_job_id_parsers_accept_study(self, subcommand, args_before_study, args_after_study):
        parser = argparse.ArgumentParser(prog="nvflare")
        subparsers = parser.add_subparsers(dest="command")
        job_cli.def_job_cli_parser(subparsers)

        args = parser.parse_args(["job", subcommand, *args_before_study, "--study", "cancer", *args_after_study])

        assert args.study == "cancer"

    def test_job_log_schema_uses_invoked_alias_name(self, monkeypatch, capsys):
        parser = argparse.ArgumentParser(prog="nvflare")
        subparsers = parser.add_subparsers(dest="command")
        job_cli.def_job_cli_parser(subparsers)

        cmd_args = argparse.Namespace(job_id="job-1", level=None, config=None, site="all", study="default")
        monkeypatch.setattr(sys, "argv", ["nvflare", "job", "log", "job-1", "--schema"])

        with pytest.raises(SystemExit) as exc_info:
            job_cli.cmd_job_log(cmd_args)

        assert exc_info.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["command"] == "nvflare job log"

    def test_submit_parser_accepts_submit_token(self):
        parser = argparse.ArgumentParser(prog="nvflare")
        subparsers = parser.add_subparsers(dest="command")
        job_cli.def_job_cli_parser(subparsers)

        args = parser.parse_args(["job", "submit", "-j", "/tmp/job", "--submit-token", "retry.01:A_b-c"])

        assert args.submit_token == "retry.01:A_b-c"

    @pytest.mark.parametrize("token", ["", "bad token", "bad/token", "x" * 129])
    def test_submit_parser_rejects_invalid_submit_token(self, token):
        parser = argparse.ArgumentParser(prog="nvflare")
        subparsers = parser.add_subparsers(dest="command")
        job_cli.def_job_cli_parser(subparsers)

        with pytest.raises(SystemExit):
            parser.parse_args(["job", "submit", "--submit-token", token])
