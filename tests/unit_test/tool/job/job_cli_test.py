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
from pyhocon import ConfigFactory as CF

from nvflare.tool.job import job_cli
from nvflare.tool.job.job_cli import convert_args_list_to_dict


class TestJobCLI:
    @pytest.mark.parametrize("value, expected", [("0", 0), ("12", 12)])
    def test_non_negative_int(self, value, expected):
        assert job_cli._non_negative_int(value) == expected

    @pytest.mark.parametrize("value", ["-1", "not-an-int"])
    def test_non_negative_int_rejects_invalid_values(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            job_cli._non_negative_int(value)

    @pytest.mark.parametrize(
        "filename, expected",
        [("config_fed_server.conf", "config_fed_server"), ("README", "README"), ("/tmp/a.b.json", "a.b")],
    )
    def test_find_filename_basename(self, filename, expected):
        assert job_cli.find_filename_basename(filename) == expected

    def test_build_job_template_indices_finds_configured_templates(self, tmp_path):
        template = tmp_path / "fedavg"
        template.mkdir()
        (template / "config_fed_server.conf").write_text("{}")

        config = job_cli.build_job_template_indices(str(tmp_path))

        assert "fedavg" in config.get("templates")
        assert all(config.get(f"templates.fedavg.{key}") == "NA" for key in job_cli.JOB_INFO_KEYS)

    def test_get_template_info_config_parses_optional_info_file(self, tmp_path):
        assert job_cli.get_template_info_config(str(tmp_path)) is None
        (tmp_path / job_cli.JOB_INFO_CONF).write_text('description = "demo"')

        config = job_cli.get_template_info_config(str(tmp_path))

        assert config.get("description") == "demo"

    def test_get_app_dirs_from_template_finds_server_and_client_apps(self, tmp_path):
        server_app = tmp_path / "server-app"
        client_app = tmp_path / "client-app"
        ignored = tmp_path / "ignored"
        for directory in (server_app, client_app, ignored):
            directory.mkdir()
        (server_app / job_cli.CONFIG_FED_SERVER_CONF).write_text("{}")
        (client_app / job_cli.CONFIG_FED_CLIENT_CONF).write_text("{}")

        assert set(job_cli.get_app_dirs_from_template(str(tmp_path))) == {str(server_app), str(client_app)}

    def test_get_app_dirs_from_job_folder_finds_config_and_custom_parents(self, tmp_path):
        (tmp_path / "app1" / "config").mkdir(parents=True)
        (tmp_path / "app2" / "custom").mkdir(parents=True)
        (tmp_path / "unrelated").mkdir()

        assert set(job_cli.get_app_dirs_from_job_folder(str(tmp_path))) == {"app1", "app2"}

    def test_get_src_template_requires_directory_and_info_file(self, tmp_path):
        args = argparse.Namespace(template=str(tmp_path))
        assert job_cli.get_src_template(args) is None
        (tmp_path / job_cli.JOB_INFO_CONF).write_text("{}")
        assert job_cli.get_src_template(args) == str(tmp_path)

    def test_remove_pycache_and_extra_template_files(self, tmp_path):
        custom = tmp_path / "custom"
        pycache = custom / "pkg" / "__pycache__"
        pyc_dir = custom / "generated.pyc"
        pycache.mkdir(parents=True)
        pyc_dir.mkdir(parents=True)
        (pycache / "module.pyc").write_bytes(b"compiled")

        job_cli.remove_pycache_files(str(custom))

        assert not pycache.exists()
        assert not pyc_dir.exists()

        config = tmp_path / "config"
        config.mkdir()
        for filename in (job_cli.JOB_INFO_MD, job_cli.JOB_INFO_CONF, "__init__.py"):
            (config / filename).write_text("remove")
        (config / "__pycache__").mkdir()
        (config / "keep.conf").write_text("keep")

        job_cli.remove_extra_files(str(config))

        assert sorted(path.name for path in config.iterdir()) == ["keep.conf"]

    def test_check_template_exists(self):
        config = CF.parse_string('{templates = {fedavg = {description = "demo"}}}')

        job_cli.check_template_exists("fedavg", config)
        with pytest.raises(ValueError, match="Invalid template name missing"):
            job_cli.check_template_exists("missing", config)

    @pytest.mark.parametrize(
        "value, width, expected",
        [("abc", 5, "abc  "), ("abcdef", 3, "abc")],
    )
    def test_fix_length_format(self, value, width, expected):
        assert job_cli.fix_length_format(value, width) == expected

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
