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


def test_global_args_after_subcommand(monkeypatch):
    from nvflare import cli as cli_mod

    monkeypatch.setattr(
        cli_mod.sys,
        "argv",
        ["nvflare", "system", "version", "--format", "json", "--connect-timeout", "7"],
    )

    _, args, _ = cli_mod.parse_args("nvflare")
    assert args.sub_command == "system"
    assert args.system_sub_cmd == "version"
    assert args.format == "json"
    assert args.connect_timeout == 7.0


def test_jsonl_global_format_allowed_for_job_monitor(monkeypatch):
    from nvflare import cli as cli_mod

    monkeypatch.setattr(
        cli_mod.sys,
        "argv",
        ["nvflare", "job", "monitor", "abc123", "--format", "jsonl"],
    )

    _, args, _ = cli_mod.parse_args("nvflare")
    assert args.sub_command == "job"
    assert args.job_sub_cmd == "monitor"
    assert args.format == "jsonl"


def test_cert_init_deploy_version_is_command_option(monkeypatch, tmp_path):
    from nvflare import cli as cli_mod

    profile_path = tmp_path / "project_profile.yaml"
    profile_path.write_text("name: test_project\n", encoding="utf-8")

    monkeypatch.setattr(
        cli_mod.sys,
        "argv",
        [
            "nvflare",
            "cert",
            "init",
            "--profile",
            str(profile_path),
            "-o",
            str(tmp_path / "ca"),
            "--deploy-version",
            "01",
        ],
    )

    _, args, _ = cli_mod.parse_args("nvflare")
    assert args.sub_command == "cert"
    assert args.cert_sub_command == "init"
    assert args.deploy_version == "01"
