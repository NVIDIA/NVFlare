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
        ["nvflare", "system", "version", "--out-format", "json", "--connect-timeout", "7"],
    )

    _, args, _ = cli_mod.parse_args("nvflare")
    assert args.sub_command == "system"
    assert args.system_sub_cmd == "version"
    assert args.out_format == "json"
    assert args.connect_timeout == 7.0
