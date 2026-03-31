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

from types import SimpleNamespace

from nvflare.fuel.hci.client.api import AdminAPI
from nvflare.fuel.hci.client.api_spec import CommandContext


def test_do_client_command_preserves_command_props():
    api = AdminAPI.__new__(AdminAPI)
    captured = {}

    def _new_command_context(command, args, ent):
        ctx = CommandContext()
        ctx.set_command(command)
        ctx.set_command_args(args)
        ctx.set_command_entry(ent)
        return ctx

    def _handler(args, ctx):
        captured["props"] = ctx.get_command_props()
        ctx.set_command_result({"status": "ok"})

    api._new_command_context = _new_command_context
    ent = SimpleNamespace(handler=_handler)

    result = api._do_client_command("submit_job hello", ["submit_job", "hello"], ent, props={"study": "study-a"})

    assert result == {"status": "ok"}
    assert captured["props"] == {"study": "study-a"}
