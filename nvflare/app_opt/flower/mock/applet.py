# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.app_common.tie.cli_applet import CLIApplet, CommandDescriptor
from nvflare.app_opt.flower.defs import Constant


class MockClientApplet(CLIApplet):
    def __init__(self):
        CLIApplet.__init__(self)

    def get_command(self, ctx: dict) -> CommandDescriptor:
        main_module = "nvflare.app_opt.flower.mock.flower_client"
        addr = ctx.get(Constant.APP_CTX_SERVER_ADDR)
        num_rounds = ctx.get(Constant.APP_CTX_NUM_ROUNDS)
        client_name = ctx.get(Constant.APP_CTX_CLIENT_NAME)

        return CommandDescriptor(
            cmd=f"python -m {main_module} -a {addr} -n {num_rounds} -c {client_name}",
            log_file_name="flower_client_log.txt",
            stdout_msg_prefix="FLWR-CA",
        )


class MockServerApplet(CLIApplet):
    def __init__(self):
        CLIApplet.__init__(self)

    def get_command(self, ctx: dict) -> CommandDescriptor:
        main_module = "nvflare.app_opt.flower.mock.flower_server"
        addr = ctx.get(Constant.APP_CTX_SERVER_ADDR)
        num_rounds = ctx.get(Constant.APP_CTX_NUM_ROUNDS)

        return CommandDescriptor(
            cmd=f"python -m {main_module} -a {addr} -n {num_rounds}",
            log_file_name="flower_server_log.txt",
            stdout_msg_prefix="FLWR-SA",
        )
