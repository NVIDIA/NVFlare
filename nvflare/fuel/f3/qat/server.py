# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.fuel.f3.cellnet import FQCN
from .cell_runner import CellRunner, NetConfig

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.builtin import new_command_register_with_builtin_module
from nvflare.fuel.hci.server.login import LoginModule, SessionManager, SimpleAuthenticator
from nvflare.fuel.hci.security import hash_password


class Server(CellRunner, CommandModule):

    def __init__(
            self,
            config_path: str,
    ):
        net_config = NetConfig()
        admin_host, admin_port = net_config.get_admin()
        if not admin_host or not admin_port:
            raise RuntimeError("missing admin host/port in net config")

        CellRunner.__init__(
            self,
            config_path=config_path,
            my_name=FQCN.ROOT_SERVER,
        )

        # set up admin server
        users = {"admin": hash_password("admin")}
        cmd_reg = new_command_register_with_builtin_module(app_ctx=self)
        authenticator = SimpleAuthenticator(users)
        sess_mgr = SessionManager()
        login_module = LoginModule(authenticator, sess_mgr)
        cmd_reg.register_module(login_module)
        cmd_reg.register_module(sess_mgr)
        cmd_reg.register_module(self)
        self.sess_mgr = sess_mgr

        self.admin = AdminServer(
            cmd_reg=cmd_reg,
            host=admin_host,
            port=int(admin_port)
        )

    def start(self):
        super().start()
        self.admin.start()

    def stop(self):
        self.sess_mgr.shutdown()
        self.admin.stop()
        super().stop()

    def get_spec(self) -> CommandModuleSpec:
        return CommandModuleSpec(
            name="sys",
            cmd_specs=[
                CommandSpec(
                    name="cells",
                    description="get system cells info",
                    usage="cells",
                    handler_func=self._cmd_cells,
                    visible=True,
                ),
                CommandSpec(
                    name="stop",
                    description="stop system",
                    usage="stop",
                    handler_func=self._cmd_stop,
                    visible=True,
                )])

    def _cmd_cells(self, conn: Connection, args: [str]):
        cell_fqcns = self.request_cells_info()
        for c in cell_fqcns:
            conn.append_string(c)

    def _cmd_stop(self, conn: Connection, args: [str]):
        conn.append_string("system stopped")
        self.stop()
