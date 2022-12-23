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
import os
import signal

from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.fuel.f3.cellnet import FQCN, new_message, MessageHeaderKey, ReturnCode
from .cell_runner import CellRunner, NetConfig

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.builtin import new_command_register_with_builtin_module
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.login import LoginModule, SessionManager, SimpleAuthenticator
from nvflare.fuel.hci.security import hash_password


class Server(CellRunner, CommandModule):

    def __init__(
            self,
            config_path: str,
            config_file: str
    ):
        net_config = NetConfig(config_file)
        admin_host, admin_port = net_config.get_admin()
        if not admin_host or not admin_port:
            raise RuntimeError("missing admin host/port in net config")

        CellRunner.__init__(
            self,
            config_path=config_path,
            config_file=config_file,
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

    def clean_up(self):
        os.kill(os.getpid(), signal.SIGKILL)
        self.sess_mgr.shutdown()
        self.admin.stop()
        super().clean_up()
        print("SERVER Cleaned Up!")

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
                    name="route",
                    description="send message to a cell and show route",
                    usage="route to_cell [from_cell]",
                    handler_func=self._cmd_route,
                    visible=True,
                ),
                CommandSpec(
                    name="agents",
                    description="show agents of a cell",
                    usage="agents target_cell",
                    handler_func=self._cmd_agents,
                    visible=True,
                ),
                CommandSpec(
                    name="conns",
                    description="show connectors of a cell",
                    usage="conns target_cell",
                    handler_func=self._cmd_connectors,
                    visible=True,
                ),
                CommandSpec(
                    name="url_use",
                    description="show use of a url",
                    usage="url_use url",
                    handler_func=self._cmd_url_use,
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

    def _cmd_url_use(self, conn: Connection, args: [str]):
        if len(args) != 2:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}")
            return
        url = args[1]
        results = self.get_url_use(url)
        conn.append_dict(results)

    def _cmd_route(self, conn: Connection, args: [str]):
        if len(args) < 2:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}")
            return
        target_fqcn = args[1]

        from_fqcn = "server"
        if len(args) > 2:
            from_fqcn = args[2]

        if from_fqcn == "server":
            # from_fqcn not explicitly specified: use server (me)
            reply_headers, req_headers = self.get_route_info(target_fqcn)
        else:
            reply = self.cell.send_request(
                channel="admin",
                topic="start_route",
                target=from_fqcn,
                timeout=1.0,
                request=new_message(payload=target_fqcn)
            )
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == ReturnCode.OK:
                result = reply.payload
                if not isinstance(result, dict):
                    conn.append_error(f"reply payload should be dict but got {type(reply.payload)}")
                    return
                reply_headers = result.get("reply")
                req_headers = result.get("request")
            else:
                conn.append_error(f"error in reply")
                conn.append_dict(reply.headers)
                return

        conn.append_string(f"Route Info from {from_fqcn} to {target_fqcn}")
        conn.append_string("Request Headers:")
        conn.append_dict(req_headers)
        conn.append_string("Reply Headers:")
        conn.append_dict(reply_headers)

    def _cmd_agents(self, conn: Connection, args: [str]):
        if len(args) != 2:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}")
            return

        target_fqcn = args[1]
        if target_fqcn == "server":
            agents = self.get_agents()
            for a in agents:
                conn.append_string(a)
        else:
            reply = self.cell.send_request(
                channel="admin",
                topic="agents",
                target=target_fqcn,
                timeout=1.0,
                request=new_message()
            )
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == ReturnCode.OK:
                result = reply.payload
                if not isinstance(result, list):
                    conn.append_error(f"reply payload should be list but got {type(reply.payload)}")
                    return
                if not result:
                    conn.append_string("not agents")
                else:
                    for a in result:
                        conn.append_string(a)
            else:
                conn.append_error("Error processing command")
                conn.append_dict(reply.headers)

    def _cmd_connectors(self, conn: Connection, args: [str]):
        if len(args) != 2:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}")
            return

        target_fqcn = args[1]
        if target_fqcn == "server":
            result = self.get_connectors()
            conn.append_dict(result)
            return

        reply = self.cell.send_request(
            channel="admin",
            topic="connectors",
            target=target_fqcn,
            timeout=1.0,
            request=new_message()
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc == ReturnCode.OK:
            result = reply.payload
            if not isinstance(result, dict):
                conn.append_error(f"reply payload should be dict but got {type(reply.payload)}")
                return
            if not result:
                conn.append_string("not connectors")
            else:
                conn.append_dict(result)
        else:
            conn.append_error("Error processing command")
            conn.append_dict(reply.headers)

    def _cmd_stop(self, conn: Connection, args: [str]):
        result = self.stop()
        conn.append_dict(result)
        conn.append_shutdown("System Stopped")
