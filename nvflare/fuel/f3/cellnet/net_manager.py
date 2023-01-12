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

from .net_agent import NetAgent

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.constants import ConnProps


class NetManager(CommandModule):

    def __init__(
            self,
            agent: NetAgent
    ):
        self.agent = agent

    def get_spec(self) -> CommandModuleSpec:
        return CommandModuleSpec(
            name="cellnet",
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
                    name="speed",
                    description="test communication speed between cells",
                    usage="speed from_fqcn to_fqcn [num_tries] [payload_size]",
                    handler_func=self._cmd_speed_test,
                    visible=True,
                ),
                CommandSpec(
                    name="stress",
                    description="stress test communication among cells",
                    usage="stress [num_tries] [timeout]",
                    handler_func=self._cmd_stress_test,
                    visible=True,
                ),
                CommandSpec(
                    name="bulk",
                    description="test bulk message processing",
                    usage="bulk",
                    handler_func=self._cmd_bulk_test,
                    visible=True,
                ),
                CommandSpec(
                    name="change_root",
                    description="change to a new root server",
                    usage="change_root url",
                    handler_func=self._cmd_change_root,
                    visible=True,
                ),
                CommandSpec(
                    name="msg_stats",
                    description="show request stats",
                    usage="msg_stats target [mode]",
                    handler_func=self._cmd_msg_stats,
                    visible=True,
                ),
                CommandSpec(
                    name="shutdown",
                    description="shutdown the whole cellnet",
                    usage="shutdown",
                    handler_func=self._cmd_shutdown,
                    visible=True,
                )])

    def _cmd_cells(self, conn: Connection, args: [str]):
        err, cell_fqcns = self.agent.request_cells_info()
        if err:
            conn.append_error(err)

        for c in cell_fqcns:
            conn.append_string(c)

    def _cmd_url_use(self, conn: Connection, args: [str]):
        if len(args) != 2:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}")
            return
        url = args[1]
        results = self.agent.get_url_use(url)
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

        err, reply_headers, req_headers = self.agent.start_route(from_fqcn, target_fqcn)
        conn.append_string(f"Route Info from {from_fqcn} to {target_fqcn}")
        if err:
            conn.append_error(err)
        if req_headers:
            conn.append_string("Request Headers:")
            conn.append_dict(req_headers)
        if reply_headers:
            conn.append_string("Reply Headers:")
            conn.append_dict(reply_headers)

    def _cmd_agents(self, conn: Connection, args: [str]):
        if len(args) != 2:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}")
            return

        target_fqcn = args[1]
        err_dict, agents = self.agent.get_agents(target_fqcn)
        if err_dict:
            conn.append_dict(err_dict)
        if agents:
            for a in agents:
                conn.append_string(a)

    def _cmd_connectors(self, conn: Connection, args: [str]):
        if len(args) != 2:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}")
            return
        target_fqcn = args[1]
        err_dict, result = self.agent.get_connectors(target_fqcn)
        if err_dict:
            conn.append_dict(err_dict)
        if result:
            conn.append_dict(result)

    def _cmd_speed_test(self, conn: Connection, args: [str]):
        if len(args) < 3:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}")
            return
        from_fqcn = args[1]
        to_fqcn = args[2]
        num_tries = 100
        payload_size = 1000
        if len(args) > 3:
            num_tries = int(args[3])
        if len(args) > 4:
            payload_size = int(args[4])

        result = self.agent.speed_test(
            from_fqcn=from_fqcn,
            to_fqcn=to_fqcn,
            num_tries=num_tries,
            payload_size=payload_size
        )
        conn.append_dict(result)

    def _cmd_stress_test(self, conn: Connection, args: [str]):
        num_tries = 10
        timeout = 5.0
        if len(args) > 1:
            num_tries = int(args[1])
        if len(args) > 2:
            timeout = int(args[2])
        err, targets = self.agent.request_cells_info()
        if err:
            conn.append_error(err)

        if not targets:
            conn.append_error("no targets to test")

        conn.append_string(f"starting stress test on {targets}", flush=True)
        result = self.agent.start_stress_test(
            targets=targets,
            num_rounds=num_tries,
            timeout=timeout
        )

        total_errors = 0
        for t, v in result.items():
            if not isinstance(v, dict):
                continue
            err_dict = v.get("errors")
            cell_errs = 0
            for _, c in err_dict.items():
                cell_errs += c
            total_errors += cell_errs
            if cell_errs == 0:
                v.pop('errors')
        conn.append_dict(result)
        conn.append_string(f"total errors: {total_errors}")

    def _cmd_bulk_test(self, conn: Connection, args: [str]):
        bulk_size = 1
        if len(args) > 1:
            try:
                bulk_size = int(args[1])
            except:
                conn.append_error(f"invalid bulk_size {args[1]} - must be int")
                return

        err, targets = self.agent.request_cells_info()
        if err:
            conn.append_error(err)

        if not targets:
            conn.append_error("no targets to test")

        conn.append_string(f"starting bulk test on {targets}", flush=True)
        result = self.agent.start_bulk_test(targets, bulk_size)
        conn.append_dict(result)

    def _cmd_msg_stats(self, conn: Connection, args: [str]):
        if len(args) < 2:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}")
            return

        valid_modes = ['avg', 'count', 'percent', 'min', 'max']
        target = args[1]
        mode = 'count'
        if len(args) > 2:
            mode = args[2]

        if mode.startswith('p'):
            mode = 'percent'
        elif mode.startswith('c'):
            mode = 'count'
        elif mode.startswith('a'):
            mode = 'avg'

        if mode not in valid_modes:
            conn.append_error(f"invalid mode '{mode}': must be one of {valid_modes}")
            return

        reply = self.agent.get_msg_stats_table(target, mode)
        if isinstance(reply, str):
            conn.append_error(reply)
            return
        if not isinstance(reply, dict):
            conn.append_error(f"expect dict bt got {type(reply)}")
            return
        t = conn.append_table(reply.get("headers"))
        rows = reply.get("rows")
        for _, c in rows.items():
            t.add_row(c)

    def _cmd_change_root(self, conn: Connection, args: [str]):
        if len(args) < 2:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_string(f"Usage: {cmd_entry.usage}")
            return

        url = args[1]
        self.agent.change_root(url)
        conn.append_shutdown("Root Changed. Good Bye!")

    def _cmd_shutdown(self, conn: Connection, args: [str]):
        self.agent.stop()
        conn.append_shutdown("System Stopped")
