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

import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.cellnet.net_manager import NetManager
from nvflare.fuel.f3.mpm import MainProcessMonitor
from nvflare.fuel.hci.security import hash_password
from nvflare.fuel.hci.server.builtin import new_command_register_with_builtin_module
from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.fuel.hci.server.login import LoginModule, SessionManager, SimpleAuthenticator


def build_cell(name: str, root_url: str):
    cell = Cell(
        fqcn=name,
        root_url=root_url,
        secure=False,
        credentials={},
    )

    agent = NetAgent(cell, None, None)
    return cell, agent


class Combo:
    def __init__(self, num_clients, root_url, admin_port, max_workers):
        self.workers = []
        self.root_url = root_url
        self.max_workers = max_workers
        for i in range(max_workers):
            self.workers[i] = None

        self.server_cell, server_agent = build_cell("server", root_url)
        net_mgr = NetManager(server_agent, diagnose=True)
        server_agent.agent_closed_cb = self.stop

        start = time.time()
        for i in range(num_clients):
            name = f"c{i}"
            build_cell(name, root_url)
        print(f"===== Time for creating {len(Cell.ALL_CELLS)} cells: {time.time()-start} secs")

        users = {"admin": hash_password("admin")}
        cmd_reg = new_command_register_with_builtin_module(app_ctx=None)
        authenticator = SimpleAuthenticator(users)
        sess_mgr = SessionManager()
        login_module = LoginModule(authenticator, sess_mgr)
        cmd_reg.register_module(login_module)
        cmd_reg.register_module(sess_mgr)
        cmd_reg.register_module(net_mgr)

        self.admin = AdminServer(cmd_reg=cmd_reg, host="localhost", port=admin_port)
        MainProcessMonitor.add_cleanup_cb(self._cleanup)
        self.waiter = threading.Event()

    def _cleanup(self):
        self.admin.stop()
        start = time.time()
        for c in Cell.ALL_CELLS.values():
            assert isinstance(c, Cell)
            try:
                c.stop()
            except:
                traceback.print_exc()
        print(f"Time for stopping {len(Cell.ALL_CELLS)} cells: {time.time()-start} secs")

    def start(self):
        print("starting cells ...")
        start = time.time()
        for c in Cell.ALL_CELLS.values():
            assert isinstance(c, Cell)
            c.start()
        print(f"Time for starting {len(Cell.ALL_CELLS)} cells: {time.time()-start} secs")
        self.admin.start()

    def run(self):
        time.sleep(1.0)
        for i in range(self.max_workers):
            fqcn = f"c{i}.1234"
            p = self._create_subprocess(fqcn)
            print(f"Created worker {fqcn}")
            self.workers[i] = p
        self.waiter.wait()

    def _monitor_workers(self):
        while True:
            for i in range(self.max_workers):
                p = self.workers[i]

    def stop(self):
        # self.agent.stop()
        self.waiter.set()

    def _create_subprocess(self, name: str):
        parts = [f"{sys.executable} -m combo_worker", f"-r {self.root_url}", f"-n {name}", f"-t {time.time()}"]
        command = " ".join(parts)
        print(f"Start Worker Command: {command}")
        return subprocess.Popen(shlex.split(command), preexec_fn=os.setsid, env=os.environ.copy())


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    num_clients = 1000
    root_url = "grpc://localhost:8008"
    admin_port = 8003
    combo = Combo(num_clients, root_url, admin_port)
    combo.start()
    combo.run()


if __name__ == "__main__":
    MainProcessMonitor.run(main)
