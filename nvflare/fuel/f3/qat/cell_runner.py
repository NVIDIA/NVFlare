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

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.netbot import NetBot
from .net_config import NetConfig

import os
import shlex
import subprocess
import sys
import time


class _RunnerInfo:

    def __init__(self, name: str, fqcn: str, process):
        self.name = name
        self.fqcn = fqcn
        self.process = process


class CellRunner:

    def __init__(
            self,
            config_path: str,
            config_file: str,
            my_name: str,
            parent_url: str = "",
            parent_fqcn: str = ""
    ):
        self.asked_to_stop = False
        self.config_path = config_path
        self.config_file = config_file

        if not parent_fqcn:
            my_fqcn = my_name
        else:
            my_fqcn = FQCN.join([parent_fqcn, my_name])

        net_config = NetConfig(config_file)
        self.root_url = net_config.get_root_url()
        self.children = net_config.get_children(my_name)
        self.clients = net_config.get_clients()
        self.create_internal_listener = self.children and len(self.children) > 0

        self.cell = Cell(
            fqcn=my_fqcn,
            root_url=self.root_url,
            secure=False,
            credentials={},
            create_internal_listener=self.create_internal_listener,
            parent_url=parent_url,
        )
        self.bot = NetBot(self.cell)

        self.child_runners = {}
        self.client_runners = {}

    def _create_subprocess(self, name: str, parent_fqcn: str, parent_url: str):
        parts = [
            f"{sys.executable} -m run_cell",
            f"-c {self.config_path}",
            f"-f {self.config_file}",
            f"-n {name}",
        ]
        if parent_fqcn:
            parts.append(f"-pn {parent_fqcn}")

        if parent_url:
            parts.append(f"-pu {parent_url}")

        command = " ".join(parts)
        return subprocess.Popen(shlex.split(command), preexec_fn=os.setsid, env=os.environ.copy())

    def start(self):
        self.cell.start()
        if self.create_internal_listener:
            # create children
            int_url = self.cell.get_internal_listener_url()
            for child_name in self.children:
                p = self._create_subprocess(
                    name=child_name,
                    parent_url=int_url,
                    parent_fqcn=self.cell.get_fqcn()
                )
                child_fqcn = FQCN.join([self.cell.get_fqcn(), child_name])
                info = _RunnerInfo(child_name, child_fqcn, p)
                self.child_runners[child_name] = info

        if self.cell.get_fqcn() == FQCN.ROOT_SERVER and self.clients:
            # I'm the server root: create clients
            for client_name in self.clients:
                p = self._create_subprocess(
                    name=client_name,
                    parent_url="",
                    parent_fqcn=""
                )
                self.client_runners[client_name] = _RunnerInfo(client_name, client_name, p)

    def stop(self):
        self.asked_to_stop = True
        self.bot.stop()

    def run(self):
        while not self.asked_to_stop:
            time.sleep(0.5)
