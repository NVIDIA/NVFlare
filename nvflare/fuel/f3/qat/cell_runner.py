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

from nvflare.fuel.f3.cellnet import Cell, FQCN, Message, Headers, MessageHeaderKey, ReturnCode, new_message
from .net_config import NetConfig

from typing import Union
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

        self.child_runners = {}
        self.client_runners = {}

        self.cell.register_request_cb(
            channel='admin',
            topic='cells',
            cb=self._do_report_cells,
            runner=self
        )

        self.cell.register_request_cb(
            channel='admin',
            topic='route',
            cb=self._do_route,
            runner=self
        )

        self.cell.register_request_cb(
            channel='admin',
            topic='stop',
            cb=self._do_stop,
            runner=self
        )

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

    def _do_stop(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: Message,
            runner
    ) -> Union[None, Message]:
        self.stop()
        return None

    def _do_route(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: Message,
            runner
    ) -> Union[None, Message]:
        req_headers = request.headers
        return new_message(payload=req_headers)

    def request_cells_info(self):
        result = [self.cell.get_fqcn()]
        targets = []
        if self.child_runners:
            for _, r in self.child_runners.items():
                targets.append(r.fqcn)

        if self.client_runners:
            for _, r in self.client_runners.items():
                targets.append(r.fqcn)

        if targets:
            replies = self.cell.broadcast_request(
                channel='admin',
                topic='cells',
                targets=targets,
                request=Message(Headers(), None),
                timeout=2.0
            )
            for t, r in replies.items():
                assert isinstance(r, Message)
                rc = r.get_header(MessageHeaderKey.RETURN_CODE)
                if rc == ReturnCode.OK:
                    sub_result = r.payload
                    result.extend(sub_result)
                else:
                    print(f"no reply from {t}: {rc}")
        return result

    def _do_report_cells(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: Message,
            runner
    ) -> Union[None, Message]:
        results = self.request_cells_info()
        return new_message(payload=results)

    def stop(self):
        if self.asked_to_stop:
            return

        # ask all children to stop
        sub_runners = []
        for _, r in self.child_runners.items():
            sub_runners.append(r)

        for _, r in self.client_runners.items():
            sub_runners.append(r)

        if sub_runners:
            targets = [x.fqcn for x in sub_runners]
            self.cell.fire_and_forget(
                channel='admin',
                topic='stop',
                message=Message(Headers(), payload=None),
                targets=targets
            )

            # wait for sub processes
            for r in sub_runners:
                try:
                    r.process.wait(timeout=2.0)
                except:
                    print(f"subprocess {r.fqcn} did not end gracefully")
                    r.process.terminate()

        print(f"======= {self.cell.get_fqcn()} is asked to stop!")
        self.asked_to_stop = True
        self.cell.stop()

    def run(self):
        while not self.asked_to_stop:
            time.sleep(0.5)
        print(f"====== {self.cell.get_fqcn()} STOPPED!")
