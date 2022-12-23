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

from nvflare.fuel.f3.cellnet import (
    Cell, FQCN, Message, Headers, MessageHeaderKey, ReturnCode,
    new_message, make_reply)
from nvflare.fuel.f3.connector_manager import ConnectorInfo
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
        )

        self.cell.register_request_cb(
            channel='admin',
            topic='route',
            cb=self._do_route,
        )

        self.cell.register_request_cb(
            channel='admin',
            topic='start_route',
            cb=self._do_start_route,
        )

        self.cell.register_request_cb(
            channel='admin',
            topic='stop',
            cb=self._do_stop,
        )

        self.cell.register_request_cb(
            channel='admin',
            topic='agents',
            cb=self._do_agents,
        )

        self.cell.register_request_cb(
            channel='admin',
            topic='connectors',
            cb=self._do_connectors,
        )

        self.cell.register_request_cb(
            channel='admin',
            topic='url_use',
            cb=self._do_url_use,
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
            request: Message
    ) -> Union[None, Message]:
        replies = self.stop()
        return new_message(payload=replies)

    def _do_route(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: Message
    ) -> Union[None, Message]:
        req_headers = request.headers
        return new_message(payload=req_headers)

    def _do_start_route(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: Message
    ) -> Union[None, Message]:
        target_fqcn = request.payload
        if not isinstance(target_fqcn, str):
            return make_reply(ReturnCode.PROCESS_EXCEPTION, "bad target fqcn")
        reply_headers, req_headers = self.get_route_info(target_fqcn)
        return new_message(payload={"request": req_headers, "reply": reply_headers})

    def get_agents(self):
        return list(self.cell.agents.keys())

    def _do_agents(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: Message
    ) -> Union[None, Message]:
        return new_message(payload=self.get_agents())

    def _connector_info(self, info: ConnectorInfo) -> dict:
        return {
            "url": info.connect_url,
            "handle": info.handle,
            "type": "connector" if info.active else "listener"
        }

    def get_connectors(self) -> dict:
        cell = self.cell
        result = {}
        if cell.int_listener:
            result["int_listener"] = self._connector_info(cell.int_listener)
        if cell.ext_listener:
            result["ext_listener"] = self._connector_info(cell.ext_listener)
        if cell.bb_ext_connector:
            result["bb_ext_connector"] = self._connector_info(cell.bb_ext_connector)
        if cell.bb_int_connector:
            result["bb_int_connector"] = self._connector_info(cell.bb_int_connector)
        if cell.adhoc_connectors:
            conns = {}
            for k, v in cell.adhoc_connectors.items():
                conns[k] = self._connector_info(v)
            result["adhoc_connectors"] = conns
        return result

    def _do_connectors(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: Message
    ) -> Union[None, Message]:
        return new_message(payload=self.get_connectors())

    def request_cells_info(self):
        result = [self.cell.get_fqcn()]
        replies = self._bcast_to_subs(topic='cells')
        for t, r in replies.items():
            assert isinstance(r, Message)
            rc = r.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == ReturnCode.OK:
                sub_result = r.payload
                result.extend(sub_result)
            else:
                result.append(f"no reply from {t}: {rc}")
        return result

    def _get_url_use_of_cell(self, url: str):
        cell = self.cell
        if cell.int_listener and cell.int_listener.connect_url == url:
            return "int_listen"
        if cell.ext_listener and cell.ext_listener.connect_url == url:
            return "ext_listen"
        if cell.bb_ext_connector and cell.bb_ext_connector.connect_url == url:
            return "bb_ext_connect"
        if cell.bb_int_connector and cell.bb_int_connector.connect_url == url:
            return "int_connect"
        if cell.adhoc_connectors:
            for _, h in cell.adhoc_connectors.items():
                if h.connect_url == url:
                    return "adhoc_connect"
        return "none"

    def get_url_use(self, url):
        result = {self.cell.get_fqcn(): self._get_url_use_of_cell(url)}
        replies = self._bcast_to_subs(
            topic='url_use',
            message=new_message(payload=url)
        )
        for t, r in replies.items():
            assert isinstance(r, Message)
            rc = r.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == ReturnCode.OK:
                if not isinstance(r.payload, dict):
                    result[t] = f"bad reply type {type(r.payload)}"
                else:
                    result.update(r.payload)
            else:
                result[t] = f"error {rc}"
        return result

    def _do_url_use(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: Message
    ) -> Union[None, Message]:
        results = self.get_url_use(request.payload)
        return new_message(payload=results)

    def get_route_info(self, target_fqcn: str) -> (dict, dict):
        reply = self.cell.send_request(
            channel="admin",
            topic="route",
            target=target_fqcn,
            timeout=1.0,
            request=new_message()
        )
        reply_headers = reply.headers
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
        if rc == ReturnCode.OK:
            if not isinstance(reply.payload, dict):
                return reply_headers, {"error": f"reply payload got {type(reply.payload)}"}
            return reply_headers, reply.payload
        else:
            return reply_headers, {"error": f"Reply ReturnCode: {rc}"}

    def _do_report_cells(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: Message
    ) -> Union[None, Message]:
        results = self.request_cells_info()
        return new_message(payload=results)

    def stop(self):
        if self.asked_to_stop:
            return {}

        result = {self.cell.get_fqcn(): 'done'}

        # ask all children to stop
        replies = self._bcast_to_subs(topic='stop')
        for t, v in replies.items():
            result.update(v.payload)

        sub_runners = []
        for _, r in self.child_runners.items():
            sub_runners.append(r)

        for _, r in self.client_runners.items():
            sub_runners.append(r)

        for r in sub_runners:
            r.process.terminate()

        print(f"======= {self.cell.get_fqcn()} is asked to stop!")
        self.asked_to_stop = True
        return result

    def _bcast_to_subs(
            self,
            topic: str,
            message=None,
            timeout=1.0
    ):
        if not message:
            message = new_message()

        targets = []
        if self.child_runners:
            for _, r in self.child_runners.items():
                targets.append(r.fqcn)

        if self.client_runners:
            for _, r in self.client_runners.items():
                targets.append(r.fqcn)

        if targets:
            return self.cell.broadcast_request(
                channel='admin',
                topic=topic,
                targets=targets,
                request=message,
                timeout=timeout
            )
        return {}

    def clean_up(self):
        pass

    def run(self):
        while not self.asked_to_stop:
            time.sleep(0.5)

        print(f"====== {self.cell.get_fqcn()} Cleaning Up ...")
        self.clean_up()
        self.cell.stop()
        print(f"====== {self.cell.get_fqcn()} STOPPED!")
