# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import shlex
import subprocess
import sys
import threading
import time

from nvflare.fuel.f3.cellnet.core_cell import CellAgent, CoreCell, Message, MessageHeaderKey, MessageType
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.mpm import MainProcessMonitor
from nvflare.fuel.f3.stats_pool import StatsPoolManager

from .net_config import NetConfig


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
        parent_fqcn: str = "",
        log_level: str = "info",
    ):
        self.new_root_url = None
        self.config_path = config_path
        self.config_file = config_file
        self.log_level = log_level
        self.waiter = threading.Event()

        if not parent_fqcn:
            my_fqcn = my_name
        else:
            my_fqcn = FQCN.join([parent_fqcn, my_name])

        net_config = NetConfig(config_file)
        self.root_url = net_config.get_root_url()
        self.children = net_config.get_children(my_name)
        self.clients = net_config.get_clients()
        self.create_internal_listener = self.children and len(self.children) > 0

        self.cell = CoreCell(
            fqcn=my_fqcn,
            root_url=self.root_url,
            secure=False,
            credentials={},
            create_internal_listener=self.create_internal_listener,
            parent_url=parent_url,
        )
        self.agent = NetAgent(
            self.cell,
            self._change_root,
            self._agent_closed,
        )

        self.child_runners = {}
        self.client_runners = {}

        self.cell.set_cell_connected_cb(cb=self._cell_connected)
        self.cell.set_cell_disconnected_cb(cb=self._cell_disconnected)
        self.cell.add_incoming_reply_filter(channel="*", topic="*", cb=self._filter_incoming_reply)
        self.cell.add_incoming_request_filter(channel="*", topic="*", cb=self._filter_incoming_request)
        self.cell.add_outgoing_reply_filter(channel="*", topic="*", cb=self._filter_outgoing_reply)
        self.cell.add_outgoing_request_filter(channel="*", topic="*", cb=self._filter_outgoing_request)
        self.cell.set_message_interceptor(cb=self._inspect_message)
        # MainProcessMonitor.add_run_monitor(self._check_new_root)

    def _inspect_message(self, message: Message):
        header_name = "inspected_by"
        inspectors = message.get_header(header_name)
        if not inspectors:
            inspectors = []
            message.set_header(header_name, inspectors)
        inspectors.append(self.cell.get_fqcn())

    def _cell_connected(self, connected_cell: CellAgent):
        self.cell.logger.info(f"{self.cell.get_fqcn()}: Cell {connected_cell.get_fqcn()} connected")

    def _cell_disconnected(self, disconnected_cell: CellAgent):
        self.cell.logger.info(f"{self.cell.get_fqcn()}: Cell {disconnected_cell.get_fqcn()} disconnected")

    def _filter_incoming_reply(self, message: Message):
        channel = message.get_header(MessageHeaderKey.CHANNEL, "")
        topic = message.get_header(MessageHeaderKey.TOPIC, "")
        msg_type = message.get_header(MessageHeaderKey.MSG_TYPE)
        destination = message.get_header(MessageHeaderKey.DESTINATION, "")
        assert len(channel) > 0
        assert len(topic) > 0
        assert msg_type == MessageType.REPLY
        assert destination == self.cell.get_fqcn()
        self.cell.logger.debug(f"{self.cell.get_fqcn()}: _filter_incoming_reply called")

    def _filter_incoming_request(self, message: Message):
        channel = message.get_header(MessageHeaderKey.CHANNEL, "")
        topic = message.get_header(MessageHeaderKey.TOPIC, "")
        msg_type = message.get_header(MessageHeaderKey.MSG_TYPE)
        destination = message.get_header(MessageHeaderKey.DESTINATION, "")
        assert len(channel) > 0
        assert len(topic) > 0
        assert msg_type == MessageType.REQ
        assert destination == self.cell.get_fqcn()
        self.cell.logger.debug(f"{self.cell.get_fqcn()}: _filter_incoming_request called")

    def _filter_outgoing_reply(self, message: Message):
        channel = message.get_header(MessageHeaderKey.CHANNEL, "")
        topic = message.get_header(MessageHeaderKey.TOPIC, "")
        msg_type = message.get_header(MessageHeaderKey.MSG_TYPE)
        origin = message.get_header(MessageHeaderKey.ORIGIN, "")
        assert len(channel) > 0
        assert len(topic) > 0
        assert msg_type == MessageType.REPLY
        assert origin == self.cell.get_fqcn()
        self.cell.logger.debug(f"{self.cell.get_fqcn()}: _filter_outgoing_reply called")

    def _filter_outgoing_request(self, message: Message):
        channel = message.get_header(MessageHeaderKey.CHANNEL, "")
        topic = message.get_header(MessageHeaderKey.TOPIC, "")
        msg_type = message.get_header(MessageHeaderKey.MSG_TYPE)
        origin = message.get_header(MessageHeaderKey.ORIGIN, "")
        assert len(channel) > 0
        assert len(topic) > 0
        assert msg_type == MessageType.REQ
        assert origin == self.cell.get_fqcn()
        self.cell.logger.debug(f"{self.cell.get_fqcn()}: _filter_outgoing_request called")

    def _create_subprocess(self, name: str, parent_fqcn: str, parent_url: str, start_it=True):
        time.sleep(0.2)
        parts = [
            f"{sys.executable} -m run_cell",
            f"-c {self.config_path}",
            f"-f {self.config_file}",
            f"-n {name}",
            f"-l {self.log_level}",
        ]
        if parent_fqcn:
            parts.append(f"-pn {parent_fqcn}")

        if parent_url:
            parts.append(f"-pu {parent_url}")

        command = " ".join(parts)
        print(f"Start Cell Command: {command}")

        if start_it:
            return subprocess.Popen(shlex.split(command), preexec_fn=os.setsid, env=os.environ.copy())
        else:
            return None

    def start(self, start_all=True):
        self.cell.start()

        if self.create_internal_listener:
            # create children
            int_url = self.cell.get_internal_listener_url()
            for child_name in self.children:
                p = self._create_subprocess(
                    name=child_name, parent_url=int_url, parent_fqcn=self.cell.get_fqcn(), start_it=start_all
                )
                child_fqcn = FQCN.join([self.cell.get_fqcn(), child_name])
                info = _RunnerInfo(child_name, child_fqcn, p)
                self.child_runners[child_name] = info

        if self.cell.get_fqcn() == FQCN.ROOT_SERVER and self.clients:
            # I'm the server root: create clients
            time.sleep(1.0)
            for client_name in self.clients:
                p = self._create_subprocess(name=client_name, parent_url="", parent_fqcn="", start_it=start_all)
                self.client_runners[client_name] = _RunnerInfo(client_name, client_name, p)

    def stop(self):
        # self.agent.stop()
        self.waiter.set()

    def _agent_closed(self):
        self.stop()

    def _change_root(self, url: str):
        self.cell.change_server_root(url)

    def dump_stats(self):
        StatsPoolManager.dump_summary(f"{self.cell.get_fqcn()}_stats.json")

    def run(self):
        MainProcessMonitor.set_name(self.cell.get_fqcn())
        MainProcessMonitor.add_cleanup_cb(self.dump_stats)
        MainProcessMonitor.add_cleanup_cb(self.cell.stop)
        self.waiter.wait()
