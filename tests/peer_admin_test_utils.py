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

"""Site-owned probe and passive evidence for the provisioned peer-admin regression.

Loaded through normal local resources configuration; no security gate is replaced.
The probe uses only the enrolled attacker's existing Cell and authentication filters.
"""

import json
import os
import threading
import traceback
from pathlib import Path

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.private.admin_defs import Message
from nvflare.private.defs import CellChannel, CellMessageHeaderKeys, SysCommandTopic, new_cell_message


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def _metadata(message):
    endpoint = message.get_prop(MessagePropKey.ENDPOINT)
    return {
        "pid": os.getpid(),
        "origin": message.get_header(MessageHeaderKey.ORIGIN),
        "destination": message.get_header(MessageHeaderKey.DESTINATION),
        "endpoint": endpoint.name if endpoint else None,
        "route": [hop[0] for hop in message.get_header(MessageHeaderKey.ROUTE, [])],
        "topic": message.get_header(MessageHeaderKey.TOPIC),
        "type": message.get_header(MessageHeaderKey.MSG_TYPE),
    }


class PeerAdminProbe(FLComponent):
    def __init__(self, directory, target=None):
        super().__init__()
        self.directory = Path(directory)
        self.target = target
        self.stop = threading.Event()
        self.worker = None
        self.lock = threading.Lock()
        self.forge_origin = False
        self.omit_token = False

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.SYSTEM_START:
            cell = fl_ctx.get_engine().get_cell()
            # Append to the production wildcard chain. A more-specific registration
            # would shadow that chain and invalidate this security regression.
            cell.core_cell.add_incoming_filter("*", "*", self.observe)
            marker = f"{fl_ctx.get_identity_name()} pid={os.getpid()} marker={self.directory.name}\n"
            Path("peer-marker.txt").write_text(marker)
            if self.target:
                cell.core_cell.add_outgoing_request_filter("*", "*", self.forge_server)
                self.worker = threading.Thread(target=self.probe, args=(cell,))
                self.worker.start()
            _write_json(
                self.directory / "ready.json",
                {
                    "pid": os.getpid(),
                    "marker": marker,
                    "incoming_filters": [
                        f.cb.__qualname__
                        for f in cell.core_cell.in_filter_reg.find(CellChannel.CLIENT_MAIN, SysCommandTopic.SHELL)
                    ],
                    "outgoing_filters": [
                        f.cb.__qualname__
                        for f in (
                            cell.core_cell.out_req_filter_reg.find(CellChannel.CLIENT_MAIN, SysCommandTopic.SHELL) or []
                        )
                    ],
                },
            )
        elif event_type == EventType.SYSTEM_END:
            self.stop.set()
            if self.worker:
                self.worker.join(timeout=15)

    def observe(self, message):
        if message.get_header(MessageHeaderKey.CHANNEL) == CellChannel.CLIENT_MAIN:
            with self.lock, (self.directory / "received.jsonl").open("a") as stream:
                stream.write(json.dumps(_metadata(message)) + "\n")

    def forge_server(self, message):
        if message.get_header(MessageHeaderKey.CHANNEL) != CellChannel.CLIENT_MAIN:
            return
        if self.omit_token:
            message.remove_header(CellMessageHeaderKeys.TOKEN)
        if self.forge_origin:
            # An enrolled hostile site can modify its own outgoing wire headers.
            # Keep its real token/signature; do not impersonate any TLS endpoint.
            message.set_header(MessageHeaderKey.ORIGIN, "server")
            message.remove_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED)

    def probe(self, cell):
        try:
            while not self.stop.wait(0.05):
                plan = self.directory / "request.json"
                if not plan.exists():
                    continue
                cases = json.loads(plan.read_text())
                plan.unlink()
                results = []
                for case in cases:
                    self.forge_origin = case.get("forge_origin", False)
                    self.omit_token = case.get("omit_token", False)
                    payload = Message(topic=SysCommandTopic.SHELL, body="cat peer-marker.txt")
                    for key, value in case.get("headers", {}).items():
                        payload.set_header(key, value)
                    reply = cell.send_request(
                        channel=CellChannel.CLIENT_MAIN,
                        topic=SysCommandTopic.SHELL,
                        target=self.target,
                        request=new_cell_message({}, payload),
                        timeout=10,
                    )
                    results.append(
                        dict(
                            _metadata(reply),
                            case=case["name"],
                            rc=reply.get_header(MessageHeaderKey.RETURN_CODE),
                            body=reply.payload.body if isinstance(reply.payload, Message) else None,
                        )
                    )
                _write_json(self.directory / "results.json", results)
        except Exception:
            _write_json(self.directory / "results.json", {"error": traceback.format_exc()})
