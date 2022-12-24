#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from .cell import Cell, Message
from .connector_manager import ConnectorInfo
from .defs import MessageHeaderKey, ReturnCode
from .fqcn import FQCN
from .utils import make_reply, new_message

from typing import Union, List

_CHANNEL = "_net_manager"
_TOPIC_AGENTS = "agents"
_TOPIC_CELLS = "cells"
_TOPIC_ROUTE = "route"
_TOPIC_START_ROUTE = "start_route"
_TOPIC_STOP = "stop"
_TOPIC_URL_USE = "url_use"
_TOPIC_CONNS = "conns"


class NetBot:

    def __init__(self, cell: Cell):
        self.cell = cell

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_CELLS,
            cb=self._do_report_cells,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_ROUTE,
            cb=self._do_route,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_START_ROUTE,
            cb=self._do_start_route,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_STOP,
            cb=self._do_stop,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_AGENTS,
            cb=self._do_agents,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_CONNS,
            cb=self._do_connectors,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_URL_USE,
            cb=self._do_url_use,
        )

    def _do_stop(
            self,
            request: Message
    ) -> Union[None, Message]:
        print(f"============== {self.cell.get_fqcn()}: GOT Stop Request")
        self.stop()
        return None

    def _do_route(
            self,
            request: Message
    ) -> Union[None, Message]:
        req_headers = request.headers
        return new_message(payload=req_headers)

    def _do_start_route(
            self,
            request: Message
    ) -> Union[None, Message]:
        target_fqcn = request.payload
        err = FQCN.validate(target_fqcn)
        if err:
            return make_reply(ReturnCode.PROCESS_EXCEPTION, f"bad target fqcn {err}")
        assert isinstance(target_fqcn, str)
        reply_headers, req_headers = self.get_route_info(target_fqcn)
        return new_message(payload={"request": req_headers, "reply": reply_headers})

    def _get_agents(self) -> List[str]:
        return list(self.cell.agents.keys())

    def _do_agents(
            self,
            request: Message
    ) -> Union[None, Message]:
        return new_message(payload=self._get_agents())

    def get_agents(self, target_fqcn: str) -> (Union[None, dict], List[str]):
        if target_fqcn == self.cell.get_fqcn():
            return None, self._get_agents()

        reply = self.cell.send_request(
            channel=_CHANNEL,
            topic=_TOPIC_AGENTS,
            target=target_fqcn,
            timeout=1.0,
            request=new_message()
        )

        err = ""
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc == ReturnCode.OK:
            result = reply.payload
            if not isinstance(result, list):
                err = f"reply payload should be list but got {type(reply.payload)}"
                result = None
        else:
            result = None
            err = f"return code: {rc}"

        if err:
            return {"error": err, "reply": reply.headers}, None
        else:
            return None, result

    @staticmethod
    def _connector_info(info: ConnectorInfo) -> dict:
        return {
            "url": info.connect_url,
            "handle": info.handle,
            "type": "connector" if info.active else "listener"
        }

    def _get_connectors(self) -> dict:
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
            request: Message
    ) -> Union[None, Message]:
        return new_message(payload=self._get_connectors())

    def get_connectors(self, target_fqcn: str) -> (dict, dict):
        if target_fqcn == self.cell.get_fqcn():
            return {}, self._get_connectors()

        reply = self.cell.send_request(
            channel=_CHANNEL,
            topic=_TOPIC_CONNS,
            target=target_fqcn,
            timeout=1.0,
            request=new_message()
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc == ReturnCode.OK:
            result = reply.payload
            if not isinstance(result, dict):
                return {
                    "error": f"reply payload should be dict but got {type(reply.payload)}",
                    "reply": reply.headers
                }, {}
            if not result:
                return {}, {}
            else:
                return {}, result
        else:
            return {
                "error": "processing error",
                "reply": reply.headers
            }, {}

    def request_cells_info(self) -> List[str]:
        result = [self.cell.get_fqcn()]
        replies = self._broadcast_to_subs(topic=_TOPIC_CELLS)
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

    def get_url_use(self, url) -> dict:
        result = {self.cell.get_fqcn(): self._get_url_use_of_cell(url)}
        replies = self._broadcast_to_subs(
            topic=_TOPIC_URL_USE,
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
            request: Message
    ) -> Union[None, Message]:
        results = self.get_url_use(request.payload)
        return new_message(payload=results)

    def get_route_info(self, target_fqcn: str) -> (dict, dict):
        reply = self.cell.send_request(
            channel=_CHANNEL,
            topic=_TOPIC_ROUTE,
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

    def start_route(self, from_fqcn: str, target_fqcn: str) -> (str, dict, dict):
        err = ""
        reply_headers = {}
        req_headers = {}
        if from_fqcn == self.cell.get_fqcn():
            # from_fqcn not explicitly specified: use server (me)
            reply_headers, req_headers = self.get_route_info(target_fqcn)
        else:
            reply = self.cell.send_request(
                channel=_CHANNEL,
                topic=_TOPIC_START_ROUTE,
                target=from_fqcn,
                timeout=1.0,
                request=new_message(payload=target_fqcn)
            )
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == ReturnCode.OK:
                result = reply.payload
                if not isinstance(result, dict):
                    err = f"reply payload should be dict but got {type(reply.payload)}"
                else:
                    reply_headers = result.get("reply")
                    req_headers = result.get("request")
            else:
                err = f"error in reply {rc}"
                reply_headers = reply.headers
        return err, reply_headers, req_headers

    def _do_report_cells(
            self,
            request: Message
    ) -> Union[None, Message]:
        results = self.request_cells_info()
        return new_message(payload=results)

    def stop(self):
        # ask all children to stop
        self._broadcast_to_subs(topic=_TOPIC_STOP, timeout=0.0)
        self.cell.stop()

    def _broadcast_to_subs(
            self,
            topic: str,
            message=None,
            timeout=1.0
    ):
        if not message:
            message = new_message()

        children, clients = self.cell.get_sub_cell_names()
        targets = []
        targets.extend(children)
        targets.extend(clients)

        if targets:
            if timeout > 0.0:
                return self.cell.broadcast_request(
                    channel=_CHANNEL,
                    topic=topic,
                    targets=targets,
                    request=message,
                    timeout=timeout
                )
            else:
                print(f"============= {self.cell.get_fqcn()}: broadcasting {topic} to {targets} ...")
                self.cell.fire_and_forget(
                    channel=_CHANNEL,
                    topic=topic,
                    targets=targets,
                    message=message
                )
                print(f"============= {self.cell.get_fqcn()}: broadcasted {topic} to {targets}")
        return {}
