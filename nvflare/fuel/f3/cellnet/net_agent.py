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
import hashlib
import os
import random
import time

from .cell import Cell, Message
from .connector_manager import ConnectorInfo
from .defs import MessageHeaderKey, ReturnCode
from .fqcn import FQCN
from .utils import make_reply, new_message

from typing import Union, List

_CHANNEL = "_net_manager"
_TOPIC_PEERS = "peers"
_TOPIC_CELLS = "cells"
_TOPIC_ROUTE = "route"
_TOPIC_START_ROUTE = "start_route"
_TOPIC_STOP = "stop"
_TOPIC_STOP_CELL = "stop_cell"
_TOPIC_URL_USE = "url_use"
_TOPIC_CONNS = "conns"
_TOPIC_SPEED = "speed"
_TOPIC_ECHO = "echo"
_TOPIC_STRESS = "stress"
_TOPIC_CHANGE_ROOT = "change_root"
_TOPIC_BULK_TEST = "bulk_test"
_TOPIC_BULK_ITEM = "bulk_item"
_TOPIC_MSG_STATS = "msg_stats"

_ONE_K = bytes([1] * 1024)


class NetAgent:

    def __init__(
            self,
            cell: Cell,
            change_root_cb=None
    ):
        self.cell = cell
        self.change_root_cb = change_root_cb

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
            topic=_TOPIC_STOP_CELL,
            cb=self._do_stop_cell,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_PEERS,
            cb=self._do_peers,
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

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_SPEED,
            cb=self._do_speed,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_ECHO,
            cb=self._do_echo,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_STRESS,
            cb=self._do_stress,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_CHANGE_ROOT,
            cb=self._do_change_root,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_BULK_TEST,
            cb=self._do_bulk_test,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_BULK_ITEM,
            cb=self._do_bulk_item,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_MSG_STATS,
            cb=self._do_msg_stats,
        )

    def _do_stop(
            self,
            request: Message
    ) -> Union[None, Message]:
        print(f"============== {self.cell.get_fqcn()}: GOT Stop Request")
        self.stop()
        return None

    def _do_stop_cell(
            self,
            request: Message
    ) -> Union[None, Message]:
        print(f"============== {self.cell.get_fqcn()}: GOT Stop Request")
        self.stop()
        return new_message()

    def _do_route(
            self,
            request: Message
    ) -> Union[None, Message]:
        return new_message(payload=dict(request.headers))

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
        return new_message(payload={"request": dict(req_headers), "reply": dict(reply_headers)})

    def _get_peers(self) -> List[str]:
        return list(self.cell.agents.keys())

    def _do_peers(
            self,
            request: Message
    ) -> Union[None, Message]:
        return new_message(payload=self._get_peers())

    def get_peers(self, target_fqcn: str) -> (Union[None, dict], List[str]):
        if target_fqcn == self.cell.get_fqcn():
            return None, self._get_peers()

        reply = self.cell.send_request(
            channel=_CHANNEL,
            topic=_TOPIC_PEERS,
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
        if cell.ext_listeners:
            listeners = [self._connector_info(x) for _, x in cell.ext_listeners.items()]
            result["ext_listeners"] = listeners
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

    def request_cells_info(self) -> (str, List[str]):
        result = [self.cell.get_fqcn()]
        err = ""
        replies = self._broadcast_to_subs(topic=_TOPIC_CELLS)
        for t, r in replies.items():
            assert isinstance(r, Message)
            rc = r.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == ReturnCode.OK:
                sub_result = r.payload
                result.extend(sub_result)
            else:
                err = f"no reply from {t}: {rc}"
                result.append(err)
        return err, result

    def _get_url_use_of_cell(self, url: str):
        cell = self.cell
        if cell.int_listener and cell.int_listener.connect_url == url:
            return "int_listen"
        if cell.ext_listeners:
            for k in cell.ext_listeners.keys():
                if k == url:
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
        _, results = self.request_cells_info()
        return new_message(payload=results)

    def stop(self):
        # ask all children to stop
        self._broadcast_to_subs(topic=_TOPIC_STOP, timeout=0.0)
        self.cell.stop()

    def stop_cell(self, target: str) -> str:
        if self.cell.get_fqcn() == target:
            self.stop()
            return ReturnCode.OK
        reply = self.cell.send_request(
            channel=_CHANNEL,
            topic=_TOPIC_STOP_CELL,
            request=new_message(),
            target=target,
            timeout=1.0
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        return rc

    def _request_speed_test(self, target_fqcn: str, num, size) -> Message:
        start = time.perf_counter()
        payload = bytes(_ONE_K * size)
        payload_size = len(payload)
        h = hashlib.md5(payload)
        dig1 = h.digest()
        end = time.perf_counter()
        payload_prep_time = end - start
        errs = 0
        timeouts = 0
        comm_errs = 0
        proc_errs = 0
        size_errs = 0
        start = time.perf_counter()
        for i in range(num):
            r = self.cell.send_request(
                channel=_CHANNEL,
                topic=_TOPIC_ECHO,
                target=target_fqcn,
                request=new_message(payload=payload),
                timeout=10.0
            )
            rc = r.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc == ReturnCode.OK:
                if len(r.payload) != payload_size:
                    self.cell.logger.error(
                        f"{self.cell.get_fqcn()}: expect {payload_size} bytes but received {len(r.payload)}")
                    proc_errs += 1
                else:
                    h = hashlib.md5(r.payload)
                    dig2 = h.digest()
                    if dig1 != dig2:
                        self.cell.logger.error(
                            f"{self.cell.get_fqcn()}: digest mismatch!")
                        proc_errs += 1
            elif rc == ReturnCode.TIMEOUT:
                timeouts += 1
            elif rc == ReturnCode.COMM_ERROR:
                comm_errs += 1
            elif rc == ReturnCode.MSG_TOO_BIG:
                size_errs += 1
            else:
                errs += 1
        end = time.perf_counter()
        total = end - start
        avg = total / num
        return new_message(payload={
            "test": f"{size:,}KB {num} rounds between {self.cell.get_fqcn()} and {target_fqcn}",
            "prep": payload_prep_time,
            "timeouts": timeouts,
            "comm_errors": comm_errs,
            "size_errors": size_errs,
            "proc_errors": proc_errs,
            "other_errors": errs,
            "total": total,
            "average": avg})

    def _do_speed(
            self,
            request: Message
    ) -> Union[None, Message]:
        params = request.payload
        if not isinstance(params, dict):
            return make_reply(ReturnCode.INVALID_REQUEST, f"request body must be dict but got {type(params)}")
        to_fqcn = params.get("to")
        if not to_fqcn:
            return make_reply(ReturnCode.INVALID_REQUEST, "missing 'to' param in request")
        err = FQCN.validate(to_fqcn)
        if err:
            return make_reply(ReturnCode.INVALID_REQUEST, f"bad target FQCN: {err}")
        num = params.get("num", 100)
        size = params.get("size", 1000)
        if size <= 0:
            size = 1000
        if num <= 0:
            num = 100
        return self._request_speed_test(to_fqcn, num, size)

    def _do_echo(
            self,
            request: Message
    ) -> Union[None, Message]:
        return new_message(payload=request.payload)

    def _do_stress_test(self, params):
        if not isinstance(params, dict):
            return {"error": f"bad params - expect dict but got {type(params)}"}
        targets = params.get('targets')
        if not targets:
            return {"error": "no targets specified"}
        num_rounds = params.get('num')
        if not num_rounds:
            return {"error": "missing num of rounds"}
        my_fqcn = self.cell.get_fqcn()
        if my_fqcn in targets:
            targets.remove(my_fqcn)

        if not targets:
            return {"error": "no targets to try"}

        counts = {}
        errors = {}

        start = time.perf_counter()
        for i in range(num_rounds):
            payload = os.urandom(1024)
            h = hashlib.md5(payload)
            d1 = h.digest()
            target = targets[random.randrange(len(targets))]
            req = new_message(payload=payload)
            reply = self.cell.send_request(
                channel=_CHANNEL,
                topic=_TOPIC_ECHO,
                target=target,
                request=req,
                timeout=1.0
            )
            if target not in counts:
                counts[target] = 0
            counts[target] += 1
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.cell.logger.error(f"{self.cell.get_fqcn()}: return code from {target}: {rc}")
                if target not in errors:
                    errors[target] = 0
                errors[target] += 1
            else:
                h = hashlib.md5(reply.payload)
                d2 = h.digest()
                if d1 != d2:
                    self.cell.logger.error(f"{self.cell.get_fqcn()}: digest mismatch from {target}")
                    if target not in errors:
                        errors[target] = 0
                    errors[target] += 1
        end = time.perf_counter()
        return {"counts": counts, "errors": errors, "time": end - start}

    def _do_stress(
            self,
            request: Message
    ) -> Union[None, Message]:
        params = request.payload
        result = self._do_stress_test(params)
        return new_message(payload=result)

    def start_stress_test(self, targets: list, num_rounds=10, timeout=5.0):
        self.cell.logger.info(f"{self.cell.get_fqcn()}: starting stress test on {targets}")
        result = {}
        payload = {"targets": targets, "num": num_rounds}
        msg_targets = [x for x in targets]
        my_fqcn = self.cell.get_fqcn()
        if my_fqcn in msg_targets:
            msg_targets.remove(my_fqcn)
        if not msg_targets:
            return {"error": "no targets for stress test"}

        replies = self.cell.broadcast_request(
            channel=_CHANNEL,
            topic=_TOPIC_STRESS,
            targets=msg_targets,
            request=new_message(payload=payload),
            timeout=timeout
        )
        for t, r in replies.items():
            rc = r.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc != ReturnCode.OK:
                result[t] = f"RC={rc}"
            else:
                result[t] = r.payload
        return result

    def speed_test(
            self,
            from_fqcn: str,
            to_fqcn: str,
            num_tries,
            payload_size) -> dict:
        err = FQCN.validate(from_fqcn)
        if err:
            return {"error": f"invalid from_fqcn {from_fqcn}: {err}"}

        err = FQCN.validate(to_fqcn)
        if err:
            return {"error": f"invalid to_fqcn {to_fqcn}: {err}"}

        if from_fqcn == to_fqcn:
            return {"error": f"from and to FQCNs {from_fqcn} must not be the same"}

        result = {}
        if self.cell.get_fqcn() == from_fqcn:
            reply = self._request_speed_test(to_fqcn, num_tries, payload_size)
        else:
            start = time.perf_counter()
            reply = self.cell.send_request(
                channel=_CHANNEL,
                topic=_TOPIC_SPEED,
                request=new_message(payload={"to": to_fqcn, "num": num_tries, "size": payload_size}),
                target=from_fqcn,
                timeout=100.0
            )
            end = time.perf_counter()
            result['test_time'] = end - start
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
        if rc != ReturnCode.OK:
            result.update({"error": f"return code {rc}"})
        elif not isinstance(reply.payload, dict):
            result.update({"error": f"bad reply: expect dict but got {type(reply.payload)}"})
        else:
            result.update(reply.payload)
        return result

    def change_root(self, new_root_url: str):
        self._broadcast_to_subs(
            topic=_TOPIC_CHANGE_ROOT,
            message=new_message(payload=new_root_url),
            timeout=0.0
        )

    def _do_change_root(
            self,
            request: Message
    ) -> Union[None, Message]:
        new_root_url = request.payload
        assert isinstance(new_root_url, str)
        if self.change_root_cb is not None:
            self.change_root_cb(new_root_url)
        return None

    def start_bulk_test(self, targets: list, size: int):
        self.cell.logger.info(f"{self.cell.get_fqcn()}: starting bulk test on {targets}")
        msg_targets = [x for x in targets]
        my_fqcn = self.cell.get_fqcn()
        if my_fqcn in msg_targets:
            msg_targets.remove(my_fqcn)
        if not msg_targets:
            return {"error": "no targets for bulk test"}

        result = {}
        replies = self.cell.broadcast_request(
            channel=_CHANNEL,
            topic=_TOPIC_BULK_TEST,
            targets=msg_targets,
            request=new_message(payload=size),
            timeout=1.0
        )
        for t, r in replies.items():
            rc = r.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc != ReturnCode.OK:
                result[t] = f"RC={rc}"
            else:
                result[t] = r.payload
        return result

    def _do_bulk_test(
            self,
            request: Message
    ) -> Union[None, Message]:
        size = request.payload
        assert isinstance(size, int)
        nums = []
        for _ in range(size):
            num = random.randint(0, 100)
            nums.append(num)
            msg = new_message(payload=num)
            self.cell.queue_message(
                channel=_CHANNEL,
                topic=_TOPIC_BULK_ITEM,
                targets=FQCN.ROOT_SERVER,
                message=msg,
            )
        return new_message(payload=f"queued: {nums}")

    def _do_bulk_item(
            self,
            request: Message
    ) -> Union[None, Message]:
        num = request.payload
        origin = request.get_header(MessageHeaderKey.ORIGIN)
        self.cell.logger.info(f"{self.cell.get_fqcn()}: got {num} from {origin}")
        return None

    def get_msg_stats_table(self, target: str, mode: str):
        if target == self.cell.get_fqcn():
            headers, rows = self.cell.msg_stats_pool.get_table(mode)
            return {
                "headers": headers,
                "rows": rows
            }
        reply = self.cell.send_request(
            channel=_CHANNEL,
            topic=_TOPIC_MSG_STATS,
            request=new_message(payload={"mode": mode}),
            timeout=1.0,
            target=target
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != ReturnCode.OK:
            return f"error: {rc}"
        return reply.payload

    def _do_msg_stats(
            self,
            request: Message
    ) -> Union[None, Message]:
        p = request.payload
        assert isinstance(p, dict)
        mode = p.get('mode')
        headers, rows = self.cell.msg_stats_pool.get_table(mode)
        reply = {
            "headers": headers,
            "rows": rows
        }
        return new_message(payload=reply)

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
                if self.cell.my_info.is_root and self.cell.my_info.is_on_server:
                    timeout = timeout + 0.1
                else:
                    timeout = timeout / self.cell.my_info.gen
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
