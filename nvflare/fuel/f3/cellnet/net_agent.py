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

import copy
import hashlib
import logging
import os
import random
import resource
import threading
import time
from abc import ABC
from typing import List, Union

from nvflare.fuel.f3.cellnet.cell import Cell, Message
from nvflare.fuel.f3.cellnet.connector_manager import ConnectorData
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply, new_message
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.utils.config_service import ConfigService

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
_TOPIC_LIST_POOLS = "list_pools"
_TOPIC_SHOW_POOL = "show_pool"
_TOPIC_COMM_CONFIG = "comm_config"
_TOPIC_CONFIG_VARS = "config_vars"
_TOPIC_PROCESS_INFO = "process_info"
_TOPIC_HEARTBEAT = "heartbeat"

_ONE_K = bytes([1] * 1024)


class _Member:

    STATE_UNKNOWN = 0
    STATE_ONLINE = 1
    STATE_OFFLINE = 2

    def __init__(self, fqcn):
        self.fqcn = fqcn
        self.state = _Member.STATE_UNKNOWN
        self.last_heartbeat_time = time.time()
        self.lock = threading.Lock()


class SubnetMonitor(ABC):
    def __init__(self, subnet_id: str, member_cells: List[str], trouble_alert_threshold: float):
        if not member_cells:
            raise ValueError("member cells must not be empty")
        self.agent = None
        self.subnet_id = subnet_id
        self.trouble_alert_threshold = trouble_alert_threshold
        self.lock = threading.Lock()
        self.members = {}
        for m in member_cells:
            self.members[m] = _Member(m)

    def member_online(self, member_cell_fqcn: str):
        pass

    def member_offline(self, member_cell_fqcn: str):
        pass

    def put_member_online(self, member: _Member):
        with self.lock:
            member.last_heartbeat_time = time.time()
            current_state = member.state
            member.state = member.STATE_ONLINE
            if current_state in [member.STATE_UNKNOWN, member.STATE_OFFLINE]:
                self.member_online(member.fqcn)

    def put_member_offline(self, member: _Member):
        with self.lock:
            if time.time() - member.last_heartbeat_time <= self.trouble_alert_threshold:
                return

            if member.state in [member.STATE_ONLINE]:
                self.member_offline(member.fqcn)
                member.state = member.STATE_OFFLINE

    def stop_subnet(self):
        if not self.agent:
            raise RuntimeError("No NetAgent in this monitor. Make sure the monitor is added to a NetAgent.")
        return self.agent.stop_subnet(self)


class NetAgent:
    def __init__(self, cell: Cell, change_root_cb=None, agent_closed_cb=None):
        self.cell = cell
        self.change_root_cb = change_root_cb
        self.agent_closed_cb = agent_closed_cb
        self.logger = logging.getLogger(self.__class__.__name__)

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

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_LIST_POOLS,
            cb=self._do_list_pools,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_SHOW_POOL,
            cb=self._do_show_pool,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_COMM_CONFIG,
            cb=self._do_comm_config,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_CONFIG_VARS,
            cb=self._do_config_vars,
        )

        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_PROCESS_INFO,
            cb=self._do_process_info,
        )
        cell.register_request_cb(
            channel=_CHANNEL,
            topic=_TOPIC_HEARTBEAT,
            cb=self._do_heartbeat,
        )

        self.heartbeat_thread = None
        self.monitor_thread = None
        self.asked_to_close = False
        self.subnets = {}
        self.monitors = {}
        self.hb_lock = threading.Lock()
        self.monitor_lock = threading.Lock()

    def add_to_subnet(self, subnet_id: str, monitor_fqcn: str = FQCN.ROOT_SERVER):
        with self.hb_lock:
            self.subnets[subnet_id] = monitor_fqcn
            if self.heartbeat_thread is None:
                self.heartbeat_thread = threading.Thread(target=self._subnet_heartbeat)
                self.heartbeat_thread.start()

    def add_subnet_monitor(self, monitor: SubnetMonitor):
        if not isinstance(monitor, SubnetMonitor):
            raise ValueError(f"monitor must be SubnetMonitor but got {type(monitor)}")
        if monitor.subnet_id in self.monitors:
            raise ValueError(f"monitor for subnet {monitor.subnet_id} already exists")
        monitor.agent = self
        with self.monitor_lock:
            self.monitors[monitor.subnet_id] = monitor
            if self.monitor_thread is None:
                self.monitor_thread = threading.Thread(target=self._monitor_subnet)
                self.monitor_thread.start()

    def stop_subnet(self, monitor: SubnetMonitor):
        cells_to_stop = []
        for member_fqcn, member in monitor.members.items():
            if member.state == member.STATE_ONLINE:
                cells_to_stop.append(member_fqcn)
        if cells_to_stop:
            return self.cell.broadcast_request(
                channel=_CHANNEL, topic=_TOPIC_STOP_CELL, request=new_message(), targets=cells_to_stop, timeout=1.0
            )
        else:
            return None

    def delete_subnet_monitor(self, subnet_id: str):
        with self.monitor_lock:
            self.monitors.pop(subnet_id, None)

    def close(self):
        if self.asked_to_close:
            return
        self.asked_to_close = True
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
        if self.agent_closed_cb:
            self.agent_closed_cb()

    def _subnet_heartbeat(self):
        cc = self.cell.comm_configurator
        interval = cc.get_subnet_heartbeat_interval(5.0)
        if interval <= 0:
            interval = 5.0

        while True:
            with self.hb_lock:
                for subnet_id, target in self.subnets.items():
                    self.cell.fire_and_forget(
                        channel=_CHANNEL,
                        topic=_TOPIC_HEARTBEAT,
                        targets=target,
                        message=new_message(payload={"subnet_id": subnet_id}),
                    )

            # wait for interval time, but watch for "asked_to_stop" every 0.1 secs
            start = time.time()
            while True:
                time.sleep(0.1)
                if self.asked_to_close:
                    return
                if time.time() - start >= interval:
                    break

    @staticmethod
    def _check_monitor(m: SubnetMonitor):
        for member_fqcn, member in m.members.items():
            m.put_member_offline(member)

    def _monitor_subnet(self):
        while not self.asked_to_close:
            with self.monitor_lock:
                monitors = copy.copy(self.monitors)
            for _, m in monitors.items():
                self._check_monitor(m)
            time.sleep(0.5)

    def _do_heartbeat(self, request: Message) -> Union[None, Message]:
        origin = request.get_header(MessageHeaderKey.ORIGIN, "?")
        if not self.monitors:
            self.logger.warning(f"got subnet heartbeat from {origin} but no monitors")
            return

        payload = request.payload
        assert isinstance(payload, dict)
        subnet_id = payload.get("subnet_id", "")
        m = self.monitors.get(subnet_id)
        if not m:
            self.logger.warning(f"got subnet heartbeat from {origin} for subnet_id {subnet_id} but no monitor")
            return

        assert isinstance(m, SubnetMonitor)
        member = m.members.get(origin)
        if not member:
            self.logger.warning(f"got subnet heartbeat from {origin} for subnet_id {subnet_id} but it's not a member")
            return

        m.put_member_online(member)

    def _do_stop(self, request: Message) -> Union[None, Message]:
        self.stop()
        return None

    def _do_stop_cell(self, request: Message) -> Union[None, Message]:
        self.stop()
        return new_message()

    def _do_route(self, request: Message) -> Union[None, Message]:
        return new_message(payload=dict(request.headers))

    def _do_start_route(self, request: Message) -> Union[None, Message]:
        target_fqcn = request.payload
        err = FQCN.validate(target_fqcn)
        if err:
            return make_reply(ReturnCode.PROCESS_EXCEPTION, f"bad target fqcn {err}")
        assert isinstance(target_fqcn, str)
        reply_headers, req_headers = self.get_route_info(target_fqcn)
        return new_message(payload={"request": dict(req_headers), "reply": dict(reply_headers)})

    def _do_peers(self, request: Message) -> Union[None, Message]:
        return new_message(payload=list(self.cell.agents.keys()))

    def get_peers(self, target_fqcn: str) -> (Union[None, dict], List[str]):
        reply = self.cell.send_request(
            channel=_CHANNEL, topic=_TOPIC_PEERS, target=target_fqcn, timeout=1.0, request=new_message()
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
    def _connector_info(info: ConnectorData) -> dict:
        return {"url": info.connect_url, "handle": info.handle, "type": "connector" if info.active else "listener"}

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

    def _do_connectors(self, request: Message) -> Union[None, Message]:
        return new_message(payload=self._get_connectors())

    def get_connectors(self, target_fqcn: str) -> (dict, dict):
        reply = self.cell.send_request(
            channel=_CHANNEL, topic=_TOPIC_CONNS, target=target_fqcn, timeout=1.0, request=new_message()
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc == ReturnCode.OK:
            result = reply.payload
            if not isinstance(result, dict):
                return {
                    "error": f"reply payload should be dict but got {type(reply.payload)}",
                    "reply": reply.headers,
                }, {}
            if not result:
                return {}, {}
            else:
                return {}, result
        else:
            return {"error": "processing error", "reply": reply.headers}, {}

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
        replies = self._broadcast_to_subs(topic=_TOPIC_URL_USE, message=new_message(payload=url))
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

    def _do_url_use(self, request: Message) -> Union[None, Message]:
        results = self.get_url_use(request.payload)
        return new_message(payload=results)

    def get_route_info(self, target_fqcn: str) -> (dict, dict):
        reply = self.cell.send_request(
            channel=_CHANNEL, topic=_TOPIC_ROUTE, target=target_fqcn, timeout=1.0, request=new_message()
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
        reply = self.cell.send_request(
            channel=_CHANNEL,
            topic=_TOPIC_START_ROUTE,
            target=from_fqcn,
            timeout=1.0,
            request=new_message(payload=target_fqcn),
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

    def _do_report_cells(self, request: Message) -> Union[None, Message]:
        _, results = self.request_cells_info()
        return new_message(payload=results)

    def stop(self):
        # ask all children to stop
        self._broadcast_to_subs(topic=_TOPIC_STOP, timeout=0.0)
        self.close()

    def stop_cell(self, target: str) -> str:
        # if self.cell.get_fqcn() == target:
        #     self.stop()
        #     return ReturnCode.OK
        reply = self.cell.send_request(
            channel=_CHANNEL, topic=_TOPIC_STOP_CELL, request=new_message(), target=target, timeout=1.0
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
                timeout=10.0,
            )
            rc = r.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc == ReturnCode.OK:
                if len(r.payload) != payload_size:
                    self.cell.logger.error(
                        f"{self.cell.get_fqcn()}: expect {payload_size} bytes but received {len(r.payload)}"
                    )
                    proc_errs += 1
                else:
                    h = hashlib.md5(r.payload)
                    dig2 = h.digest()
                    if dig1 != dig2:
                        self.cell.logger.error(f"{self.cell.get_fqcn()}: digest mismatch!")
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
        return new_message(
            payload={
                "test": f"{size:,}KB {num} rounds between {self.cell.get_fqcn()} and {target_fqcn}",
                "prep": payload_prep_time,
                "timeouts": timeouts,
                "comm_errors": comm_errs,
                "size_errors": size_errs,
                "proc_errors": proc_errs,
                "other_errors": errs,
                "total": total,
                "average": avg,
            }
        )

    def _do_speed(self, request: Message) -> Union[None, Message]:
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

    def _do_echo(self, request: Message) -> Union[None, Message]:
        return new_message(payload=request.payload)

    def _do_stress_test(self, params):
        if not isinstance(params, dict):
            return {"error": f"bad params - expect dict but got {type(params)}"}
        targets = params.get("targets")
        if not targets:
            return {"error": "no targets specified"}
        num_rounds = params.get("num")
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
            reply = self.cell.send_request(channel=_CHANNEL, topic=_TOPIC_ECHO, target=target, request=req, timeout=1.0)
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

    def _do_stress(self, request: Message) -> Union[None, Message]:
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
            timeout=timeout,
        )
        for t, r in replies.items():
            rc = r.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc != ReturnCode.OK:
                result[t] = f"RC={rc}"
            else:
                result[t] = r.payload
        return result

    def speed_test(self, from_fqcn: str, to_fqcn: str, num_tries, payload_size) -> dict:
        err = FQCN.validate(from_fqcn)
        if err:
            return {"error": f"invalid from_fqcn {from_fqcn}: {err}"}

        err = FQCN.validate(to_fqcn)
        if err:
            return {"error": f"invalid to_fqcn {to_fqcn}: {err}"}

        result = {}

        start = time.perf_counter()
        reply = self.cell.send_request(
            channel=_CHANNEL,
            topic=_TOPIC_SPEED,
            request=new_message(payload={"to": to_fqcn, "num": num_tries, "size": payload_size}),
            target=from_fqcn,
            timeout=100.0,
        )
        end = time.perf_counter()
        result["test_time"] = end - start
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
        if rc != ReturnCode.OK:
            result.update({"error": f"return code {rc}"})
        elif not isinstance(reply.payload, dict):
            result.update({"error": f"bad reply: expect dict but got {type(reply.payload)}"})
        else:
            result.update(reply.payload)
        return result

    def change_root(self, new_root_url: str):
        self._broadcast_to_subs(topic=_TOPIC_CHANGE_ROOT, message=new_message(payload=new_root_url), timeout=0.0)

    def _do_change_root(self, request: Message) -> Union[None, Message]:
        new_root_url = request.payload
        assert isinstance(new_root_url, str)
        self.change_root(new_root_url)
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
            timeout=1.0,
        )
        for t, r in replies.items():
            rc = r.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if rc != ReturnCode.OK:
                result[t] = f"RC={rc}"
            else:
                result[t] = r.payload
        return result

    def _do_bulk_test(self, request: Message) -> Union[None, Message]:
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

    def _do_bulk_item(self, request: Message) -> Union[None, Message]:
        num = request.payload
        origin = request.get_header(MessageHeaderKey.ORIGIN)
        self.cell.logger.info(f"{self.cell.get_fqcn()}: got {num} from {origin}")
        return None

    def get_msg_stats_table(self, target: str, mode: str):
        reply = self.cell.send_request(
            channel=_CHANNEL,
            topic=_TOPIC_MSG_STATS,
            request=new_message(payload={"mode": mode}),
            timeout=1.0,
            target=target,
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != ReturnCode.OK:
            return f"error: {rc}"
        return reply.payload

    def _do_msg_stats(self, request: Message) -> Union[None, Message]:
        p = request.payload
        assert isinstance(p, dict)
        mode = p.get("mode")
        headers, rows = self.cell.msg_stats_pool.get_table(mode)
        reply = {"headers": headers, "rows": rows}
        return new_message(payload=reply)

    def get_pool_list(self, target: str):
        reply = self.cell.send_request(
            channel=_CHANNEL, topic=_TOPIC_LIST_POOLS, request=new_message(), timeout=1.0, target=target
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        err = reply.get_header(MessageHeaderKey.ERROR, "")
        if rc != ReturnCode.OK:
            return f"{rc}: {err}"
        return reply.payload

    def _do_list_pools(self, request: Message) -> Union[None, Message]:
        headers, rows = StatsPoolManager.get_table()
        reply = {"headers": headers, "rows": rows}
        return new_message(payload=reply)

    def show_pool(self, target: str, pool_name: str, mode: str):
        reply = self.cell.send_request(
            channel=_CHANNEL,
            topic=_TOPIC_SHOW_POOL,
            request=new_message(payload={"mode": mode, "pool": pool_name}),
            timeout=1.0,
            target=target,
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != ReturnCode.OK:
            err = reply.get_header(MessageHeaderKey.ERROR, "")
            return f"{rc}: {err}"
        return reply.payload

    def _do_show_pool(self, request: Message) -> Union[None, Message]:
        p = request.payload
        assert isinstance(p, dict)
        pool_name = p.get("pool", "")
        mode = p.get("mode", "")
        pool = StatsPoolManager.get_pool(pool_name)
        if not pool:
            return new_message(
                headers={
                    MessageHeaderKey.RETURN_CODE: ReturnCode.INVALID_REQUEST,
                    MessageHeaderKey.ERROR: f"unknown pool '{pool_name}'",
                }
            )
        headers, rows = pool.get_table(mode)
        reply = {"headers": headers, "rows": rows}
        return new_message(payload=reply)

    def get_comm_config(self, target: str):
        reply = self.cell.send_request(
            channel=_CHANNEL, topic=_TOPIC_COMM_CONFIG, request=new_message(), timeout=1.0, target=target
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != ReturnCode.OK:
            err = reply.get_header(MessageHeaderKey.ERROR, "")
            return f"{rc}: {err}"
        return reply.payload

    def get_config_vars(self, target: str):
        reply = self.cell.send_request(
            channel=_CHANNEL, topic=_TOPIC_CONFIG_VARS, request=new_message(), timeout=1.0, target=target
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != ReturnCode.OK:
            err = reply.get_header(MessageHeaderKey.ERROR, "")
            return f"{rc}: {err}"
        return reply.payload

    def get_process_info(self, target: str):
        reply = self.cell.send_request(
            channel=_CHANNEL, topic=_TOPIC_PROCESS_INFO, request=new_message(), timeout=1.0, target=target
        )
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != ReturnCode.OK:
            err = reply.get_header(MessageHeaderKey.ERROR, "")
            return f"{rc}: {err}"
        return reply.payload

    def _do_comm_config(self, request: Message) -> Union[None, Message]:
        info = self.cell.connector_manager.get_config_info()
        return new_message(payload=info)

    def _do_config_vars(self, request: Message) -> Union[None, Message]:
        info = ConfigService.get_var_values()
        return new_message(payload=info)

    def _do_process_info(self, request: Message) -> Union[None, Message]:

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rows = [
            ["Process ID", str(os.getpid())],
            ["Memory Usage", str(usage)],
            ["Thread Count", str(threading.active_count())],
        ]

        for thread in threading.enumerate():
            rows.append([f"Thread:{thread.ident}", thread.name])

        return new_message(payload={"headers": ["Resource", "Value"], "rows": rows})

    def _broadcast_to_subs(self, topic: str, message=None, timeout=1.0):
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
                    channel=_CHANNEL, topic=topic, targets=targets, request=message, timeout=timeout
                )
            else:
                self.cell.fire_and_forget(channel=_CHANNEL, topic=topic, targets=targets, message=message)

        return {}
