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

"""The cell session: what a live task worker says to its Client Job.

The worker opens a Cell under the Client Job cell with the connect URL from its bootstrap. Every
request carries the launch token, the worker's only proof of launch: HELLO proves it before the
script runs, and each metric is a LOG request the Client Job answers once the metric is fired into
the job. The input, the result, ``completion.json``, the checkpoint and ``failure.json``
stay on disk: disk carries what must survive a process, the session only what two live processes say
to each other. The worker exits after its one attempt; the Client Job keeps the process poll as the
authority on the worker.
"""

import logging
import secrets
import threading
import time
import uuid

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.utils import fobs

CHANNEL = "task_worker"


class Topic:
    HELLO = "task_worker.hello"
    LOG = "task_worker.log"


_CONTROL_TIMEOUT = 30.0
_LOGGER = logging.getLogger(__name__)


def _check_reply(reply, what):
    rc = reply.get_header(MessageHeaderKey.RETURN_CODE) if reply is not None else None
    if rc != CellReturnCode.OK:
        error = reply.get_header(MessageHeaderKey.ERROR) if reply is not None else None
        raise RuntimeError(f"task worker {what} failed: rc={rc} {error or ''}".rstrip())
    return reply


def _reject(reason):
    return make_reply(CellReturnCode.INVALID_REQUEST, error=reason)


# ---------------------------------------------------------------- Client Job half

# Several TaskWorkerExecutors share one Client Job cell, whose callback registry holds one
# handler per topic; requests are routed to their session by the worker's FQCN.
_sessions = {}


def _dispatch(request, topic):
    session = _sessions.get(request.get_header(MessageHeaderKey.ORIGIN))
    if session is None:
        return _reject(f"unknown task worker {request.get_header(MessageHeaderKey.ORIGIN)!r}")
    return session.handle(topic, request)


def listen(cell):
    """Make the Client Job cell's loopback listener and route session requests; idempotent, once per executor."""
    resources = {
        DriverParams.HOST.value: "localhost",
        DriverParams.LISTEN_HOST.value: "127.0.0.1",
        DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.CLEAR,
    }
    cell.make_internal_listener(scheme="tcp", resources=resources)
    for topic in (Topic.HELLO, Topic.LOG):
        cell.register_request_cb(CHANNEL, topic, _dispatch, topic)


class CellWorkerSession:
    """Client Job half: one launched worker, identified by its FQCN and proven by its launch token."""

    def __init__(self, cell, on_analytics):
        self.cell = cell
        self.on_analytics = on_analytics
        self.token = secrets.token_urlsafe(32)
        self.worker_fqcn = FQCN.join([cell.get_fqcn(), f"task_worker_{uuid.uuid4().hex[:8]}"])
        # Requests arrive on the cell's frame pool; metrics share the attempt's FLContext, so fire them one at a
        # time, and never after close(): a request already waiting for the lock belongs to a settled attempt.
        # close() only sets the event, so a receiver that stalls never holds up the executor thread.
        self._lock = threading.Lock()
        self._closed = threading.Event()
        _sessions[self.worker_fqcn] = self

    def bootstrap_fields(self):
        connect_url = self.cell.get_internal_listener_url()
        if not connect_url:
            raise RuntimeError("Client Job cell has no internal listener for task workers to connect to")
        return {"connect_url": connect_url, "worker_fqcn": self.worker_fqcn, "token": self.token}

    def close(self):
        self._closed.set()
        if _sessions.get(self.worker_fqcn) is self:
            del _sessions[self.worker_fqcn]

    def handle(self, topic, request):
        payload = request.payload
        if isinstance(payload, (bytes, bytearray, memoryview)):
            payload = fobs.loads(bytes(payload))
        if not isinstance(payload, dict):
            return _reject("payload must be a dict")
        proof = payload.get("token")
        if not isinstance(proof, str) or not secrets.compare_digest(proof.encode(), self.token.encode()):
            return _reject("launch token mismatch")
        if topic == Topic.LOG:
            try:
                with self._lock:
                    if self._closed.is_set():
                        return _reject("session is closed")
                    self.on_analytics(payload.get("data"))
            except Exception as e:
                _LOGGER.error("rejecting task worker %s: %s", topic, e)
                return _reject(f"{type(e).__name__}: {e}")
        return make_reply(CellReturnCode.OK)


# ---------------------------------------------------------------- worker half


class CellWorkerTransport:
    """Worker half: a child Cell of the Client Job cell that carries analytics live."""

    def __init__(self, config):
        self.config = config
        self.cj_fqcn = FQCN.get_parent(config["worker_fqcn"])
        self._cell = None

    def open(self):
        self._cell = Cell(
            fqcn=self.config["worker_fqcn"],
            root_url=None,
            secure=False,
            credentials={},
            parent_url=self.config["connect_url"],
            create_internal_listener=False,
        )
        self._cell.start()
        deadline = time.monotonic() + _CONTROL_TIMEOUT
        while not self._cell.is_cell_connected(self.cj_fqcn):
            if time.monotonic() > deadline:
                raise RuntimeError(f"no connection to the Client Job after {_CONTROL_TIMEOUT}s")
            time.sleep(0.05)
        self._request(Topic.HELLO, {}, "HELLO")

    def _request(self, topic, payload, what):
        # One framed message on the parent connection, not a blob stream. Encoded natively here so array
        # values travel inline rather than as download references, which that receive path cannot resolve.
        encoded = fobs.dumps({"token": self.config["token"], **payload}, fobs_ctx={"native": True})
        reply = self._cell.core_cell.send_request(
            channel=CHANNEL,
            topic=topic,
            target=self.cj_fqcn,
            request=new_cell_message({}, encoded),
            timeout=_CONTROL_TIMEOUT,
        )
        return _check_reply(reply, what)

    def log(self, data):
        """One metric as a DXO Shareable; the Client Job answers after firing it, so a rejected one fails the attempt."""
        self._request(Topic.LOG, {"data": data}, "analytics")

    def close(self):
        if self._cell is not None:
            self._cell.stop()
