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

A worker reaches its Client Job exactly as the Client Job reaches its CP. The Client Job cell opens its
internal listener with the site's internal connection settings, as the CP does, and the worker connects as
a child cell with the Client Job's own credentials: in secure mode the per-job certificate, which is scoped
to the job cell and its descendants, so only this job's processes can join under this job's names.

Requests are routed to the attempt by the worker's FQCN. HELLO checks the connection before the script runs,
and each metric is a LOG request the Client Job answers once the metric is fired into the job. The input,
the result, ``completion.json``, the checkpoint and ``failure.json`` stay on disk. The worker exits after
its one attempt; the Client Job keeps the process poll as the authority on the worker.
"""

import threading
import time
import uuid

from nvflare.apis.fl_constant import ConnPropKey
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.utils import fobs
from nvflare.private.fed.resource_stats.message_traffic import snapshot_message_traffic

from .protocol import CELL_SESSION_CREDENTIALS

CHANNEL = "task_worker"


class Topic:
    HELLO = "task_worker.hello"
    LOG = "task_worker.log"


_CONTROL_TIMEOUT = 30.0
_CLOSE_TIMEOUT = 10.0


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


def _dispatch(request, session_topic):
    origin = request.get_header(MessageHeaderKey.ORIGIN)
    session = _sessions.get(origin)
    if session is None:
        return _reject(f"unknown task worker {origin!r}")
    return session.handle(session_topic, request)


# The site's internal settings name the CP's own listener address; the Client Job listens elsewhere.
_CP_ADDRESS = (DriverParams.HOST.value, DriverParams.PORT.value, DriverParams.LISTEN_HOST.value)


def listen(cell, host):
    """Open the Client Job cell's internal listener for its workers, and route session requests.

    The scheme and connection security are the site's internal settings, as for the CP's own listener, mTLS
    included when the site configures it. The address is not: the worker-reachable ``host`` comes from the task
    launcher, and the port is the Client Job's own. Idempotent: several executors share one Client Job cell.
    """
    manager = cell.core_cell.connector_manager
    resources = {key: value for key, value in manager.int_resources.items() if key not in _CP_ADDRESS}
    resources[DriverParams.HOST.value] = host
    cell.make_internal_listener(scheme=manager.int_scheme, resources=resources)
    for topic in (Topic.HELLO, Topic.LOG):
        cell.register_request_cb(channel=CHANNEL, topic=topic, cb=_dispatch, session_topic=topic)


class CellWorkerSession:
    """Client Job half: one launched worker, identified by its FQCN under the job cell."""

    def __init__(self, cell, on_analytics):
        self.cell = cell
        self.on_analytics = on_analytics
        self.worker_fqcn = FQCN.join([cell.get_fqcn(), f"task_worker_{uuid.uuid4().hex[:8]}"])
        # Requests arrive on the cell's frame pool; metrics share the attempt's FLContext, so fire them one at a
        # time, and never after close(): a request already waiting for the lock belongs to a settled attempt.
        self._lock = threading.Lock()
        self._closed = threading.Event()
        _sessions[self.worker_fqcn] = self

    def bootstrap_fields(self):
        """How to reach the Client Job, as the CP tells a Client Job how to reach it."""
        connect_url = self.cell.get_internal_listener_url()
        if not connect_url:
            raise RuntimeError("Client Job cell has no internal listener for task workers to connect to")
        core = self.cell.core_cell
        fields = {"connect_url": connect_url, "worker_fqcn": self.worker_fqcn}
        connection_security = (self.cell.get_internal_listener_params() or {}).get(ConnPropKey.CONNECTION_SECURITY)
        if connection_security:
            fields["connection_security"] = connection_security
        own = core.endpoint.conn_props
        credentials = {key: own[key] for key in CELL_SESSION_CREDENTIALS if own.get(key)}
        if credentials:
            fields["credentials"] = credentials
        identity = core.identity_resolver.resolve(self.cell.get_fqcn())
        if identity:
            fields["identity"] = identity
        return fields

    def close(self, timeout=_CLOSE_TIMEOUT):
        """Stop firing metrics; wait up to ``timeout`` for one already firing, so a stalled receiver is bounded."""
        self._closed.set()
        if _sessions.get(self.worker_fqcn) is self:
            del _sessions[self.worker_fqcn]
        if self._lock.acquire(timeout=timeout):
            self._lock.release()

    def handle(self, topic, request):
        if topic == Topic.LOG:
            # A rejection is reported in the reply; the worker raises it and records it in failure.json.
            try:
                data = fobs.loads(bytes(request.payload))
                with self._lock:
                    if self._closed.is_set():
                        return _reject("session is closed")
                    self.on_analytics(data)
            except Exception as e:
                return _reject(f"{type(e).__name__}: {e}")
        return make_reply(CellReturnCode.OK)


# ---------------------------------------------------------------- worker half


class CellWorkerTransport:
    """Worker half: a child cell of the Client Job cell, joined with the Client Job's own credentials."""

    def __init__(self, config):
        self.config = config
        self.cj_fqcn = FQCN.get_parent(config["worker_fqcn"])
        self._cell = None

    def open(self):
        connection_security = self.config.get("connection_security")
        identity = self.config.get("identity")
        # secure only governs external connections; the link to the Client Job is internal, and its TLS
        # follows connection_security and the credentials, as for the Client Job's link to the CP.
        self._cell = Cell(
            fqcn=self.config["worker_fqcn"],
            root_url=None,
            secure=False,
            credentials=dict(self.config.get("credentials") or {}),
            parent_url=self.config["connect_url"],
            parent_resources=(
                {DriverParams.CONNECTION_SECURITY.value: connection_security} if connection_security else None
            ),
            create_internal_listener=False,
            auth_identity=identity,
            auth_identity_map={self.cj_fqcn: identity} if identity else None,
        )
        self._cell.start()
        deadline = time.monotonic() + _CONTROL_TIMEOUT
        while not self._cell.is_cell_connected(self.cj_fqcn):
            if time.monotonic() > deadline:
                raise RuntimeError(f"no connection to the Client Job after {_CONTROL_TIMEOUT}s")
            time.sleep(0.05)
        self._request(Topic.HELLO, None, "HELLO")

    def _request(self, topic, payload, what):
        # One framed message on the parent connection, not a blob stream; the payload is FOBS-encoded natively
        # so array values travel inline rather than as download references that path cannot resolve.
        reply = self._cell.core_cell.send_request(
            channel=CHANNEL,
            topic=topic,
            target=self.cj_fqcn,
            request=new_cell_message({}, payload),
            timeout=_CONTROL_TIMEOUT,
        )
        return _check_reply(reply, what)

    def log(self, data):
        """Send one metric as a DXO Shareable and wait until the Client Job has fired it.

        A metric the Client Job rejects fails the attempt. A receiver that raises does not: event dispatch logs it,
        as for a resident executor in job lifetime.
        """
        self._request(Topic.LOG, fobs.dumps(data, fobs_ctx={"native": True}), "analytics")

    def close(self):
        if self._cell is not None:
            self._cell.stop()

    def traffic_snapshot(self, *, incomplete=False):
        """Return this endpoint's counters, including its normal close activity."""

        return snapshot_message_traffic(self._cell, incomplete=incomplete)
