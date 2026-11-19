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

"""Harmless jobs used by the secure server-command authorization regression."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nvflare
from nvflare.apis.controller_spec import Task
from nvflare.apis.executor import Executor
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.collab.api.app import ClientApp
from nvflare.collab.api.context import get_call_context
from nvflare.collab.api.decorators import publish
from nvflare.collab.runtime.defs import (
    CALL_PROTOCOL_VERSION,
    MSG_CHANNEL,
    MSG_TOPIC,
    CallHeaderKey,
    CallReplyKey,
    ObjectCallKey,
)
from nvflare.collab.runtime.dispatch import prepare_for_remote_call
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey, MessagePropKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.private.defs import CellMessageHeaderKeys, new_cell_message


def record(fl_ctx, name, data):
    run = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()))
    path = run / name
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data))
    pending.replace(path)


def collab_call(cell, target, name, caller, method="mutate"):
    return cell.send_request(
        channel=MSG_CHANNEL,
        topic=MSG_TOPIC,
        target=target,
        request=new_cell_message(
            {
                CallHeaderKey.PROTOCOL_VERSION: CALL_PROTOCOL_VERSION,
                CallHeaderKey.TARGET_NAME: f"{name}.client",
                CallHeaderKey.METHOD_NAME: method,
            },
            {
                ObjectCallKey.CALLER: caller,
                ObjectCallKey.TARGET_NAME: f"{name}.client",
                ObjectCallKey.METHOD_NAME: method,
            },
        ),
        timeout=3,
    )


class CollabProbe:
    def __init__(self, cell, fl_ctx, participants):
        self.cell, self.fl_ctx, self.participants = cell, fl_ctx, participants
        self.callers = []

    @publish
    def mutate(self):
        self.callers.append(get_call_context().caller)
        record(self.fl_ctx, "collab_calls.json", self.callers)
        return "mutated"

    @publish
    def check_direct_auth(self):
        caller = get_call_context().caller
        target = next(fqcn for fqcn, name in self.participants.items() if name == caller)
        reply = self.cell.core_cell.send_request("auth_probe", "inspect", target, Message(), timeout=5)
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert reply.payload["peer"] == self.cell.get_fqcn()
        return not reply.payload["has_auth"]

    @publish
    def roundtrip(self):
        caller = get_call_context().caller
        target = next(fqcn for fqcn, name in self.participants.items() if name == caller)
        reply = collab_call(self.cell, target, caller, "server")
        assert reply.payload[CallReplyKey.RESULT] == "mutated"
        return "roundtrip"


def prepare_probe(cell, fl_ctx, participants, executor):
    cell.core_cell.register_request_cb(
        "auth_probe",
        "inspect",
        lambda msg: Message(
            payload={
                "peer": msg.get_prop(MessagePropKey.ENDPOINT).name,
                "has_auth": any(
                    msg.get_header(key)
                    for key in (
                        CellMessageHeaderKeys.CLIENT_NAME,
                        CellMessageHeaderKeys.TOKEN,
                        CellMessageHeaderKeys.TOKEN_SIGNATURE,
                        CellMessageHeaderKeys.SSID,
                    )
                ),
            }
        ),
    )
    probe = CollabProbe(cell, fl_ctx, participants)
    app = ClientApp(probe)
    app.name = participants[cell.get_fqcn()]
    prepare_for_remote_call(cell, app, probe_logger, executor, participants)
    return probe


probe_logger = logging.getLogger(__name__)


class ControlExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if task_name == "attack":
            cell = fl_ctx.get_engine().get_cell()
            victim = shareable["victim"]
            assert victim != fl_ctx.get_job_id()
            target = f"server.{victim}"
            reply = cell.send_request(
                channel=CellChannel.SERVER_COMMAND,
                topic="abort",
                target=target,
                request=new_cell_message({}, Shareable()),
                timeout=10,
            )
            # Obtain the headers from an ordinary authenticated parent reply, without accessing its key.
            parent_reply = cell.send_request(
                channel=CellChannel.SERVER_PARENT_LISTENER,
                topic="get_clients",
                target="server",
                request=new_cell_message({CellMessageHeaderKeys.JOB_ID: fl_ctx.get_job_id()}, None),
                timeout=10,
            )
            assert parent_reply.get_header(MessageHeaderKey.RETURN_CODE) == "ok"
            credentials = {
                key: parent_reply.get_header(key)
                for key in (
                    CellMessageHeaderKeys.CLIENT_NAME,
                    CellMessageHeaderKeys.TOKEN,
                    CellMessageHeaderKeys.TOKEN_SIGNATURE,
                )
            }
            assert all(value is None for value in credentials.values())
            # Model credentials captured from an older deployment. Only the test's
            # server job supplies these; ordinary client-directed replies must not.
            credentials = shareable["captured_parent_credentials"]

            def replay_parent(request):
                if (
                    request.get_header(StreamHeaderKey.CHANNEL) == CellChannel.SERVER_COMMAND
                    and request.get_header(StreamHeaderKey.TOPIC) == "abort"
                ):
                    request.add_headers(credentials)
                    request.set_header(MessageHeaderKey.ORIGIN, "server")

            cell.core_cell.add_outgoing_request_filter("*", "*", replay_parent)
            replay = cell.send_request(
                channel=CellChannel.SERVER_COMMAND,
                topic="abort",
                target=target,
                request=new_cell_message({}, Shareable()),
                timeout=3,
            )
            # The same old server credential must not authorize a forged job origin
            # on the Collab stream channel, which is separate from command handling.
            victim_client = shareable["victim_client"]

            def forge_collab(request):
                if request.get_header(StreamHeaderKey.CHANNEL) == MSG_CHANNEL:
                    request.add_headers(credentials)
                    request.set_header(MessageHeaderKey.ORIGIN, f"server.{victim}")

            cell.core_cell.add_outgoing_request_filter("*", "*", forge_collab)
            collab_reply = collab_call(cell, victim_client, "site-2", "server")
            assert collab_reply.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK
            # A sibling job on this site reaches its victim through the client
            # parent, without crossing the server's authentication filter.
            local_victim = f"{cell.get_fqcn().rsplit('.', 1)[0]}.{victim}"
            local_reply = collab_call(cell, local_victim, "site-1", "server")
            assert local_reply.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK
            result = {
                "local_collab_replay_rc": local_reply.get_header(MessageHeaderKey.RETURN_CODE),
                "collab_replay_rc": collab_reply.get_header(MessageHeaderKey.RETURN_CODE),
                "parent_reply_route": [entry[0] for entry in parent_reply.get_header(MessageHeaderKey.ROUTE)],
                "replay_rc": replay.get_header(MessageHeaderKey.RETURN_CODE),
                "origin": cell.get_fqcn(),
                "target": target,
                "attacker_job": fl_ctx.get_job_id(),
                "victim_job": victim,
                "rc": reply.get_header(MessageHeaderKey.RETURN_CODE),
                "source": nvflare.__file__,
            }
            record(fl_ctx, "attack.json", result)
            return Shareable(result)
        run = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()))
        cell = fl_ctx.get_engine().get_cell()
        with ThreadPoolExecutor(max_workers=2) as executor:
            prepare_probe(cell, fl_ctx, shareable["participants"], executor)
            reply = collab_call(
                cell, f"server.{fl_ctx.get_job_id()}", "server", fl_ctx.get_identity_name(), "roundtrip"
            )
            assert reply.payload[CallReplyKey.RESULT] == "roundtrip"
            if cell.core_cell.connector_manager.adhoc_allowed:
                server_job = f"server.{fl_ctx.get_job_id()}"
                deadline = time.monotonic() + 30
                while True:
                    agent = cell.core_cell.agents.get(server_job)
                    if agent and agent.endpoint.name == server_job:
                        break
                    assert time.monotonic() < deadline, "ad-hoc server-job connection not established"
                    time.sleep(0.1)
                direct_reply = cell.core_cell.send_request("auth_probe", "inspect", server_job, Message(), timeout=5)
                assert direct_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
                assert direct_reply.get_prop(MessagePropKey.ENDPOINT).name == server_job
                assert not any(
                    direct_reply.get_header(key)
                    for key in (
                        CellMessageHeaderKeys.CLIENT_NAME,
                        CellMessageHeaderKeys.TOKEN,
                        CellMessageHeaderKeys.TOKEN_SIGNATURE,
                        CellMessageHeaderKeys.SSID,
                    )
                )
                checked = collab_call(cell, server_job, "server", fl_ctx.get_identity_name(), "check_direct_auth")
                assert checked.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
                assert checked.payload[CallReplyKey.RESULT] is True
                record(fl_ctx, "adhoc_auth.json", {"request_no_auth": True, "reply_no_auth": True})
            if shareable.get("peer") and fl_ctx.get_identity_name() == "site-1":
                target = next(fqcn for fqcn, name in shareable["participants"].items() if name == "site-2")
                deadline = time.monotonic() + 30
                while True:
                    reply = collab_call(cell, target, "site-2", "site-1")
                    if reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK:
                        break
                    assert time.monotonic() < deadline, reply.headers
                    time.sleep(0.2)
                assert reply.payload[CallReplyKey.RESULT] == "mutated"
            record(fl_ctx, "holding.json", {"job": fl_ctx.get_job_id(), "pid": os.getpid(), "source": nvflare.__file__})
            while not abort_signal.triggered and not (run / "release").exists():
                time.sleep(0.1)
        return Shareable({"job": fl_ctx.get_job_id(), "aborted": abort_signal.triggered})


class ControlController(Controller):
    def __init__(self, victim=""):
        super().__init__()
        self.victim = victim

    def start_controller(self, fl_ctx):
        pass

    def control_flow(self, abort_signal, fl_ctx):
        with ThreadPoolExecutor(max_workers=2) as executor:
            self._control_flow(abort_signal, fl_ctx, executor)

    def _control_flow(self, abort_signal, fl_ctx, executor):
        cell = fl_ctx.get_engine().get_cell()
        participants = {f"server.{fl_ctx.get_job_id()}": "server"}
        participants.update(
            {f"{c.get_fqcn()}.{fl_ctx.get_job_id()}": c.name for c in fl_ctx.get_engine().get_clients()}
        )
        prepare_probe(cell, fl_ctx, participants, executor)
        record(fl_ctx, "running.json", {"job": fl_ctx.get_job_id(), "pid": os.getpid(), "source": nvflare.__file__})
        data = Shareable(
            {"victim": self.victim, "participants": participants, "peer": not self.victim and len(participants) == 3}
        )
        if self.victim:
            cell = fl_ctx.get_engine().get_cell()
            parent_reply = cell.send_request(
                CellChannel.SERVER_PARENT_LISTENER,
                "get_clients",
                "server",
                new_cell_message({CellMessageHeaderKeys.JOB_ID: fl_ctx.get_job_id()}, None),
                timeout=10,
            )
            data["victim_client"] = f"site-2.{self.victim}"
            data["captured_parent_credentials"] = {
                key: parent_reply.get_header(key)
                for key in (
                    CellMessageHeaderKeys.CLIENT_NAME,
                    CellMessageHeaderKeys.TOKEN,
                    CellMessageHeaderKeys.TOKEN_SIGNATURE,
                )
            }
            assert data["captured_parent_credentials"][CellMessageHeaderKeys.TOKEN_SIGNATURE]
        task = Task("attack" if self.victim else "hold", data, timeout=90, result_received_cb=self.received)
        self.broadcast_and_wait(task, fl_ctx, min_responses=len(participants) - 1, abort_signal=abort_signal)
        if self.victim:
            cell = fl_ctx.get_engine().get_cell()

            def claim_parent(request):
                if (
                    request.get_header(StreamHeaderKey.CHANNEL) == CellChannel.SERVER_COMMAND
                    and request.get_header(StreamHeaderKey.TOPIC) == "abort"
                ):
                    request.set_header(MessageHeaderKey.ORIGIN, "server")

            # Keep the job's real signed token; change only the claimed sender after transport stamping.
            cell.core_cell.add_outgoing_request_filter("*", "*", claim_parent)
            reply = cell.send_request(
                channel=CellChannel.SERVER_COMMAND,
                topic="abort",
                target=f"server.{self.victim}",
                request=new_cell_message({}, Shareable()),
                timeout=3,
            )
            record(
                fl_ctx,
                "forged_origin.json",
                {"sender": cell.get_fqcn(), "rc": reply.get_header(MessageHeaderKey.RETURN_CODE)},
            )

    def received(self, client_task, fl_ctx):
        record(fl_ctx, "received.json", {k: v for k, v in client_task.result.items() if k != ReservedHeaderKey.HEADERS})

    def stop_controller(self, fl_ctx):
        pass
