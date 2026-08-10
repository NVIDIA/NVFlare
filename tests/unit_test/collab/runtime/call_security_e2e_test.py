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

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from nvflare.collab.api.app import ClientApp
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
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.utils.network_utils import get_open_ports


class _Victim:
    def __init__(self):
        self.call_count = 0

    @publish
    def mutate(self):
        self.call_count += 1
        return "mutated"


def _make_request(caller: str):
    return new_cell_message(
        {
            CallHeaderKey.PROTOCOL_VERSION: CALL_PROTOCOL_VERSION,
            CallHeaderKey.TARGET_NAME: "victim.client",
            CallHeaderKey.METHOD_NAME: "mutate",
        },
        {
            ObjectCallKey.CALLER: caller,
            ObjectCallKey.TARGET_NAME: "victim.client",
            ObjectCallKey.METHOD_NAME: "mutate",
        },
    )


@pytest.mark.timeout(30)
def test_raw_cross_job_cell_request_is_rejected_before_published_method_runs():
    port = get_open_ports(1)[0]
    suffix = uuid.uuid4().hex[:8]
    victim_fqcn = f"victim-job-{suffix}"
    attacker_fqcn = f"attacker-job-{suffix}"
    url = f"tcp://localhost:{port}"
    victim_cell = Cell(victim_fqcn, url, secure=False, credentials={})
    attacker_cell = Cell(attacker_fqcn, url, secure=False, credentials={})
    victim = _Victim()
    app = ClientApp(victim)
    app.name = "victim"
    executor = ThreadPoolExecutor(max_workers=1)

    prepare_for_remote_call(
        victim_cell,
        app,
        MagicMock(),
        executor,
        participants={victim_fqcn: "victim"},
    )
    victim_cell.core_cell.start()
    attacker_cell.core_cell.start()
    time.sleep(1.0)

    request = _make_request("server")

    try:
        started = time.monotonic()
        reply = attacker_cell.send_request(
            channel=MSG_CHANNEL,
            target=victim_fqcn,
            topic=MSG_TOPIC,
            request=request,
            timeout=5.0,
        )
        elapsed = time.monotonic() - started

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.COMM_ERROR
        assert reply.get_header(MessageHeaderKey.ERROR) == "Collab call rejected"
        assert elapsed < 4.0
        assert victim.call_count == 0
    finally:
        attacker_cell.core_cell.stop()
        victim_cell.core_cell.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        CoreCell.ALL_CELLS.pop(attacker_fqcn, None)
        CoreCell.ALL_CELLS.pop(victim_fqcn, None)


@pytest.mark.timeout(30)
def test_raw_same_job_cell_uses_authenticated_caller_and_preserves_valid_calls():
    port = get_open_ports(1)[0]
    suffix = uuid.uuid4().hex[:8]
    victim_fqcn = f"victim-job-{suffix}"
    participant_fqcn = f"participant-job-{suffix}"
    url = f"tcp://localhost:{port}"
    victim_cell = Cell(victim_fqcn, url, secure=False, credentials={})
    participant_cell = Cell(participant_fqcn, url, secure=False, credentials={})
    victim = _Victim()
    app = ClientApp(victim)
    app.name = "victim"
    executor = ThreadPoolExecutor(max_workers=1)

    prepare_for_remote_call(
        victim_cell,
        app,
        MagicMock(),
        executor,
        participants={victim_fqcn: "victim", participant_fqcn: "site-a"},
    )
    victim_cell.core_cell.start()
    participant_cell.core_cell.start()
    time.sleep(1.0)

    try:
        spoofed_reply = participant_cell.send_request(
            channel=MSG_CHANNEL,
            target=victim_fqcn,
            topic=MSG_TOPIC,
            request=_make_request("server"),
            timeout=5.0,
        )
        assert spoofed_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
        assert victim.call_count == 0

        valid_reply = participant_cell.send_request(
            channel=MSG_CHANNEL,
            target=victim_fqcn,
            topic=MSG_TOPIC,
            request=_make_request("site-a"),
            timeout=5.0,
        )
        assert valid_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert valid_reply.payload[CallReplyKey.RESULT] == "mutated"
        assert victim.call_count == 1
    finally:
        participant_cell.core_cell.stop()
        victim_cell.core_cell.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        CoreCell.ALL_CELLS.pop(participant_fqcn, None)
        CoreCell.ALL_CELLS.pop(victim_fqcn, None)
