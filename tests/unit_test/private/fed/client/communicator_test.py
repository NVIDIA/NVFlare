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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.f3.cellnet.core_cell import MessageHeaderKey, ReturnCode
from nvflare.private.defs import ClientRegMsgKey, new_cell_message
from nvflare.private.fed.client.communicator import Communicator


def test_get_site_config_for_registration_from_loaded_client_config():
    site_config = {"labels": {"region": "us-east"}}
    communicator = Communicator(client_config={"client_name": "site-1", ClientRegMsgKey.SITE_CONFIG: site_config})

    assert communicator._get_site_config_for_registration(FLContext()) == site_config


def test_get_site_config_for_registration_ignores_non_dict_config():
    communicator = Communicator(client_config={"client_name": "site-1", ClientRegMsgKey.SITE_CONFIG: ["bad"]})

    assert communicator._get_site_config_for_registration(FLContext()) is None


@pytest.mark.parametrize("acknowledgement", [True, False, None, "true"])
def test_submit_update_keeps_transport_status_separate_from_result_acceptance(monkeypatch, acknowledgement):
    communicator = Communicator(client_config={"client_name": "site-1"})
    fl_ctx = FLContextManager(identity_name="site-1", job_id="job-1").new_context()
    fl_ctx.set_prop(FLContextKey.SSID, "session", private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_ACCEPTED, True, private=True, sticky=False)
    response = Shareable()
    response.set_header(ReservedHeaderKey.TASK_ID, "task-1")
    response.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "attempt-1")
    if acknowledgement is not None:
        response.set_header(ReservedHeaderKey.TASK_RESULT_ACCEPTED, acknowledgement)
    communicator.cell = Mock()

    def send_request(**kwargs):
        kwargs["request"].set_header(MessageHeaderKey.PAYLOAD_LEN, 0)
        return new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, response)

    communicator.cell.send_request.side_effect = send_request
    monkeypatch.setattr("nvflare.private.fed.client.communicator.determine_parent_fqcn", lambda *_args: "server")
    submitted = Shareable()
    submitted.add_cookie(FLContextKey.TASK_ID, "task-1")
    submitted.add_cookie(FLContextKey.TASK_ATTEMPT_ID, "attempt-1")
    rc = communicator.submit_update("project", "token", "session", fl_ctx, "site-1", submitted, "train")
    assert rc == ReturnCode.OK
    expected = acknowledgement if isinstance(acknowledgement, bool) else None
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is expected


@pytest.mark.parametrize(
    "ack_task, ack_attempt", [("other-task", "attempt-1"), ("task-1", "old-attempt"), (None, None)]
)
def test_result_acceptance_ack_must_match_the_submitted_assignment_and_attempt(monkeypatch, ack_task, ack_attempt):
    communicator = Communicator(client_config={"client_name": "site-1"})
    fl_ctx = FLContextManager(identity_name="site-1", job_id="job-1").new_context()
    fl_ctx.set_prop(FLContextKey.SSID, "session", private=True, sticky=False)
    response = Shareable()
    response.set_header(ReservedHeaderKey.TASK_RESULT_ACCEPTED, True)
    response.set_header(ReservedHeaderKey.TASK_ID, ack_task)
    response.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, ack_attempt)
    communicator.cell = Mock()

    def send_request(**kwargs):
        kwargs["request"].set_header(MessageHeaderKey.PAYLOAD_LEN, 0)
        return new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, response)

    communicator.cell.send_request.side_effect = send_request
    monkeypatch.setattr("nvflare.private.fed.client.communicator.determine_parent_fqcn", lambda *_args: "server")
    submitted = Shareable()
    submitted.add_cookie(FLContextKey.TASK_ID, "task-1")
    submitted.add_cookie(FLContextKey.TASK_ATTEMPT_ID, "attempt-1")
    assert (
        communicator.submit_update("project", "token", "session", fl_ctx, "site-1", submitted, "train") == ReturnCode.OK
    )
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is None
