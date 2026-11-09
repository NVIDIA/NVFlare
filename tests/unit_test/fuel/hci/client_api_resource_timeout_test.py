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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.hci.client.api import AdminAPI
from nvflare.fuel.hci.client.api_spec import CommandContext
from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.proto import MetaKey, ProtoKey, validate_proto


@pytest.mark.parametrize(
    "command,explicit_timeout,expected_timeout",
    [
        (f"{AdminCommandNames.GET_JOB_RESOURCES} job-1 --site site-1", None, 300.0),
        (AdminCommandNames.GET_STUDY_RESOURCES, None, 300.0),
        (f"job.{AdminCommandNames.GET_STUDY_RESOURCES}", None, 300.0),
        (AdminCommandNames.LIST_JOBS, None, 5.0),
        (AdminCommandNames.GET_STUDY_RESOURCES, 12.5, 12.5),
        (f"{AdminCommandNames.GET_JOB_RESOURCES} job-1", 420.0, 420.0),
    ],
)
def test_resource_query_wait_is_operation_local_and_explicit_timeout_wins(command, explicit_timeout, expected_timeout):
    api = AdminAPI.__new__(AdminAPI)
    api.token = None
    api.cmd_timeout = explicit_timeout
    api.cell = SimpleNamespace(send_request=Mock(return_value=SimpleNamespace(payload=None)))
    context = CommandContext()
    context.set_command(command)
    context.set_command_args(split_to_args(command))
    context.set_json_processor(Mock())

    api._send_to_cell(context)

    sent = api.cell.send_request.call_args.kwargs
    assert sent["timeout"] == expected_timeout
    assert sent["channel"] == CellChannel.HCI
    assert sent["topic"] == "command"
    assert sent["target"] == FQCN.ROOT_SERVER
    assert api.cmd_timeout == explicit_timeout
    # Only a caller's explicit timeout enters command metadata. Choosing a
    # local default must not change the session or another server operation.
    request = validate_proto(sent["request"].payload)
    metadata = request.get(ProtoKey.META, {})
    if explicit_timeout is None:
        assert MetaKey.CMD_TIMEOUT not in metadata
    else:
        assert metadata[MetaKey.CMD_TIMEOUT] == explicit_timeout
