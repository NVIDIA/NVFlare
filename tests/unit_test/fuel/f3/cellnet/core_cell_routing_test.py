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

"""Endpoint-resolution invariants for ordinary hierarchical Cell names."""

import logging

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message


class _FakeAgent:
    def __init__(self, fqcn):
        self.endpoint = Endpoint(fqcn)


def _routing_cell(fqcn, connected):
    cell = CoreCell.__new__(CoreCell)
    cell.my_info = FqcnInfo(fqcn)
    cell.logger = logging.getLogger(__name__)
    cell.agents = {name: _FakeAgent(name) for name in connected}
    return cell


def test_ancestor_path_miss_does_not_fall_back_to_server_root():
    cell = _routing_cell("site-1", ["server"])

    assert cell._try_find_ep("site-1.job-dead.worker", None) is None


def test_regular_child_without_connection_does_not_fall_back_to_server_root():
    cell = _routing_cell("site-1", ["server"])

    assert cell._try_find_ep("site-1.job-dead", None) is None


def test_same_family_routing_prefers_fqcn_parent():
    cell = _routing_cell("site-1.job-123", ["site-1"])

    endpoint = cell._try_find_ep("site-1.other-job", None)

    assert endpoint is not None
    assert endpoint.name == "site-1"


def test_server_transit_required_bypasses_direct_cross_site_endpoint():
    cell = _routing_cell("site-1", ["server", "site-2"])
    message = Message(headers={MessageHeaderKey.SERVER_TRANSIT_REQUIRED: True})

    endpoint = cell._try_find_ep("site-2", message)

    assert endpoint is not None
    assert endpoint.name == "server"


def test_server_transit_required_job_cell_uses_local_parent():
    cell = _routing_cell("site-1.job-1", ["site-1", "site-2.job-2"])
    message = Message(headers={MessageHeaderKey.SERVER_TRANSIT_REQUIRED: True})

    endpoint = cell._try_find_ep("site-2.job-2", message)

    assert endpoint is not None
    assert endpoint.name == "site-1"


def test_find_endpoint_refuses_next_leg_already_on_route():
    cell = _routing_cell("server", ["site-1"])
    message = Message(headers={MessageHeaderKey.ROUTE: [("site-1", 0.0)]})

    rc, endpoint = cell._find_endpoint("site-1.job-dead", message)

    assert endpoint is None
    assert rc == ReturnCode.TARGET_UNREACHABLE


def test_find_endpoint_allows_final_destination_on_route():
    cell = _routing_cell("site-1", ["server"])
    message = Message(headers={MessageHeaderKey.ROUTE: [("server", 0.0)]})

    rc, endpoint = cell._find_endpoint("server", message)

    assert rc == ""
    assert endpoint is not None
    assert endpoint.name == "server"
