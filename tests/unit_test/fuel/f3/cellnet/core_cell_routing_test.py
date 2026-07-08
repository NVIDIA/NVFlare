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

"""Endpoint resolution for topology-shaped CellPipe FQCNs."""

import logging

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.endpoint import Endpoint


class _FakeAgent:
    def __init__(self, fqcn):
        self.endpoint = Endpoint(fqcn)


def _routing_cell(fqcn, connected):
    cell = CoreCell.__new__(CoreCell)
    cell.my_info = FqcnInfo(fqcn)
    cell.logger = logging.getLogger(__name__)
    cell.agents = {f: _FakeAgent(f) for f in connected}
    return cell


def test_pipe_cell_reaches_peer_through_connected_cp():
    cell = _routing_cell("site-1.cellpipe~plain~job-123~active", ["site-1"])

    ep = cell._try_find_ep("site-1.cellpipe~plain~job-123~passive", None)

    assert ep is not None
    assert ep.name == "site-1"


def test_pipe_cell_reaches_server_job_through_connected_cp():
    cell = _routing_cell("site-1.cellpipe~plain~job-123~active", ["site-1"])

    ep = cell._try_find_ep("server.job-123", None)

    assert ep is not None
    assert ep.name == "site-1"


def test_pipe_cell_reaches_peer_through_server_root():
    # With pipe_connect_type VIA_ROOT the pipe cell connects only to the
    # server root; the same-family peer must be routed through it.
    cell = _routing_cell("site-1.cellpipe~plain~job-123~active", ["server"])

    ep = cell._try_find_ep("site-1.cellpipe~plain~job-123~passive", None)

    assert ep is not None
    assert ep.name == "server"


def test_relay_alias_pipe_cell_reaches_peer_through_connected_relay():
    # A pipe cell behind a relay is named <relay>.cellpipe~alias~<site>~<token>~<mode>: its
    # FQCN parent is the connected relay, so normal parent routing applies.
    cell = _routing_cell("relay-1.cellpipe~alias~site-1~job-123~active", ["relay-1"])

    ep = cell._try_find_ep("relay-1.cellpipe~alias~site-1~job-123~passive", None)

    assert ep is not None
    assert ep.name == "relay-1"


def test_relay_alias_pipe_cell_reaches_server_job_through_connected_relay():
    cell = _routing_cell("relay-1.cellpipe~alias~site-1~job-123~active", ["relay-1"])

    ep = cell._try_find_ep("server.job-123", None)

    assert ep is not None
    assert ep.name == "relay-1"


def test_server_routes_to_pipe_cell_through_cp_not_cj():
    # NVBug 6423409: the server holds direct agents for both the site CP and
    # the CJ. A message addressed to the site's pipe cell (e.g. a swarm
    # cross-site download to the training subprocess) must route to the CP,
    # which the pipe cell actually connects to. Under the #4801 hierarchical
    # name site-1.job-123.active, longest-prefix resolution picked the CJ
    # (site-1.job-123), which has no connection to the pipe cell and dropped
    # the request.
    cell = _routing_cell("server", ["site-1", "site-1.job-123"])

    ep = cell._try_find_ep("site-1.cellpipe~plain~job-123~active", None)

    assert ep is not None
    assert ep.name == "site-1"


def test_pipe_cell_reaches_site_ancestor_through_connected_cp():
    cell = _routing_cell("site-1.cellpipe~plain~job-123~active", ["site-1"])

    ep = cell._try_find_ep("site-1", None)

    assert ep is not None
    assert ep.name == "site-1"


def test_same_family_routing_still_prefers_fqcn_parent():
    # A normal job cell connected to its parent keeps the original behavior.
    cell = _routing_cell("site-1.job-123", ["site-1"])

    ep = cell._try_find_ep("site-1.other-job", None)

    assert ep is not None
    assert ep.name == "site-1"


def test_pipe_cell_with_no_connection_is_unreachable():
    cell = _routing_cell("site-1.cellpipe~plain~job-123~active", [])

    assert cell._try_find_ep("site-1.cellpipe~plain~job-123~passive", None) is None
